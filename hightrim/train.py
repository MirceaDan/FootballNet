import os
from PIL import Image
from scipy.optimize import linear_sum_assignment
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 320
BATCH_SIZE = 2
EPOCHS = 10
LR = 1e-4

NUM_QUERIES = 10
NUM_CLASSES = 2  # ball + no-object

ALPHA = 0.5
GAMMA = 0.9
EPS = 1e-6
LAMBDA_PRIOR = 1.0
BETA = 1.0

MODEL_SAVE_PATH = "./moca_bg_det.pth"

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
])

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
])

class YOLODataset(Dataset):
    def __init__(self, root):
        self.img_dir = os.path.join(root, "train")
        self.lbl_dir = os.path.join(root, "labels")
        self.images = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.images) - 1

    def load_labels(self, img_name):
        path = os.path.join(self.lbl_dir, img_name.replace(".jpg", ".txt"))

        boxes, labels = [], []

        if not os.path.exists(path):
            return torch.zeros((0,4)), torch.zeros((0,), dtype=torch.long)

        with open(path) as f:
            for line in f:
                cls, x, y, w, h = map(float, line.split())
                boxes.append([x,y,w,h])
                labels.append(int(cls))

        if len(boxes) == 0:
            return torch.zeros((0,4)), torch.zeros((0,), dtype=torch.long)

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels)

    def load_image(self, path):
        img = Image.open(path).convert("RGB")
        return transform(img)

    def __getitem__(self, idx):
        idx = idx + 1

        img_t_name = self.images[idx]
        img_prev_name = self.images[idx-1]

        img_t = self.load_image(os.path.join(self.img_dir, img_t_name))
        img_prev = self.load_image(os.path.join(self.img_dir, img_prev_name))

        boxes, labels = self.load_labels(img_t_name)

        return img_t, img_prev, {"boxes": boxes, "labels": labels}

    def collate_fn(batch):
    imgs = []
    prevs = []
    targets = []

    for img, prev, target in batch:
        imgs.append(img)
        prevs.append(prev)

        targets.append({
            "boxes": target["boxes"],
            "labels": target["labels"]
        })

    imgs = torch.stack(imgs)
    prevs = torch.stack(prevs)

    return imgs, prevs, targets
        
def build_dataloader(root):
    dataset = YOLODataset(root)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,64,3,2,1), nn.ReLU(),
            nn.Conv2d(64,128,3,2,1), nn.ReLU(),
            nn.Conv2d(128,256,3,2,1), nn.ReLU(),
            nn.Conv2d(256,512,3,2,1), nn.ReLU(),
        )

    def forward(self,x):
        return self.net(x)


class MotionModule(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x_t, x_prev):
        F_t = self.backbone(x_t)
        F_prev = self.backbone(x_prev)

        delta = F_t - GAMMA * F_prev

        M = torch.norm(delta, dim=1, keepdim=True)
        M = M / (M.amax(dim=(2,3), keepdim=True) + EPS)

        F_mod = F_t * (1 + ALPHA * M)

        return F_mod, M


class PriorHead(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C, C//2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(C//2,1,1)
        )

    def forward(self,x):
        return torch.sigmoid(self.net(x))


class PriorAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scale = d_model ** -0.5

    def forward(self, Q, K, V, P):
        B,_,H,W = P.shape
        P_flat = P.view(B,-1).unsqueeze(1)

        Q = Q.permute(1,0,2)
        K = K.permute(1,2,0)
        V = V.permute(1,0,2)

        attn = torch.bmm(Q,K) * self.scale
        attn = attn + LAMBDA_PRIOR * torch.log(EPS + P_flat)

        attn = torch.softmax(attn, dim=-1)

        out = torch.bmm(attn,V)
        return out.permute(1,0,2)

class MoCA_BG_DETR(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = CNNBackbone()
        self.motion = MotionModule(backbone)

        self.proj = nn.Conv2d(512,256,1)

        self.prior = PriorHead(256)

        self.query = nn.Embedding(NUM_QUERIES,256)

        self.attn = PriorAttention(256)

        self.cls = nn.Linear(256,NUM_CLASSES)
        self.box = nn.Linear(256,4)

    def forward(self,x_t,x_prev):
        F_mod, M = self.motion(x_t,x_prev)

        h = self.proj(F_mod)
        B,C,H,W = h.shape

        P_hat = self.prior(h)

        h_flat = h.flatten(2).permute(2,0,1)

        Q = self.query.weight.unsqueeze(1).repeat(1,B,1)

        hs = self.attn(Q, h_flat, h_flat, P_hat)

        logits = self.cls(hs)
        boxes = torch.sigmoid(self.box(hs))

        return logits, boxes, P_hat, M

class Matcher:
    def __call__(self, logits, boxes, targets):
        bs = logits.shape[0]
        indices = []

        for b in range(bs):
            prob = logits[b].softmax(-1)
            bbox = boxes[b]

            tgt_ids = targets[b]["labels"]
            tgt_box = targets[b]["boxes"]

            if len(tgt_ids) == 0:
                indices.append((torch.empty(0,dtype=torch.long), torch.empty(0,dtype=torch.long)))
                continue

            cost_class = -prob[:, tgt_ids]
            cost_bbox = torch.cdist(bbox, tgt_box, p=1)

            C = cost_class + 5 * cost_bbox
            C = C.detach().cpu().numpy()

            r,c = linear_sum_assignment(C)
            indices.append((torch.tensor(r), torch.tensor(c)))

        return indices


class Criterion:
    def __init__(self):
        self.matcher = Matcher()

    def __call__(self, outputs, targets):
        logits, boxes, P_hat, M = outputs
        indices = self.matcher(logits, boxes, targets)

        loss_cls, loss_box = 0, 0

        for b,(src,tgt) in enumerate(indices):
            if len(src)==0: continue
            loss_cls += F.cross_entropy(logits[b][src], targets[b]["labels"][tgt])
            loss_box += F.l1_loss(boxes[b][src], targets[b]["boxes"][tgt])

        loss_motion = F.binary_cross_entropy(P_hat, M)

        return loss_cls + loss_box + BETA * loss_motion

model = MoCA_BG_DETR().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = Criterion()

dataloader = build_dataloader("dataset")

def train_one_epoch():
    model.train()
    total_loss = 0

    for img, prev, target in dataloader:
        img = img.to(DEVICE)
        prev = prev.to(DEVICE)

        targets = [
            {
                "boxes": t["boxes"].to(DEVICE),
                "labels": t["labels"].to(DEVICE)
            }
            for t in target
        ]

        outputs = model(img, prev)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss

print("--- TRAINING STARTED ---")
for epoch in range(EPOCHS):
    start_time = time.time()
    loss = train_one_epoch() # loss = L_cls + L_box + β · L_motion, lower is better
    epoch_time = time.time() - start_time    
    print(f"Epoch {epoch}: Loss = {loss:.4f}:"
          f"{epoch_time/60:.2f} minutes")
print("--- TRAINING ENDED ---")

scripted = torch.jit.script(model)
scripted.save(MODEL_PATH)
print("Model saved.")
