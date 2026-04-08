import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
from PIL import Image
from torchvision import transforms

# ================= CONFIG =================
DATASET_ROOT = "dataset"
IMG_SIZE = 224
EPOCHS = 20

IMG_DIR = os.path.join(DATASET_ROOT, "train")
LABEL_DIR = os.path.join(DATASET_ROOT, "labels")

# ================= FILE MATCHING =================
images = []

for file in os.listdir(IMG_DIR):
    if file.endswith(".jpg") or file.endswith(".png"):
        label_file = file.replace(".jpg", ".txt").replace(".png", ".txt")
        if os.path.exists(os.path.join(LABEL_DIR, label_file)):
            images.append(file)

random.shuffle(images)

split_idx = int(0.8 * len(images))
train_files = images[:split_idx]
val_files = images[split_idx:]

# ================= TRANSFORMS =================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ================= DATASET =================
class YoloDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]

        img_path = os.path.join(IMG_DIR, img_name)
        label_path = os.path.join(LABEL_DIR, img_name.replace(".jpg",".txt").replace(".png",".txt"))

        # ===== IMAGE (PIL) =====
        img = Image.open(img_path).convert("RGB")
        img = transform(img)

        # ===== LABEL =====
        with open(label_path) as f:
            lines = f.readlines()

        line = lines[0].strip().split()
        cls, xc, yc, w, h = map(float, line)

        target = torch.tensor([xc, yc, w, h, 1.0], dtype=torch.float32)

        return img, target

# ================= MODEL =================
class SimpleYOLO(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3,16,3,1,1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,1,1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.head = nn.Linear(64, 5)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.head(x)

# ================= TRAIN =================
device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = YoloDataset(train_files)
val_dataset = YoloDataset(val_files)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)

model = SimpleYOLO().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ================= TRAIN LOOP =================
for epoch in range(EPOCHS):
    print("--- TRAINING STARTED ---")
    model.train()
    train_loss = 0

    for imgs, targets in train_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        preds = model(imgs)

        loss_bbox = ((preds[:,:4] - targets[:,:4])**2).mean()
        loss_obj = nn.BCEWithLogitsLoss()(preds[:,4], targets[:,4])

        loss = loss_bbox + loss_obj

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ===== VALIDATION =====
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            preds = model(imgs)

            loss_bbox = ((preds[:,:4] - targets[:,:4])**2).mean()
            loss_obj = nn.BCEWithLogitsLoss()(preds[:,4], targets[:,4])

            val_loss += (loss_bbox + loss_obj).item()

    print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
print("--- TRAINING ENDED ---")

# ================= SAVE =================
scripted = torch.jit.script(model)
scripted.save("ball_yolo.pth")