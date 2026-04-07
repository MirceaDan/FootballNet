from ultralytics import YOLO
import cv2

# load pretrained model
model = YOLO("yolov8n.pt")  # rapid; poți folosi s/m pentru mai bun

cap = cv2.VideoCapture("input.avi")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # COCO class 32 = sports ball
        if cls == 32 and conf > 0.3:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.circle(frame, (cx,cy), 4, (0,0,255), -1)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()