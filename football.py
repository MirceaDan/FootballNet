from ultralytics import YOLO
import cv2
import numpy as np
import csv

# ================= CONFIG =================
INPUT_VIDEO = "input.avi"
OUTPUT_VIDEO = "output_full.avi"
OUTPUT_CSV = "trajectory.csv"

MODEL_WEIGHTS = "yolov8n.pt"
CONF_THRESHOLD = 0.3

BALL_CLASS_ID = 32

# camera approx
FOCAL_LENGTH = 800   # pixels
BALL_DIAMETER = 0.22  # meters

# ================= LOAD MODEL =================
model = YOLO(MODEL_WEIGHTS)

# ================= VIDEO IO =================
cap = cv2.VideoCapture(INPUT_VIDEO)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cx_img = width // 2
cy_img = height // 2

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width * 2, height))  # side-by-side

# ================= KALMAN FILTER =================
kf = cv2.KalmanFilter(6, 3)
kf.measurementMatrix = np.eye(3, 6, dtype=np.float32)
kf.transitionMatrix = np.array([
    [1,0,0,1,0,0],
    [0,1,0,0,1,0],
    [0,0,1,0,0,1],
    [0,0,0,1,0,0],
    [0,0,0,0,1,0],
    [0,0,0,0,0,1]
], np.float32)

kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1

# ================= DATA =================
trajectory_2d = []
trajectory_top = []

csv_file = open(OUTPUT_CSV, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame", "X", "Y", "Z"])

frame_id = 0

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    best_ball = None
    best_conf = 0

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls == BALL_CLASS_ID and conf > CONF_THRESHOLD:
            if conf > best_conf:
                best_conf = conf
                best_ball = box

    X = Y = Z = None

    if best_ball is not None:
        x1, y1, x2, y2 = map(int, best_ball.xyxy[0])

        u = (x1 + x2) // 2
        v = (y1 + y2) // 2

        d = max(x2 - x1, y2 - y1)

        if d > 0:
            # ===== DEPTH =====
            Z = (FOCAL_LENGTH * BALL_DIAMETER) / d

            # ===== 3D COORDS =====
            X = (u - cx_img) * Z / FOCAL_LENGTH
            Y = (v - cy_img) * Z / FOCAL_LENGTH

            measurement = np.array([[X], [Y], [Z]], dtype=np.float32)
            kf.correct(measurement)

            prediction = kf.predict()
            Xp, Yp, Zp = prediction[:3].flatten()

            # ===== SAVE =====
            csv_writer.writerow([frame_id, Xp, Yp, Zp])

            # ===== TRAJECTORY =====
            trajectory_2d.append((u, v))
            trajectory_top.append((Xp, Zp))

            # ===== DRAW DETECTION =====
            radius = int(d / 2)
            cv2.circle(frame, (u, v), radius, (0,255,0), 2)
            cv2.circle(frame, (u, v), 4, (0,0,255), -1)

            cv2.putText(frame, f"Z={Zp:.2f}m", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    else:
        kf.predict()

    # ===== DRAW TRAJECTORY (image space) =====
    for i in range(1, len(trajectory_2d)):
        cv2.line(frame, trajectory_2d[i-1], trajectory_2d[i], (255,0,0), 2)

    # ===== TOP VIEW MAP =====
    map_view = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(1, len(trajectory_top)):
        x_prev, z_prev = trajectory_top[i-1]
        x_curr, z_curr = trajectory_top[i]

        scale = 200  # px per meter

        px_prev = int(width//2 + x_prev * scale)
        py_prev = int(height - z_prev * scale)

        px_curr = int(width//2 + x_curr * scale)
        py_curr = int(height - z_curr * scale)

        cv2.line(map_view, (px_prev, py_prev), (px_curr, py_curr), (0,255,255), 2)

    # ===== CONCAT OUTPUT =====
    combined = np.hstack((frame, map_view))
    out.write(combined)

    cv2.imshow("Output", combined)
    if cv2.waitKey(1) == 27:
        break

    frame_id += 1

# ================= CLEANUP =================
cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()