import os
import cv2
import torch
import pandas as pd

# -------------------------
# Config
# -------------------------
CONF_THRESHOLD = 0.5  # confidence threshold
VIDEO_OUTPUT_PATH = "data/videos/output.mp4"
CSV_OUTPUT_PATH = "runs/detect/webcam_detections.csv"

# Make sure folders exist
os.makedirs("data/videos", exist_ok=True)
os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)

# -------------------------
# Load YOLOv5 model
# -------------------------
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# -------------------------
# Open webcam
# -------------------------
cap = cv2.VideoCapture(0)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 20

# Video writer
out = cv2.VideoWriter(
    VIDEO_OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

# -------------------------
# Run detection loop
# -------------------------
all_detections = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = model(frame)

    # Render boxes on frame
    frame = results.render()[0]

    # Save frame to video
    out.write(frame)

    # Show live
    cv2.imshow("YOLOv5 Webcam Detection", frame)

    # Log detections
    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        if conf >= CONF_THRESHOLD:
            x_min, y_min, x_max, y_max = box
            all_detections.append({
                "xmin": x_min,
                "ymin": y_min,
                "xmax": x_max,
                "ymax": y_max,
                "confidence": conf,
                "class": int(cls),
                "name": model.names[int(cls)]
            })

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------
# Cleanup
# -------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

# Save detections to CSV
if all_detections:
    df = pd.DataFrame(all_detections)
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"Detections saved to {CSV_OUTPUT_PATH}")

print(f"Video saved to {VIDEO_OUTPUT_PATH}")
