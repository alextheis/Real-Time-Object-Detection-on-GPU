from ultralytics import YOLO
import torch, time

# Load pretrained YOLOv8n (nano) model
model = YOLO("yolov8n.pt")

# Test image
img = "https://ultralytics.com/images/bus.jpg"

# Warmup
model(img, device="cuda")

# Measure inference time
start = time.time()
results = model(img, device="cuda")
end = time.time()

# Show result
results[0].show()

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print(f"Inference time: {end - start:.3f} seconds")
print(f"FPS: {1/(end-start):.2f}")
