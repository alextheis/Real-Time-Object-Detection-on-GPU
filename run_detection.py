# detect_single_image.py
import torch
import cv2

# 1. Set the path to your image
image_path = "data/images/bus.jpg"  # replace with your file name

# 2. Load the image
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# 3. Load a pre-trained YOLOv5 model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 4. Run object detection
results = model(img)

# 5. Show results (with bounding boxes)
results.show()  # Opens a window displaying the image with detected objects

# Optional: save the output image
results.save("data/images/output")  # saves to data/images/ou
