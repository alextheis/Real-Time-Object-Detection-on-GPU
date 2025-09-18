# run_yolov5_batch.py

import torch
import glob
import os

# 1️⃣ Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 2️⃣ Get all image files in the folder
image_folder = "data/images/"
image_files = glob.glob(os.path.join(image_folder, "*.jpg"))  # add *.png if needed

# 3️⃣ Run detection on each image
for img_path in image_files:
    print(f"Processing {img_path}...")
    results = model(img_path)
    
    # 4️⃣ Print detection results
    results.print()
    
    # 5️⃣ Show image with bounding boxes (optional)
    # results.show()  # comment out if processing many images
    
    # 6️⃣ Save results
    results.save()  # saves to runs/detect/exp by default

    # 7️⃣ Optional: get detection data programmatically
    df = results.pandas().xyxy[0]
    print(df)
