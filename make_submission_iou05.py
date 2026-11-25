#!/usr/bin/env python3
"""
Generate Kaggle submission using extended_best.pt with optimized IoU=0.5.
"""
import csv
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# Paths
weights_path = Path("weights/extended_best.pt").resolve()
test_images_dir = Path("test_images")
submission_csv = Path("submission_iou05.csv").resolve()
test_image_ids_csv = Path("test_image_ids.csv").resolve()

# Verify test images
test_files = sorted(test_images_dir.glob("*.jpg"))
print(f"Found {len(test_files)} test images")

# Verify test image IDs
with open(test_image_ids_csv) as f:
    test_ids = [line.strip() for line in f.readlines()[1:]]  # skip header
print(f"Test image IDs: {len(test_ids)}")

# Load model
model = YOLO(str(weights_path))
IMGSZ = 640

rows = []
for img_path in test_files:
    image_id = img_path.stem
    img = np.asarray(Image.open(img_path).convert("RGB"))
    bgr = np.ascontiguousarray(img[:, :, ::-1])

    r = model.predict(bgr, imgsz=IMGSZ, conf=0.001, iou=0.5, max_det=100,
                      device="0", verbose=False)[0]

    # Build PredictionString
    pred_parts = []
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        for c, s, b in zip(cls, conf, xyxy):
            pred_parts.append(f"{int(c)} {float(s):.5f} {float(b[0]):.2f} {float(b[1]):.2f} {float(b[2]):.2f} {float(b[3]):.2f}")

    prediction_string = " ".join(pred_parts) if pred_parts else "none"
    rows.append([image_id, prediction_string])

# Write Kaggle format submission
with open(submission_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "PredictionString"])
    writer.writerows(rows)

print(f"\nSaved submission to: {submission_csv}")
print(f"Total images: {len(rows)}")
images_with_dets = sum(1 for r in rows if r[1] != "none")
print(f"Images with detections: {images_with_dets}")
total_dets = sum(len(r[1].split()) // 6 for r in rows if r[1] != "none")
print(f"Total detections: {total_dets}")

# File size
import os
size_mb = os.path.getsize(submission_csv) / (1024 * 1024)
print(f"File size: {size_mb:.2f} MB")