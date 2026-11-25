#!/usr/bin/env python3
"""
Generate Kaggle submission using extended model.
"""
import os
import csv
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# Paths
extended_best = Path("weights/extended_best.pt").resolve()
test_images_dir = Path("baseline_work/yolo_data/images/test")  # Need to check actual test path
submission_csv = Path("submission_extended.csv").resolve()

# Check for test images - look at the original kit structure
kit_test_images = Path("kit/test_images")
if not kit_test_images.exists():
    # Try other locations
    kit_test_images = Path("test_images")
if not kit_test_images.exists():
    # Check what's in the baseline_work
    for p in Path(".").rglob("test_images"):
        if p.is_dir():
            kit_test_images = p
            break

print(f"Looking for test images in: {kit_test_images}")
if kit_test_images.exists():
    test_files = sorted(kit_test_images.glob("*.jpg"))
    print(f"Found {len(test_files)} test images")
else:
    print("Test images not found, checking available directories...")
    for p in Path(".").rglob("*.jpg"):
        print(f"  {p}")
    raise FileNotFoundError("Test images directory not found")

# Load model
model = YOLO(str(extended_best))
IMGSZ = 640

rows = []
for img_path in test_files:
    image_id = img_path.stem
    # Load and preprocess image
    img = np.asarray(Image.open(img_path).convert("RGB"))
    bgr = np.ascontiguousarray(img[:, :, ::-1])  # Ultralytics expects BGR

    # Predict
    r = model.predict(bgr, imgsz=IMGSZ, conf=0.001, iou=0.6, max_det=100,
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
print(f"Images with detections: {sum(1 for r in rows if r[1])}")

# Also copy to submission.csv as the main file
import shutil
shutil.copy2(submission_csv, "submission.csv")
print("Copied to submission.csv")