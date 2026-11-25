#!/usr/bin/env python3
"""
Evaluate extended model on validation set using official scorer.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Paths
work_dir = Path("baseline_work").resolve()
data_yaml = work_dir / "data.yaml"
val_images = work_dir / "yolo_data" / "images" / "val"
val_labels_csv = Path("val_labels.csv").resolve()
val_image_ids_csv = Path("val_image_ids.csv").resolve()

# Extended model weights
extended_best = Path("weights/extended_best.pt").resolve()
baseline_best = Path("weights/best.pt").resolve()

# Output directories
val_pred_extended = Path("val_pred_extended.csv").resolve()
val_pred_baseline = Path("val_pred_baseline.csv").resolve()


def run_inference(weights_path, output_csv, model_name):
    """Run inference on validation images using the specified weights."""
    print(f"\n{'='*60}")
    print(f"Running inference with {model_name}: {weights_path}")
    print(f"{'='*60}")

    from ultralytics import YOLO
    import csv
    import numpy as np
    from PIL import Image

    model = YOLO(str(weights_path))
    IMGSZ = 640

    image_files = sorted(val_images.glob("*.jpg"))
    print(f"Found {len(image_files)} validation images")

    rows = []
    for img_path in image_files:
        image_id = img_path.stem
        # Load and preprocess image
        img = np.asarray(Image.open(img_path).convert("RGB"))
        bgr = np.ascontiguousarray(img[:, :, ::-1])  # Ultralytics expects BGR

        # Predict
        r = model.predict(bgr, imgsz=IMGSZ, conf=0.001, iou=0.6, max_det=100,
                          device="0", verbose=False)[0]

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for c, s, b in zip(cls, conf, xyxy):
                rows.append([image_id, int(c), float(s), float(b[0]), float(b[1]), float(b[2]), float(b[3])])

    # Write predictions
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "class_id", "score", "x_min", "y_min", "x_max", "y_max"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} predictions to {output_csv}")


def run_score(pred_csv, model_name):
    """Run official scorer on predictions."""
    print(f"\n{'='*60}")
    print(f"Scoring {model_name} predictions")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "score.py",
        "--pred", str(pred_csv),
        "--gt", str(val_labels_csv),
        "--image-ids", str(val_image_ids_csv),
        "--json-out", f"score_{model_name.lower()}.json"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0


def main():
    # Run inference with extended model
    run_inference(extended_best, val_pred_extended, "Extended")

    # Run inference with baseline model
    run_inference(baseline_best, val_pred_baseline, "Baseline")

    # Score both
    run_score(val_pred_extended, "Extended")
    run_score(val_pred_baseline, "Baseline")

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()