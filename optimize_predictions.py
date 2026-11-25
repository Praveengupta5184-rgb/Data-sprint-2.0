#!/usr/bin/env python3
"""
No-training prediction optimization: sweep confidence thresholds and NMS settings
on the existing extended_best.pt model using official validation scoring.
"""
import os
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

# Paths
work_dir = Path(".").resolve()
data_yaml = Path("baseline_work/data.yaml").resolve()
val_images = Path("baseline_work/yolo_data/images/val").resolve()
val_labels_csv = Path("val_labels.csv").resolve()
val_image_ids_csv = Path("val_image_ids.csv").resolve()
weights_path = Path("weights/extended_best.pt").resolve()

# Output directory
opt_dir = Path("opt_results")
opt_dir.mkdir(exist_ok=True)

# Parameter sweep configurations - MINIMAL for quick test
CONF_VALUES = [0.001, 0.05, 0.1, 0.2]
IOU_VALUES = [0.5, 0.6, 0.65]
MAX_DET_VALUES = [100]

# Also test combinations around the current best (conf=0.001, iou=0.6)
# The official scorer finds optimal confidence threshold, so we mainly test NMS IoU
# and very low conf to not filter out potential detections


def run_inference(weights_path, output_csv, conf, iou, max_det):
    """Run inference on validation images with specified parameters."""
    model = YOLO(str(weights_path))
    IMGSZ = 640

    image_files = sorted(val_images.glob("*.jpg"))

    rows = []
    for img_path in image_files:
        image_id = img_path.stem
        img = np.asarray(Image.open(img_path).convert("RGB"))
        bgr = np.ascontiguousarray(img[:, :, ::-1])

        r = model.predict(bgr, imgsz=IMGSZ, conf=conf, iou=iou, max_det=max_det,
                          device="0", verbose=False)[0]

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for c, s, b in zip(cls, confs, xyxy):
                rows.append([image_id, int(c), float(s), float(b[0]), float(b[1]), float(b[2]), float(b[3])])

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "class_id", "score", "x_min", "y_min", "x_max", "y_max"])
        writer.writerows(rows)

    return len(rows)


def run_score(pred_csv, output_json):
    """Run official scorer on predictions."""
    cmd = [
        sys.executable, "score.py",
        "--pred", str(pred_csv),
        "--gt", str(val_labels_csv),
        "--image-ids", str(val_image_ids_csv),
        "--json-out", str(output_json)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Scoring failed: {result.stderr}")
        return None
    with open(output_json) as f:
        return json.load(f)


def main():
    print("="*80)
    print("PREDICTION PARAMETER OPTIMIZATION (no training)")
    print("="*80)
    print(f"Model: {weights_path}")
    print(f"Confidence thresholds: {CONF_VALUES}")
    print(f"NMS IoU thresholds: {IOU_VALUES}")
    print(f"Max detections: {MAX_DET_VALUES}")
    print(f"Total combinations: {len(CONF_VALUES) * len(IOU_VALUES) * len(MAX_DET_VALUES)}")
    print()

    results = []
    best_score = 0
    best_config = None

    for max_det in MAX_DET_VALUES:
        for iou in IOU_VALUES:
            for conf in CONF_VALUES:
                config_name = f"conf{conf}_iou{iou}_maxdet{max_det}"
                print(f"\n--- Testing {config_name} ---")

                pred_csv = opt_dir / f"val_pred_{config_name}.csv"
                score_json = opt_dir / f"score_{config_name}.json"

                # Run inference
                n_preds = run_inference(weights_path, pred_csv, conf, iou, max_det)
                print(f"  Predictions: {n_preds}")

                # Score
                score_result = run_score(pred_csv, score_json)
                if score_result is None:
                    continue

                m = score_result["main"]
                acc_score = score_result["combined"]["accuracy_score"]

                print(f"  Accuracy Score: {acc_score:.2f}")
                print(f"  mAP@50: {m['mAP50']:.4f}, mAP@50:95: {m['mAP50_95']:.4f}")
                print(f"  Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}")

                results.append({
                    "config": config_name,
                    "conf": conf,
                    "iou": iou,
                    "max_det": max_det,
                    "n_predictions": n_preds,
                    "accuracy_score": acc_score,
                    "mAP50": m["mAP50"],
                    "mAP50_95": m["mAP50_95"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "confidence_threshold": m["confidence_threshold"],
                    "per_class": m["per_class"],
                })

                if acc_score > best_score:
                    best_score = acc_score
                    best_config = config_name
                    print(f"  *** NEW BEST ***")

    # Sort by accuracy score
    results.sort(key=lambda x: x["accuracy_score"], reverse=True)

    # Print summary
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS (Top 10)")
    print("="*80)
    print(f"{'Config':<30} {'AccScore':>8} {'mAP50':>7} {'mAP50:95':>8} {'Prec':>6} {'Rec':>6} {'Preds':>6}")
    print("-"*80)
    for r in results[:10]:
        print(f"{r['config']:<30} {r['accuracy_score']:>8.2f} {r['mAP50']:>7.4f} {r['mAP50_95']:>8.4f} {r['precision']:>6.4f} {r['recall']:>6.4f} {r['n_predictions']:>6}")

    # Save full results
    with open(opt_dir / "optimization_results.json", "w") as f:
        json.dump({
            "best_config": best_config,
            "best_accuracy_score": best_score,
            "all_results": results
        }, f, indent=2)

    print(f"\nBest config: {best_config} (Accuracy Score: {best_score:.2f})")
    print(f"Full results saved to: {opt_dir / 'optimization_results.json'}")

    # Also test the current inference.py settings (conf=0.001, iou=0.6, max_det=100)
    print("\n" + "="*80)
    print("CURRENT INFERENCE.PY SETTINGS (conf=0.001, iou=0.6, max_det=100)")
    print("="*80)

    return results


if __name__ == "__main__":
    main()