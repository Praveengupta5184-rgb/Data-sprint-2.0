#!/usr/bin/env python3
"""
Evaluate two models on official validation set using official scorer.
"""
import os
import csv
import json
import shutil
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

# Models to evaluate
models = {
    "extended": Path("weights/extended_best.pt").resolve(),
    "exp1_finetune": Path("baseline_work/runs/exp1_finetune/weights/best.pt").resolve(),
}

# Output directories
val_pred_dir = Path("eval_results")
val_pred_dir.mkdir(exist_ok=True)


def run_inference(weights_path, output_csv, model_name):
    """Run inference on validation images using the specified weights."""
    print(f"\n{'='*60}")
    print(f"Running inference with {model_name}: {weights_path}")
    print(f"{'='*60}")

    model = YOLO(str(weights_path))
    IMGSZ = 640

    image_files = sorted(val_images.glob("*.jpg"))
    print(f"Found {len(image_files)} validation images")

    rows = []
    for img_path in image_files:
        image_id = img_path.stem
        img = np.asarray(Image.open(img_path).convert("RGB"))
        bgr = np.ascontiguousarray(img[:, :, ::-1])

        r = model.predict(bgr, imgsz=IMGSZ, conf=0.001, iou=0.6, max_det=100,
                          device="0", verbose=False)[0]

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for c, s, b in zip(cls, conf, xyxy):
                rows.append([image_id, int(c), float(s), float(b[0]), float(b[1]), float(b[2]), float(b[3])])

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "class_id", "score", "x_min", "y_min", "x_max", "y_max"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} predictions to {output_csv}")
    return len(rows)


def run_score(pred_csv, model_name):
    """Run official scorer on predictions."""
    print(f"\n{'='*60}")
    print(f"Scoring {model_name} predictions")
    print(f"{'='*60}")

    json_out = val_pred_dir / f"score_{model_name}.json"
    cmd = [
        sys.executable, "score.py",
        "--pred", str(pred_csv),
        "--gt", str(val_labels_csv),
        "--image-ids", str(val_image_ids_csv),
        "--json-out", str(json_out)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Load and return the JSON result
    with open(json_out) as f:
        return json.load(f)


def run_speed_benchmark(model_name, weights_path):
    """Run benchmark_speed.py with specified weights."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {model_name}")
    print(f"{'='*60}")

    # Create modified inference.py
    original_inference = Path("inference.py").read_text()
    content = f'''"""
inference.py : the contract every team's repo must follow (this file is also the starter baseline).

The organisers' benchmark_speed.py imports this file from the ROOT of your repo and calls, in this order:

    model = load(device)                 # once, not timed. device is "cuda" or "cpu"
    detections = predict(model, image)   # once per image, timed, batch size 1

  image       : numpy array, shape [H, W, 3], dtype uint8, colour order RGB, the original image size
  detections  : a list of tuples (class_id, score, x_min, y_min, x_max, y_max)
                class_id 0 D00, 1 D10, 2 D20, 3 D40; score 0-1; box in pixels of the ORIGINAL image.
                Return low-confidence detections too (score >= 0.001). At most 100 are used.

Rules for this file: everything needed (code and weights) is in your repo; no internet access, no calls to
outside APIs; load weights from a path relative to this file. Test-time augmentation, ensembles and tiling
are allowed inside predict(), and their time is counted.

This baseline fine-tunes an Ultralytics YOLO model; replace the body of load() / predict() with your own.
"""
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = r"{weights_path}"
IMGSZ = 640


class _Model:
    def __init__(self, yolo, device):
        self.yolo, self.device = yolo, device


def load(device="cuda"):
    from ultralytics import YOLO
    return _Model(YOLO(WEIGHTS), device)


def predict(model, image):
    bgr = np.ascontiguousarray(image[:, :, ::-1])  # Ultralytics treats numpy arrays as BGR
    r = model.yolo.predict(bgr, imgsz=IMGSZ, conf=0.001, iou=0.6, max_det=100,
                           device=model.device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    return [(int(c), float(s), float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            for c, s, b in zip(cls, conf, xyxy)]
'''
    Path("inference.py").write_text(content)

    try:
        json_out = val_pred_dir / f"speed_{model_name}.json"
        cmd = [
            sys.executable, "benchmark_speed.py",
            "--repo", str(work_dir),
            "--images", str(val_images),
            "--out-csv", str(val_pred_dir / f"speed_{model_name}_submission.csv"),
            "--out-json", str(json_out),
            "--device", "cuda",
            "--warmup", "20",
            "--runs", "3",
            "--repeat-images", "300",
            "--max-minutes", "30"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        with open(json_out) as f:
            return json.load(f)
    finally:
        Path("inference.py").write_text(original_inference)


def main():
    results = {}

    for model_name, weights_path in models.items():
        print(f"\n{'#'*60}")
        print(f"# EVALUATING: {model_name.upper()}")
        print(f"#{'#'*60}")

        pred_csv = val_pred_dir / f"val_pred_{model_name}.csv"

        # 1. Inference
        run_inference(weights_path, pred_csv, model_name)

        # 2. Official scoring
        score_result = run_score(pred_csv, model_name)

        # 3. Speed benchmark
        speed_result = run_speed_benchmark(model_name, weights_path)

        results[model_name] = {
            "score": score_result,
            "speed": speed_result,
        }

    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON REPORT")
    print("="*80)

    for model_name in models:
        r = results[model_name]
        m = r["score"]["main"]
        s = r["speed"]
        print(f"\n--- {model_name.upper()} ---")
        print(f"Accuracy Score: {r['score']['combined']['accuracy_score']:.2f}")
        print(f"mAP@50:         {m['mAP50']:.4f}")
        print(f"mAP@50:95:      {m['mAP50_95']:.4f}")
        print(f"Precision:      {m['precision']:.4f}")
        print(f"Recall:         {m['recall']:.4f}")
        print(f"F1:             {m['f1']:.4f}")
        print(f"Conf Threshold: {m['confidence_threshold']:.3f}")
        for name, v in m["per_class"].items():
            print(f"  {name:<28} AP50 {v['AP50']:.4f}   AP50:95 {v['AP50_95']:.4f}")
        print(f"Speed - Median: {s['median_ms']:.1f} ms, Speed Score: {s['speed_score']:.4f}")
        print(f"Peak GPU Mem:   {s.get('peak_gpu_mem_mb', 'N/A')} MB")

    # Save full comparison
    comparison_path = val_pred_dir / "comparison_report.json"
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull comparison saved to: {comparison_path}")

    return results


if __name__ == "__main__":
    main()