#!/usr/bin/env python3
"""
Benchmark speed of extended model vs baseline using official benchmark_speed.py.
Uses environment variable to select weights.
"""
import os
import subprocess
import sys
from pathlib import Path

# Paths
work_dir = Path(".").resolve()
val_images = Path("baseline_work/yolo_data/images/val").resolve()
extended_best = Path("weights/extended_best.pt").resolve()
baseline_best = Path("weights/best.pt").resolve()

# Original inference.py content
original_inference = Path("inference.py").read_text()


def create_inference(weights_path):
    """Create modified inference.py that uses specified weights."""
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


def run_benchmark(model_name, weights_path):
    """Run benchmark_speed.py with specified weights."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {model_name}")
    print(f"{'='*60}")

    # Create modified inference.py
    create_inference(str(weights_path))

    try:
        cmd = [
            sys.executable, "benchmark_speed.py",
            "--repo", str(work_dir),
            "--images", str(val_images),
            "--out-csv", f"speed_{model_name.lower()}_submission.csv",
            "--out-json", f"speed_{model_name.lower()}.json",
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
        return result.returncode == 0
    finally:
        # Restore original inference.py
        Path("inference.py").write_text(original_inference)


def main():
    print("Running speed benchmarks...")
    run_benchmark("Baseline", baseline_best)
    run_benchmark("Extended", extended_best)

    print("\n" + "="*60)
    print("SPEED BENCHMARK COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()