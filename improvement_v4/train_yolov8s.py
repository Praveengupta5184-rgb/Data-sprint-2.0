#!/usr/bin/env python3
"""
Improvement V4: Controlled 10-epoch training of YOLOv8s starting from the validated pilot checkpoint.
All outputs, weights, and logs are saved strictly inside improvement_v4/runs/exp_v8s_10ep.
"""
import os
import sys
import time
from pathlib import Path
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
IMPROVEMENT_DIR = ROOT_DIR / "improvement_v4"

DATA_YAML = IMPROVEMENT_DIR / "data.yaml"
PILOT_CKPT = ROOT_DIR / "baseline_work" / "runs" / "exp3_v8s_pilot" / "weights" / "best.pt"
RUNS_DIR = IMPROVEMENT_DIR / "runs"

def main():
    print("=" * 80, flush=True)
    print("STARTING YOLOv8s 10-EPOCH TRAINING EXPERIMENT (IMPROVEMENT V4)", flush=True)
    print("=" * 80, flush=True)

    if not PILOT_CKPT.exists():
        raise FileNotFoundError(f"Pilot checkpoint not found at: {PILOT_CKPT}")
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found at: {DATA_YAML}")

    print(f"Starting checkpoint: {PILOT_CKPT}", flush=True)
    print(f"Dataset config:     {DATA_YAML}", flush=True)
    print(f"Output directory:   {RUNS_DIR}", flush=True)

    model = YOLO(str(PILOT_CKPT))
    print(f"Loaded architecture: {model.task} | Classes: {len(model.names)} ({model.names})", flush=True)

    t0 = time.time()
    results = model.train(
        data=str(DATA_YAML),
        epochs=10,
        imgsz=640,
        batch=8,
        device="0",
        workers=2,
        amp=True,
        patience=100,  # Early stopping disabled for this short controlled experiment
        project=str(RUNS_DIR),
        name="exp_v8s_10ep",
        exist_ok=True,
        verbose=True,
        save=True,
        plots=True,
    )
    duration = time.time() - t0
    print(f"Training completed in {duration / 60:.2f} minutes.", flush=True)

    best_pt = RUNS_DIR / "exp_v8s_10ep" / "weights" / "best.pt"
    last_pt = RUNS_DIR / "exp_v8s_10ep" / "weights" / "last.pt"
    print(f"Best checkpoint saved: {best_pt} (exists: {best_pt.exists()})", flush=True)
    print(f"Last checkpoint saved: {last_pt} (exists: {last_pt.exists()})", flush=True)

if __name__ == "__main__":
    main()
