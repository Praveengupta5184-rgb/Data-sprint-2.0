#!/usr/bin/env python3
"""
Extended YOLOv8n training: continue from baseline for ~25 additional epochs (total ~30).

Resumes from baseline_work/runs/baseline/weights/last.pt if available,
otherwise initializes from best.pt.
Saves results in a separate experiment directory: baseline_work/runs/extended
"""
import os
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    # Paths
    work_dir = Path("baseline_work").resolve()
    data_yaml = work_dir / "data.yaml"
    last_pt = work_dir / "runs" / "baseline" / "weights" / "last.pt"
    best_pt = work_dir / "runs" / "baseline" / "weights" / "best.pt"
    project_dir = work_dir / "runs"
    exp_name = "extended"

    # Verify data.yaml exists
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

    # Determine resume source
    if last_pt.exists():
        resume_from = str(last_pt)
        print(f"Resuming from last.pt: {resume_from}")
    elif best_pt.exists():
        resume_from = str(best_pt)
        print(f"Initializing from best.pt: {resume_from}")
    else:
        raise FileNotFoundError("Neither last.pt nor best.pt found in baseline run")

    # Load model
    model = YOLO(resume_from)

    # Train for additional epochs (total ~30, so 25 more from epoch 5)
    # Using resume=True will continue from the epoch in the checkpoint
    results = model.train(
        data=str(data_yaml),
        epochs=30,  # total epochs (will resume from epoch 5, so ~25 more)
        imgsz=640,
        batch=16,  # fits RTX 3050 6GB
        device="0",
        workers=2,
        project=str(project_dir),
        name=exp_name,
        exist_ok=True,
        fliplr=0.5,
        flipud=0.0,
        patience=10,  # early stopping patience
        plots=True,  # enable plots for extended run
        resume=True,  # resume from checkpoint
        amp=True,
        cos_lr=False,
        close_mosaic=10,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        auto_augment="randaugment",
        erasing=0.4,
    )

    # Copy best.pt to weights/extended_best.pt for easy access
    extended_best = model.trainer.best
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    extended_weights = weights_dir / "extended_best.pt"
    shutil.copy2(extended_best, extended_weights)
    print(f"\nSaved extended weights to: {extended_weights}")

    # Also copy last.pt
    extended_last = model.trainer.last
    extended_last_weights = weights_dir / "extended_last.pt"
    shutil.copy2(extended_last, extended_last_weights)
    print(f"Saved extended last weights to: {extended_last_weights}")

    print(f"\nExtended training complete. Results in: {project_dir / exp_name}")


if __name__ == "__main__":
    main()