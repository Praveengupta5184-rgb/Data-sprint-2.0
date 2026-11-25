#!/usr/bin/env python3
"""
Experiment 1: Fine-tune Extended Model (Extended v2)
Starting from weights/extended_best.pt (30 epochs), train additional 30 epochs
with lower LR, cosine scheduling, and stronger augmentation.
"""
import os
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    # Paths
    work_dir = Path("baseline_work").resolve()
    data_yaml = work_dir / "data.yaml"
    start_weights = Path("weights/extended_best.pt").resolve()
    project_dir = work_dir / "runs"
    exp_name = "exp1_finetune"

    # Verify
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")
    if not start_weights.exists():
        raise FileNotFoundError(f"Starting weights not found at {start_weights}")

    print(f"Starting from: {start_weights}")
    print(f"Data: {data_yaml}")
    print(f"Output: {project_dir / exp_name}")

    # Load model
    model = YOLO(str(start_weights))

    # Train with fine-tuning settings
    results = model.train(
        data=str(data_yaml),
        epochs=60,  # total epochs (resume from 30, so 30 more)
        imgsz=640,
        batch=16,  # fits RTX 3050 6GB
        device="0",
        workers=2,
        project=str(project_dir),
        name=exp_name,
        exist_ok=True,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.1,           # NEW: mixup for regularization
        copy_paste=0.1,      # NEW: copy-paste augmentation
        cutmix=0.0,
        degrees=10.0,        # INCREASED: rotation for road orientation
        translate=0.2,       # INCREASED: translation
        scale=0.9,           # INCREASED: scale range (0.1 to 1.9)
        shear=2.0,           # NEW: shear
        perspective=0.0001,  # NEW: slight perspective
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.4,         # random erasing
        auto_augment="randaugment",  # learned augmentation policy
        close_mosaic=15,     # disable mosaic last 15 epochs
        lr0=0.001,           # 10x lower LR for fine-tuning
        lrf=0.01,            # final LR factor
        cos_lr=True,         # cosine LR schedule
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        patience=15,         # early stopping patience
        plots=True,
        resume=False,        # not resuming, starting fresh from checkpoint
        amp=True,
        seed=0,
        deterministic=True,
        verbose=True,
    )

    # Copy best.pt to weights/exp1_finetune_best.pt
    exp1_best = model.trainer.best
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    exp1_weights = weights_dir / "exp1_finetune_best.pt"
    shutil.copy2(exp1_best, exp1_weights)
    print(f"\nSaved exp1 weights to: {exp1_weights}")

    # Also copy last.pt
    exp1_last = model.trainer.last
    exp1_last_weights = weights_dir / "exp1_finetune_last.pt"
    shutil.copy2(exp1_last, exp1_last_weights)
    print(f"Saved exp1 last weights to: {exp1_last_weights}")

    print(f"\nExperiment 1 complete. Results in: {project_dir / exp_name}")


if __name__ == "__main__":
    main()