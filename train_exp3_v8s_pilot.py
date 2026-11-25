#!/usr/bin/env python3
"""
Experiment 3 Pilot: YOLOv8s 3-epoch pilot at batch=4, imgsz=640
"""
import time
import torch
from pathlib import Path
from ultralytics import YOLO

def main():
    # Paths
    data_yaml = Path("baseline_work/data.yaml").resolve()
    project_dir = Path("baseline_work/runs").resolve()
    exp_name = "exp3_v8s_pilot"

    # Verify
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

    print("="*60)
    print("EXPERIMENT 3 PILOT: YOLOv8s 3-epoch @ batch=4, imgsz=640")
    print("="*60)
    print(f"Data: {data_yaml}")
    print(f"Output: {project_dir / exp_name}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Load COCO pretrained YOLOv8s
    model = YOLO("yolov8s.pt")

    # Record start time
    start_time = time.time()

    # Train for 3 epochs
    results = model.train(
        data=str(data_yaml),
        epochs=3,
        imgsz=640,
        batch=4,
        device="0",
        workers=2,
        project=str(project_dir),
        name=exp_name,
        exist_ok=True,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        degrees=10.0,
        translate=0.2,
        scale=0.9,
        shear=2.0,
        perspective=0.0001,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.4,
        auto_augment="randaugment",
        close_mosaic=15,
        lr0=0.01,
        lrf=0.01,
        cos_lr=True,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        patience=15,
        plots=True,
        amp=True,
        seed=0,
        deterministic=True,
        verbose=True,
    )

    # Record end time
    total_time = time.time() - start_time
    time_per_epoch = total_time / 3

    print("\n" + "="*60)
    print("PILOT COMPLETE")
    print("="*60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Time per epoch: {time_per_epoch/60:.1f} minutes")

    # Check GPU memory
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU memory allocated: {peak_mem:.2f} GB")

    # Print training results summary
    if hasattr(model.trainer, 'best'):
        print(f"Best weights: {model.trainer.best}")
    if hasattr(model.trainer, 'last'):
        print(f"Last weights: {model.trainer.last}")

    # Check results.csv
    results_csv = project_dir / exp_name / "results.csv"
    if results_csv.exists():
        print(f"\nResults saved to: {results_csv}")
        # Read and print last few lines
        with open(results_csv) as f:
            lines = f.readlines()
            print("Last epoch results:")
            for line in lines[-4:]:
                print(f"  {line.strip()}")

    # Check if we can try batch=8
    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        if peak_mem_gb < 4.0:
            print(f"\n✓ Batch size 8 appears SAFE (peak: {peak_mem_gb:.2f} GB < 4 GB)")
        elif peak_mem_gb < 5.0:
            print(f"\n⚠ Batch size 8 MARGINAL (peak: {peak_mem_gb:.2f} GB)")
        else:
            print(f"\n✗ Batch size 8 UNSAFE (peak: {peak_mem_gb:.2f} GB)")

    print(f"\nPilot artifacts in: {project_dir / exp_name}")

if __name__ == "__main__":
    main()