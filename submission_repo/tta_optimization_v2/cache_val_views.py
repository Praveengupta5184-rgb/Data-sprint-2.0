#!/usr/bin/env python3
"""
Precompute and cache raw multi-view detection predictions on all 2,150 validation images.
Views:
  - orig: imgsz=640, conf=0.0005, iou=0.7, max_det=300
  - flip: imgsz=640, conf=0.0005, iou=0.7, max_det=300 (x inverted correctly)
  - scale12: imgsz=768 (1.2x), conf=0.0005, iou=0.7, max_det=300
  - scale11: imgsz=704 (1.1x), conf=0.0005, iou=0.7, max_det=300
Saves:
  tta_optimization_v2/val_views_cache.pkl
"""
import os
import csv
import json
import time
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from score import load_ids

WEIGHTS_PATH = ROOT_DIR / "weights" / "extended_best.pt"
VAL_IMAGES_DIR = ROOT_DIR / "val_images"
VAL_IDS_CSV = ROOT_DIR / "val_image_ids.csv"
OUT_DIR = ROOT_DIR / "tta_optimization_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = OUT_DIR / "val_views_cache.pkl"

BASE_CONF = 0.0005
BASE_IOU = 0.70
BASE_MAX_DET = 300


def predict_view(model, bgr, imgsz):
    res = model.predict(
        bgr,
        imgsz=imgsz,
        conf=BASE_CONF,
        iou=BASE_IOU,
        max_det=BASE_MAX_DET,
        device="cuda:0",
        verbose=False,
    )[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    xyxy = res.boxes.xyxy.cpu().numpy()
    conf = res.boxes.conf.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy()
    return np.column_stack([cls, conf, xyxy]).astype(np.float32)


def main():
    print("=" * 70)
    print("MULTI-VIEW PREDICTION CACHING (Validation 2,150 images)")
    print(f"Model weights : {WEIGHTS_PATH}")
    print(f"Output cache  : {CACHE_FILE}")
    print("=" * 70)

    val_ids = load_ids(str(VAL_IDS_CSV))
    print(f"Loaded {len(val_ids)} validation image IDs")

    model = YOLO(str(WEIGHTS_PATH))

    # Warmup
    print("Warming up model...")
    sample_p = VAL_IMAGES_DIR / f"{val_ids[0]}.jpg"
    img = np.asarray(Image.open(sample_p).convert("RGB"))
    bgr = np.ascontiguousarray(img[:, :, ::-1])
    for sz in [640, 704, 768]:
        _ = predict_view(model, bgr, sz)

    cache = {}
    latencies = []
    t_start = time.perf_counter()

    for idx, iid in enumerate(val_ids, 1):
        img_p = VAL_IMAGES_DIR / f"{iid}.jpg"
        img = np.asarray(Image.open(img_p).convert("RGB"))
        bgr = np.ascontiguousarray(img[:, :, ::-1])
        H, W = bgr.shape[:2]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # 1. Original 640
        d_orig = predict_view(model, bgr, 640)

        # 2. Horizontal Flip 640 (invert x coordinates without in-place collision)
        bgr_flip = np.ascontiguousarray(bgr[:, ::-1, :])
        d_flip_raw = predict_view(model, bgr_flip, 640)
        if len(d_flip_raw):
            d_flip = d_flip_raw.copy()
            # x_orig_min = W - x_flip_max
            # x_orig_max = W - x_flip_min
            d_flip[:, 2] = np.clip(W - d_flip_raw[:, 4], 0.0, float(W))
            d_flip[:, 4] = np.clip(W - d_flip_raw[:, 2], 0.0, float(W))
            d_flip[:, 3] = np.clip(d_flip_raw[:, 3], 0.0, float(H))
            d_flip[:, 5] = np.clip(d_flip_raw[:, 5], 0.0, float(H))
        else:
            d_flip = np.zeros((0, 6), dtype=np.float32)

        # 3. Scale 1.2x (768)
        d_scale12 = predict_view(model, bgr, 768)

        # 4. Scale 1.1x (704)
        d_scale11 = predict_view(model, bgr, 704)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        cache[iid] = {
            "orig": d_orig,
            "flip": d_flip,
            "scale12": d_scale12,
            "scale11": d_scale11,
            "shape": (H, W),
        }

        if idx % 250 == 0 or idx == len(val_ids):
            elapsed = time.perf_counter() - t_start
            it_s = idx / elapsed
            print(f"  [{idx:4d}/{len(val_ids)}] {it_s:.1f} img/s | Avg: {np.mean(latencies):.2f} ms/img | Cache size: {len(cache)}")

    print(f"\nWriting cache to {CACHE_FILE}...")
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(CACHE_FILE) / 1e6
    print(f"Cache successfully saved: {size_mb:.2f} MB in {time.perf_counter() - t_start:.2f} s")


if __name__ == "__main__":
    main()
