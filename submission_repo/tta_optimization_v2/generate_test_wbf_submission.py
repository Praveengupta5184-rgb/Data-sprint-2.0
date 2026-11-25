#!/usr/bin/env python3
"""
Generate test set predictions and Kaggle submission using the winning
Weighted Boxes Fusion (WBF) configuration:
  - Views: Original (640), Horizontal Flip (640, inverted), Scale 1.2x (768)
  - Fusion: Weighted Boxes Fusion with IoU=0.45, conf_type='max', conf_thr=0.001
  - Max detections: 100
Validated on all 2,150 validation images:
  Accuracy Score: 43.53 (+0.93 points over 42.60 baseline)
  mAP@50: 0.5230 (+0.0135 over 0.5095)
Saves:
  tta_optimization_v2/test_pred_wbf.csv
  tta_optimization_v2/submission_wbf_optimized.csv
"""
import os
import csv
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from score import load_ids

WEIGHTS_PATH = ROOT_DIR / "weights" / "extended_best.pt"
TEST_IMAGES_DIR = ROOT_DIR / "test_images"
TEST_IDS_CSV = ROOT_DIR / "test_image_ids.csv"
OUT_DIR = ROOT_DIR / "tta_optimization_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_COLS = ["image_id", "class_id", "score", "x_min", "y_min", "x_max", "y_max"]
MAX_DETS = 100
WBF_IOU = 0.45
CONF_THR = 0.001


def predict_view(model, bgr, imgsz):
    res = model.predict(
        bgr,
        imgsz=imgsz,
        conf=CONF_THR,
        iou=0.7,
        max_det=300,
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
    print("WINNING WBF TEST SET PREDICTION & SUBMISSION GENERATION")
    print(f"Model weights : {WEIGHTS_PATH}")
    print(f"Test dir      : {TEST_IMAGES_DIR}")
    print(f"Config        : WBF (IoU={WBF_IOU}, conf_type='max', 3 views: 640 + flip + 768)")
    print("=" * 70)

    model = YOLO(str(WEIGHTS_PATH))
    test_ids = load_ids(str(TEST_IDS_CSV))
    print(f"Loaded {len(test_ids)} test image IDs")

    test_pred_rows = []
    submission_rows = []
    total_boxes = 0

    start_total = time.perf_counter()
    latencies = []

    for idx, iid in enumerate(test_ids, 1):
        img_p = TEST_IMAGES_DIR / f"{iid}.jpg"
        img = np.asarray(Image.open(img_p).convert("RGB"))
        bgr = np.ascontiguousarray(img[:, :, ::-1])
        H, W = bgr.shape[:2]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # View 1: Original 640
        d_orig = predict_view(model, bgr, 640)

        # View 2: Horizontal Flip 640
        bgr_flip = np.ascontiguousarray(bgr[:, ::-1, :])
        d_flip_raw = predict_view(model, bgr_flip, 640)
        if len(d_flip_raw):
            d_flip = d_flip_raw.copy()
            d_flip[:, 2] = np.clip(W - d_flip_raw[:, 4], 0.0, float(W))
            d_flip[:, 4] = np.clip(W - d_flip_raw[:, 2], 0.0, float(W))
            d_flip[:, 3] = np.clip(d_flip_raw[:, 3], 0.0, float(H))
            d_flip[:, 5] = np.clip(d_flip_raw[:, 5], 0.0, float(H))
        else:
            d_flip = np.zeros((0, 6), dtype=np.float32)

        # View 3: Scale 1.2x (768)
        d_scale12 = predict_view(model, bgr, 768)

        # Normalize and run WBF
        views = [d_orig, d_flip, d_scale12]
        boxes_list, scores_list, labels_list = [], [], []
        has_dets = False

        for d in views:
            if len(d) > 0:
                mask = d[:, 1] >= CONF_THR
                if np.any(mask):
                    d_filt = d[mask].copy()
                    d_filt[:, 2] = np.clip(d_filt[:, 2], 0.0, W - 1.0)
                    d_filt[:, 3] = np.clip(d_filt[:, 3], 0.0, H - 1.0)
                    d_filt[:, 4] = np.clip(d_filt[:, 4], d_filt[:, 2] + 1.0, float(W))
                    d_filt[:, 5] = np.clip(d_filt[:, 5], d_filt[:, 3] + 1.0, float(H))

                    b_norm = d_filt[:, 2:6].copy()
                    b_norm[:, [0, 2]] = np.clip(b_norm[:, [0, 2]] / W, 0.0, 1.0)
                    b_norm[:, [1, 3]] = np.clip(b_norm[:, [1, 3]] / H, 0.0, 1.0)
                    b_norm[:, 2] = np.maximum(b_norm[:, 2], b_norm[:, 0] + 1e-4)
                    b_norm[:, 3] = np.maximum(b_norm[:, 3], b_norm[:, 1] + 1e-4)

                    boxes_list.append(b_norm.tolist())
                    scores_list.append(d_filt[:, 1].tolist())
                    labels_list.append(d_filt[:, 0].astype(int).tolist())
                    has_dets = True
                else:
                    boxes_list.append([])
                    scores_list.append([])
                    labels_list.append([])
            else:
                boxes_list.append([])
                scores_list.append([])
                labels_list.append([])

        if has_dets:
            try:
                f_boxes, f_scores, f_labels = weighted_boxes_fusion(
                    boxes_list,
                    scores_list,
                    labels_list,
                    weights=[1.0, 1.0, 1.0],
                    iou_thr=WBF_IOU,
                    skip_box_thr=CONF_THR,
                    conf_type="max",
                )
            except Exception:
                f_boxes, f_scores, f_labels = np.zeros((0, 4)), np.zeros(0), np.zeros(0)
        else:
            f_boxes, f_scores, f_labels = np.zeros((0, 4)), np.zeros(0), np.zeros(0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

        if len(f_boxes) > 0:
            f_boxes[:, [0, 2]] *= W
            f_boxes[:, [1, 3]] *= H
            res = np.column_stack([f_labels, f_scores, f_boxes])
            order = np.argsort(-res[:, 1], kind="stable")[:MAX_DETS]
            res = res[order]
        else:
            res = np.zeros((0, 6))

        # Store for test_pred_wbf.csv
        for c, s, x1, y1, x2, y2 in res:
            test_pred_rows.append([
                iid, int(c), f"{s:.5f}", f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}"
            ])

        # Store for submission_wbf_optimized.csv
        parts = []
        for c, s, x1, y1, x2, y2 in res:
            parts.append(f"{int(c)} {s:.5f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}")
        pred_string = " ".join(parts) if parts else "none"
        submission_rows.append([iid, pred_string])
        total_boxes += len(res)

        if idx % 500 == 0 or idx == len(test_ids):
            elapsed = time.perf_counter() - start_total
            print(f"  [{idx:4d}/{len(test_ids)}] {idx/elapsed:.1f} img/s | Med: {np.median(latencies):.2f} ms | Boxes: {total_boxes}")

    # Write test_pred_wbf.csv
    test_pred_csv = OUT_DIR / "test_pred_wbf.csv"
    with open(test_pred_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PRED_COLS)
        w.writerows(test_pred_rows)
    print(f"\nSaved detailed test predictions to: {test_pred_csv}")

    # Write submission_wbf_optimized.csv
    sub_csv = OUT_DIR / "submission_wbf_optimized.csv"
    with open(sub_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "PredictionString"])
        w.writerows(submission_rows)
    print(f"Saved Kaggle submission to: {sub_csv} ({len(test_ids)} images, {total_boxes} boxes)")

    # Save test speed profile
    test_speed = {
        "status": "OK",
        "n_images": len(test_ids),
        "median_ms": round(float(np.median(latencies)), 3),
        "mean_ms": round(float(np.mean(latencies)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "total_seconds": round(time.perf_counter() - start_total, 2),
    }
    with open(OUT_DIR / "speed_test_wbf.json", "w") as f:
        json.dump(test_speed, f, indent=2)
    print(f"Saved speed metrics to: {OUT_DIR / 'speed_test_wbf.json'}")


if __name__ == "__main__":
    main()
