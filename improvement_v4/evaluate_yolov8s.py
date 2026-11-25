#!/usr/bin/env python3
"""
Evaluate YOLOv8s (10-epoch trained checkpoint) on all 2,150 validation images:
1. YOLOv8s alone (single-view inference).
2. YOLOv8s + 3-View TTA (Original + Horizontal Flip + 1.2x Scale) + WBF.
3. Ensemble: YOLOv8n (extended_best.pt) + YOLOv8s (exp_v8s_10ep/weights/best.pt) with WBF.
Saves all outputs and metrics strictly inside improvement_v4/.
Uses official score.py without modification.
"""
import os
import csv
import json
import time
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from score import evaluate, load_gt, load_ids, combine

IMPROVEMENT_DIR = ROOT_DIR / "improvement_v4"
V8S_CKPT = IMPROVEMENT_DIR / "runs" / "exp_v8s_10ep" / "weights" / "best.pt"
V8N_CKPT = ROOT_DIR / "weights" / "extended_best.pt"

VAL_DIR = ROOT_DIR / "val_images"
VAL_IDS_CSV = ROOT_DIR / "val_image_ids.csv"
VAL_LABELS_CSV = ROOT_DIR / "val_labels.csv"

def run_single_inference(model, val_ids, conf_thr=0.001, iou_thr=0.6, max_det=100):
    print("Running YOLOv8s single-view inference...", flush=True)
    preds = {}
    latencies = []

    for idx, iid in enumerate(val_ids):
        img_path = str(VAL_DIR / f"{iid}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            preds[iid] = np.zeros((0, 6), dtype=np.float64)
            continue
        H, W = img.shape[:2]

        t0 = time.perf_counter()
        results = model.predict(
            source=img,
            imgsz=640,
            conf=conf_thr,
            iou=iou_thr,
            max_det=max_det,
            device="0",
            verbose=False,
        )[0]
        dur = time.perf_counter() - t0
        latencies.append(dur * 1000)

        boxes = results.boxes
        if len(boxes) == 0:
            preds[iid] = np.zeros((0, 6), dtype=np.float64)
            continue

        cls_ids = boxes.cls.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        res = np.column_stack([cls_ids, scores, xyxy])
        order = np.argsort(-res[:, 1], kind="stable")[:max_det]
        preds[iid] = res[order]

        if (idx + 1) % 500 == 0 or (idx + 1) == len(val_ids):
            print(f"  Processed {idx + 1}/{len(val_ids)} images...", flush=True)

    return preds, float(np.mean(latencies)), float(np.median(latencies))


def run_tta_wbf_inference(model, val_ids, conf_thr=0.001, wbf_iou=0.45, max_det=100):
    print("Running YOLOv8s 3-View TTA + WBF inference...", flush=True)
    preds = {}
    latencies = []

    for idx, iid in enumerate(val_ids):
        img_path = str(VAL_DIR / f"{iid}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            preds[iid] = np.zeros((0, 6), dtype=np.float64)
            continue
        H, W = img.shape[:2]

        t0 = time.perf_counter()

        # View 1: Original
        r_orig = model.predict(source=img, imgsz=640, conf=conf_thr, iou=0.6, max_det=150, device="0", verbose=False)[0]
        # View 2: Horizontal Flip
        img_flip = cv2.flip(img, 1)
        r_flip = model.predict(source=img_flip, imgsz=640, conf=conf_thr, iou=0.6, max_det=150, device="0", verbose=False)[0]
        # View 3: 1.2x Scale
        scale_sz = int(round(640 * 1.2 / 32) * 32)
        r_scale = model.predict(source=img, imgsz=scale_sz, conf=conf_thr, iou=0.6, max_det=150, device="0", verbose=False)[0]

        # Process detections per view
        views_data = []

        # 1. Orig
        if len(r_orig.boxes) > 0:
            d_orig = np.column_stack([r_orig.boxes.cls.cpu().numpy(), r_orig.boxes.conf.cpu().numpy(), r_orig.boxes.xyxy.cpu().numpy()])
            views_data.append(d_orig)
        else:
            views_data.append(np.zeros((0, 6)))

        # 2. Flip
        if len(r_flip.boxes) > 0:
            d_flip = np.column_stack([r_flip.boxes.cls.cpu().numpy(), r_flip.boxes.conf.cpu().numpy(), r_flip.boxes.xyxy.cpu().numpy()])
            # Invert horizontal coordinates
            x1_trans = W - d_flip[:, 4]
            x2_trans = W - d_flip[:, 2]
            d_flip[:, 2] = x1_trans
            d_flip[:, 4] = x2_trans
            views_data.append(d_flip)
        else:
            views_data.append(np.zeros((0, 6)))

        # 3. Scale (predictions are already in original image coordinates by YOLO predict)
        if len(r_scale.boxes) > 0:
            d_scale = np.column_stack([r_scale.boxes.cls.cpu().numpy(), r_scale.boxes.conf.cpu().numpy(), r_scale.boxes.xyxy.cpu().numpy()])
            views_data.append(d_scale)
        else:
            views_data.append(np.zeros((0, 6)))

        # WBF fusion
        boxes_list, scores_list, labels_list = [], [], []
        has_dets = False
        for d in views_data:
            if len(d) > 0:
                mask = d[:, 1] >= conf_thr
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

        if not has_dets:
            preds[iid] = np.zeros((0, 6), dtype=np.float64)
            latencies.append((time.perf_counter() - t0) * 1000)
            continue

        try:
            f_boxes, f_scores, f_labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=[1.0, 1.0, 1.0],
                iou_thr=wbf_iou,
                skip_box_thr=conf_thr,
                conf_type="max",
            )
        except Exception:
            f_boxes, f_scores, f_labels = np.zeros((0, 4)), np.zeros(0), np.zeros(0)

        dur = time.perf_counter() - t0
        latencies.append(dur * 1000)

        if len(f_boxes) == 0:
            preds[iid] = np.zeros((0, 6), dtype=np.float64)
            continue

        f_boxes[:, [0, 2]] *= W
        f_boxes[:, [1, 3]] *= H
        res = np.column_stack([f_labels, f_scores, f_boxes])
        order = np.argsort(-res[:, 1], kind="stable")[:max_det]
        preds[iid] = res[order]

        if (idx + 1) % 500 == 0 or (idx + 1) == len(val_ids):
            print(f"  Processed {idx + 1}/{len(val_ids)} images...", flush=True)

    return preds, float(np.mean(latencies)), float(np.median(latencies))


def score_preds(gt, preds, val_ids):
    eval_res = evaluate(gt, preds, val_ids)
    comb_res = combine(eval_res)
    return {
        "accuracy_score": float(comb_res["accuracy_score"]),
        "mAP50": float(eval_res["mAP50"]),
        "mAP50_95": float(eval_res["mAP50_95"]),
        "precision": float(eval_res["precision"]),
        "recall": float(eval_res["recall"]),
        "f1": float(eval_res["f1"]),
        "conf_threshold": float(eval_res["confidence_threshold"]),
        "per_class": eval_res["per_class"],
    }


def main():
    print("=" * 80, flush=True)
    print("EVALUATING YOLOv8s (exp_v8s_10ep) ON ALL 2,150 VALIDATION IMAGES", flush=True)
    print("=" * 80, flush=True)

    val_ids = load_ids(str(VAL_IDS_CSV))
    gt = load_gt(str(VAL_LABELS_CSV))
    print(f"Loaded {len(val_ids)} validation IDs and ground truth.", flush=True)

    model_s = YOLO(str(V8S_CKPT))
    print(f"Loaded YOLOv8s checkpoint from: {V8S_CKPT}", flush=True)

    results_all = {}

    # 1. YOLOv8s Alone
    preds_alone, mean_lat_alone, med_lat_alone = run_single_inference(model_s, val_ids)
    res_alone = score_preds(gt, preds_alone, val_ids)
    res_alone["mean_latency_ms"] = mean_lat_alone
    res_alone["median_latency_ms"] = med_lat_alone
    results_all["yolov8s_alone"] = res_alone
    print("\n--- YOLOv8s Alone Results ---", flush=True)
    print(f"Accuracy Score: {res_alone['accuracy_score']:.2f}", flush=True)
    print(f"mAP@50:         {res_alone['mAP50']:.4f}", flush=True)
    print(f"mAP@50:95:      {res_alone['mAP50_95']:.4f}", flush=True)
    print(f"Precision:      {res_alone['precision']:.4f}", flush=True)
    print(f"Recall:         {res_alone['recall']:.4f}", flush=True)
    print(f"F1 Score:       {res_alone['f1']:.4f}", flush=True)
    print(f"Median Latency: {med_lat_alone:.2f} ms/image", flush=True)

    # 2. YOLOv8s + 3-View TTA + WBF
    preds_tta, mean_lat_tta, med_lat_tta = run_tta_wbf_inference(model_s, val_ids)
    res_tta = score_preds(gt, preds_tta, val_ids)
    res_tta["mean_latency_ms"] = mean_lat_tta
    res_tta["median_latency_ms"] = med_lat_tta
    results_all["yolov8s_3view_wbf"] = res_tta
    print("\n--- YOLOv8s 3-View TTA + WBF Results ---", flush=True)
    print(f"Accuracy Score: {res_tta['accuracy_score']:.2f}", flush=True)
    print(f"mAP@50:         {res_tta['mAP50']:.4f}", flush=True)
    print(f"mAP@50:95:      {res_tta['mAP50_95']:.4f}", flush=True)
    print(f"Precision:      {res_tta['precision']:.4f}", flush=True)
    print(f"Recall:         {res_tta['recall']:.4f}", flush=True)
    print(f"F1 Score:       {res_tta['f1']:.4f}", flush=True)
    print(f"Median Latency: {med_lat_tta:.2f} ms/image", flush=True)

    # Save results to JSON
    out_file = IMPROVEMENT_DIR / "evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2)
    print(f"\nSaved evaluation metrics to: {out_file}", flush=True)

if __name__ == "__main__":
    main()
