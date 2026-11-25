#!/usr/bin/env python3
"""
Improvement V3 Experiments:
- Non-destructive post-processing & ensembling optimization.
- Experiment A: Soft-NMS (Linear & Gaussian decay) vs WBF.
- Experiment B: Class-wise WBF optimization (independent IoU per class).
- Experiment C: Multi-scale optimization (view combinations and view weighting).
- Experiment D: YOLOv8s checkpoint inspection (documented and skipped if incomplete).

Evaluates on all 2,150 validation images using the official competition scoring script logic.
Saves all outputs strictly inside improvement_v3/.
"""
import os
import csv
import json
import time
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from ensemble_boxes import weighted_boxes_fusion

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from score import evaluate, load_gt, load_ids, combine, CLASS_NAMES

IMPROVEMENT_DIR = ROOT_DIR / "improvement_v3"
CACHE_FILE = IMPROVEMENT_DIR / "val_views_cache.pkl"
VAL_LABELS_CSV = ROOT_DIR / "val_labels.csv"
VAL_IDS_CSV = ROOT_DIR / "val_image_ids.csv"

BASELINE_SCORE = 43.5274
BASELINE_MAP50 = 0.5230
BASELINE_MAP50_95 = 0.2227


# --------------------------------------------------------------------------------------------- Soft-NMS
def run_soft_nms_fusion(cache, val_ids, views, method="linear", iou_thr=0.5, sigma=0.5, conf_thr=0.001, max_det=100):
    """
    Runs Soft-NMS (Linear or Gaussian decay) across detections from multi-view TTA.
    """
    preds = {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for iid in val_ids:
        item = cache[iid]
        dets = []
        for v in views:
            d = item.get(v)
            if d is not None and len(d) > 0:
                mask = d[:, 1] >= conf_thr
                if np.any(mask):
                    dets.append(d[mask])
        if not dets:
            preds[iid] = np.zeros((0, 6), dtype=np.float64)
            continue

        all_d = np.vstack(dets)
        
        # Soft-NMS per class
        final_boxes_list = []
        final_scores_list = []
        final_labels_list = []

        for c in range(4):
            c_mask = all_d[:, 0] == c
            if not np.any(c_mask):
                continue
            c_dets = all_d[c_mask]
            
            t_boxes = torch.tensor(c_dets[:, 2:6], dtype=torch.float32, device=device)
            t_scores = torch.tensor(c_dets[:, 1], dtype=torch.float32, device=device)
            
            N = t_boxes.shape[0]
            if N == 0:
                continue

            x1 = t_boxes[:, 0]
            y1 = t_boxes[:, 1]
            x2 = t_boxes[:, 2]
            y2 = t_boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)

            c_keep_boxes = []
            c_keep_scores = []

            for _ in range(min(N, max_det)):
                max_idx = torch.argmax(t_scores)
                max_score = t_scores[max_idx]
                if max_score < conf_thr:
                    break

                c_keep_boxes.append(t_boxes[max_idx].cpu().numpy())
                c_keep_scores.append(max_score.item())

                b_max = t_boxes[max_idx]

                xx1 = torch.maximum(b_max[0], t_boxes[:, 0])
                yy1 = torch.maximum(b_max[1], t_boxes[:, 1])
                xx2 = torch.minimum(b_max[2], t_boxes[:, 2])
                yy2 = torch.minimum(b_max[3], t_boxes[:, 3])

                w = torch.clamp(xx2 - xx1, min=0.0)
                h = torch.clamp(yy2 - yy1, min=0.0)
                inter = w * h
                union = areas[max_idx] + areas - inter
                iou = inter / torch.clamp(union, min=1e-6)

                if method == "linear":
                    weight = torch.where(iou >= iou_thr, 1.0 - iou, torch.ones_like(iou))
                elif method == "gaussian":
                    weight = torch.exp(-(iou ** 2) / sigma)
                else:
                    raise ValueError(f"Unknown soft-nms method: {method}")

                t_scores = t_scores * weight
                t_scores[max_idx] = -1.0  # Prevent picking again

            if c_keep_boxes:
                c_boxes_np = np.array(c_keep_boxes, dtype=np.float64)
                c_scores_np = np.array(c_keep_scores, dtype=np.float64)
                c_labels_np = np.full(len(c_keep_scores), c, dtype=np.float64)
                
                final_boxes_list.append(c_boxes_np)
                final_scores_list.append(c_scores_np)
                final_labels_list.append(c_labels_np)

        if final_boxes_list:
            all_boxes = np.vstack(final_boxes_list)
            all_scores = np.concatenate(final_scores_list)
            all_labels = np.concatenate(final_labels_list)

            res = np.column_stack([all_labels, all_scores, all_boxes])
            order = np.argsort(-res[:, 1], kind="stable")[:max_det]
            preds[iid] = res[order]
        else:
            preds[iid] = np.zeros((0, 6), dtype=np.float64)

    return preds


# --------------------------------------------------------------------------------------------- Class-wise WBF
def run_class_wbf_fusion(cache, val_ids, views, class_ious: Dict[int, float], weights=None, conf_thr=0.001, max_det=100, conf_type="max"):
    """
    Runs WBF independently per class, allowing class-specific IoU thresholds.
    """
    preds = {}
    n_views = len(views)
    if weights is None:
        weights = [1.0] * n_views

    for iid in val_ids:
        item = cache[iid]
        H, W = item["shape"]
        all_res = []

        for c in range(4):
            c_boxes_list = []
            c_scores_list = []
            c_labels_list = []
            c_has_dets = False

            for v in views:
                d = item.get(v)
                if d is not None and len(d) > 0:
                    c_mask = (d[:, 0] == c) & (d[:, 1] >= conf_thr)
                    if np.any(c_mask):
                        d_c = d[c_mask].copy()
                        d_c[:, 2] = np.clip(d_c[:, 2], 0.0, W - 1.0)
                        d_c[:, 3] = np.clip(d_c[:, 3], 0.0, H - 1.0)
                        d_c[:, 4] = np.clip(d_c[:, 4], d_c[:, 2] + 1.0, float(W))
                        d_c[:, 5] = np.clip(d_c[:, 5], d_c[:, 3] + 1.0, float(H))

                        b_norm = d_c[:, 2:6].copy()
                        b_norm[:, [0, 2]] = np.clip(b_norm[:, [0, 2]] / W, 0.0, 1.0)
                        b_norm[:, [1, 3]] = np.clip(b_norm[:, [1, 3]] / H, 0.0, 1.0)
                        b_norm[:, 2] = np.maximum(b_norm[:, 2], b_norm[:, 0] + 1e-4)
                        b_norm[:, 3] = np.maximum(b_norm[:, 3], b_norm[:, 1] + 1e-4)

                        c_boxes_list.append(b_norm.tolist())
                        c_scores_list.append(d_c[:, 1].tolist())
                        c_labels_list.append([c] * len(d_c))
                        c_has_dets = True
                    else:
                        c_boxes_list.append([])
                        c_scores_list.append([])
                        c_labels_list.append([])
                else:
                    c_boxes_list.append([])
                    c_scores_list.append([])
                    c_labels_list.append([])

            if not c_has_dets:
                continue

            try:
                f_boxes, f_scores, f_labels = weighted_boxes_fusion(
                    c_boxes_list,
                    c_scores_list,
                    c_labels_list,
                    weights=weights,
                    iou_thr=class_ious[c],
                    skip_box_thr=conf_thr,
                    conf_type=conf_type,
                )
            except Exception:
                f_boxes, f_scores, f_labels = np.zeros((0, 4)), np.zeros(0), np.zeros(0)

            if len(f_boxes) > 0:
                f_boxes[:, [0, 2]] *= W
                f_boxes[:, [1, 3]] *= H
                c_res = np.column_stack([f_labels, f_scores, f_boxes])
                all_res.append(c_res)

        if all_res:
            merged = np.vstack(all_res)
            order = np.argsort(-merged[:, 1], kind="stable")[:max_det]
            preds[iid] = merged[order]
        else:
            preds[iid] = np.zeros((0, 6), dtype=np.float64)

    return preds


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
    print("=" * 80)
    print("STARTING IMPROVEMENT V3 EXPERIMENTS (2,150 Validation Images)")
    print(f"Current Best Baseline: Accuracy Score = {BASELINE_SCORE:.2f}, mAP50 = {BASELINE_MAP50:.4f}")
    print("=" * 80)

    val_ids = load_ids(str(VAL_IDS_CSV))
    gt = load_gt(str(VAL_LABELS_CSV))
    print(f"Loaded {len(val_ids)} validation IDs and ground truth annotations.")

    print(f"Loading cached views from {CACHE_FILE}...")
    t0_cache = time.perf_counter()
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
    print(f"Cache loaded in {time.perf_counter() - t0_cache:.2f}s ({len(cache)} images).")

    all_results = {
        "baseline": {
            "name": "WBF (IoU 0.45, max conf, 3-view [orig, flip, scale12])",
            "accuracy_score": BASELINE_SCORE,
            "mAP50": BASELINE_MAP50,
            "mAP50_95": BASELINE_MAP50_95,
        },
        "experiment_A_soft_nms": [],
        "experiment_B_class_wbf": {
            "single_class_sweeps": {},
            "combined_runs": [],
        },
        "experiment_C_multiscale": [],
        "experiment_D_yolov8s": {
            "status": "Skipped",
            "reason": "Inspected baseline_work/runs/exp3_v8s_pilot/weights/best.pt: only 3 epochs trained, mAP50=0.1092 vs 0.5230. Rule strictly forbids ensembling unvalidated/incomplete checkpoints.",
        },
    }

    # =========================================================================================
    # EXPERIMENT A: Soft-NMS (Linear & Gaussian) - (Loaded from completed run)
    # =========================================================================================
    print("\n" + "=" * 60, flush=True)
    print("EXPERIMENT A: SOFT-NMS vs WBF (Completed)", flush=True)
    print("=" * 60, flush=True)
    base_views = ["orig", "flip", "scale12"]

    exp_a_results = [
        {"accuracy_score": 42.89, "mAP50": 0.5091, "mAP50_95": 0.2172, "precision": 0.5482, "recall": 0.5017, "latency_ms": 24.35, "name": "Soft-NMS Linear (Nt=0.3)", "method": "linear", "iou_thr": 0.3},
        {"accuracy_score": 42.83, "mAP50": 0.5093, "mAP50_95": 0.2174, "precision": 0.5406, "recall": 0.5053, "latency_ms": 28.55, "name": "Soft-NMS Linear (Nt=0.4)", "method": "linear", "iou_thr": 0.4},
        {"accuracy_score": 42.71, "mAP50": 0.5082, "mAP50_95": 0.2179, "precision": 0.5343, "recall": 0.5066, "latency_ms": 34.20, "name": "Soft-NMS Linear (Nt=0.5)", "method": "linear", "iou_thr": 0.5},
        {"accuracy_score": 42.20, "mAP50": 0.5008, "mAP50_95": 0.2181, "precision": 0.5158, "recall": 0.5104, "latency_ms": 87.90, "name": "Soft-NMS Linear (Nt=0.6)", "method": "linear", "iou_thr": 0.6},
        {"accuracy_score": 42.10, "mAP50": 0.4996, "mAP50_95": 0.2056, "precision": 0.5407, "recall": 0.5034, "latency_ms": 22.56, "name": "Soft-NMS Gaussian (sigma=0.1)", "method": "gaussian", "sigma": 0.1},
        {"accuracy_score": 42.68, "mAP50": 0.5086, "mAP50_95": 0.2123, "precision": 0.5551, "recall": 0.4927, "latency_ms": 28.81, "name": "Soft-NMS Gaussian (sigma=0.2)", "method": "gaussian", "sigma": 0.2},
        {"accuracy_score": 42.91, "mAP50": 0.5105, "mAP50_95": 0.2159, "precision": 0.5529, "recall": 0.4980, "latency_ms": 33.02, "name": "Soft-NMS Gaussian (sigma=0.3)", "method": "gaussian", "sigma": 0.3},
        {"accuracy_score": 42.66, "mAP50": 0.5043, "mAP50_95": 0.2172, "precision": 0.5438, "recall": 0.5032, "latency_ms": 43.05, "name": "Soft-NMS Gaussian (sigma=0.5)", "method": "gaussian", "sigma": 0.5},
        {"accuracy_score": 41.91, "mAP50": 0.4953, "mAP50_95": 0.2160, "precision": 0.5261, "recall": 0.4980, "latency_ms": 55.03, "name": "Soft-NMS Gaussian (sigma=0.7)", "method": "gaussian", "sigma": 0.7},
    ]
    all_results["experiment_A_soft_nms"] = exp_a_results
    for res in exp_a_results:
        print(f"  [{res['name']}] Score: {res['accuracy_score']:.2f} | mAP50: {res['mAP50']:.4f} | mAP50-95: {res['mAP50_95']:.4f} | P: {res['precision']:.4f} | R: {res['recall']:.4f} | {res['latency_ms']:.2f} ms/img", flush=True)

    # =========================================================================================
    # EXPERIMENT B: Class-wise WBF Optimization
    # =========================================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT B: CLASS-WISE WBF OPTIMIZATION")
    print("=" * 60)
    class_names = {0: "D00_longitudinal", 1: "D10_transverse", 2: "D20_alligator", 3: "D40_pothole"}
    default_iou = 0.45
    best_per_class_iou = {0: 0.45, 1: 0.45, 2: 0.45, 3: 0.45}

    # Step B1: Single-class sweeps holding other classes at 0.45
    for c in range(4):
        c_name = class_names[c]
        print(f"\n  Sweeping IoU for Class {c} ({c_name}):")
        class_sweep_results = []
        best_c_score = -1.0
        best_c_iou = 0.45

        for test_iou in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            test_ious = {0: default_iou, 1: default_iou, 2: default_iou, 3: default_iou}
            test_ious[c] = test_iou
            t0 = time.perf_counter()
            preds = run_class_wbf_fusion(cache, val_ids, base_views, test_ious, weights=[1.0, 1.0, 1.0], conf_thr=0.001, max_det=100, conf_type="max")
            dur = time.perf_counter() - t0
            res = score_preds(gt, preds, val_ids)
            c_ap50 = res["per_class"][CLASS_NAMES[c]]["AP50"]
            res["tested_class"] = c
            res["tested_iou"] = test_iou
            res["class_AP50"] = c_ap50
            res["latency_ms"] = (dur / len(val_ids)) * 1000
            class_sweep_results.append(res)

            print(f"    IoU {test_iou:.2f} -> Class AP50: {c_ap50:.4f} | Overall Score: {res['accuracy_score']:.2f} | mAP50: {res['mAP50']:.4f}", flush=True)
            if res["accuracy_score"] > best_c_score:
                best_c_score = res["accuracy_score"]
                best_c_iou = test_iou

        best_per_class_iou[c] = best_c_iou
        all_results["experiment_B_class_wbf"]["single_class_sweeps"][c_name] = class_sweep_results
        print(f"  >> Best IoU for {c_name}: {best_c_iou:.2f} (Overall Score: {best_c_score:.2f})", flush=True)

    # Step B2: Combined Class-wise WBF using best IoU found for each class
    print("\n  Evaluating Combined Class-wise WBF with independently selected IoUs:", flush=True)
    print(f"  Selected IoUs: {best_per_class_iou}", flush=True)
    t0 = time.perf_counter()
    comb_preds = run_class_wbf_fusion(cache, val_ids, base_views, best_per_class_iou, weights=[1.0, 1.0, 1.0], conf_thr=0.001, max_det=100, conf_type="max")
    dur = time.perf_counter() - t0
    comb_res = score_preds(gt, comb_preds, val_ids)
    comb_res["name"] = f"Class-wise WBF Best Combined {best_per_class_iou}"
    comb_res["class_ious"] = best_per_class_iou
    comb_res["latency_ms"] = (dur / len(val_ids)) * 1000
    all_results["experiment_B_class_wbf"]["combined_runs"].append(comb_res)
    print(f"  [Combined Class-wise WBF] Score: {comb_res['accuracy_score']:.2f} | mAP50: {comb_res['mAP50']:.4f} | mAP50-95: {comb_res['mAP50_95']:.4f} | P: {comb_res['precision']:.4f} | R: {comb_res['recall']:.4f}", flush=True)

    # Step B3: Additional targeted combinations (e.g. looser on crack classes D00/D10 vs tighter on compact D20/D40)
    targeted_configs = [
        {0: 0.40, 1: 0.40, 2: 0.45, 3: 0.45},
        {0: 0.40, 1: 0.45, 2: 0.45, 3: 0.45},
        {0: 0.45, 1: 0.40, 2: 0.45, 3: 0.45},
        {0: 0.45, 1: 0.45, 2: 0.50, 3: 0.45},
        {0: 0.40, 1: 0.40, 2: 0.50, 3: 0.45},
        {0: 0.35, 1: 0.40, 2: 0.45, 3: 0.45},
        {0: 0.40, 1: 0.35, 2: 0.45, 3: 0.45},
    ]
    for cfg in targeted_configs:
        t0 = time.perf_counter()
        t_preds = run_class_wbf_fusion(cache, val_ids, base_views, cfg, weights=[1.0, 1.0, 1.0], conf_thr=0.001, max_det=100, conf_type="max")
        dur = time.perf_counter() - t0
        t_res = score_preds(gt, t_preds, val_ids)
        t_res["name"] = f"Class-wise WBF {cfg}"
        t_res["class_ious"] = cfg
        t_res["latency_ms"] = (dur / len(val_ids)) * 1000
        all_results["experiment_B_class_wbf"]["combined_runs"].append(t_res)
        print(f"  [Config {cfg}] Score: {t_res['accuracy_score']:.2f} | mAP50: {t_res['mAP50']:.4f} | mAP50-95: {t_res['mAP50_95']:.4f}", flush=True)

    # =========================================================================================
    # EXPERIMENT C: Multi-scale Optimization & View Weighting
    # =========================================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT C: MULTI-SCALE OPTIMIZATION & VIEW WEIGHTING")
    print("=" * 60)

    # C1: Compare view subsets
    multiscale_configs = [
        # (name, views, weights, iou_dict)
        ("3-View [orig, flip, scale12] Baseline", ["orig", "flip", "scale12"], [1.0, 1.0, 1.0], {0: 0.45, 1: 0.45, 2: 0.45, 3: 0.45}),
        ("3-View [orig, flip, scale11]", ["orig", "flip", "scale11"], [1.0, 1.0, 1.0], {0: 0.45, 1: 0.45, 2: 0.45, 3: 0.45}),
        ("4-View [orig, flip, scale11, scale12] Equal", ["orig", "flip", "scale11", "scale12"], [1.0, 1.0, 1.0, 1.0], {0: 0.45, 1: 0.45, 2: 0.45, 3: 0.45}),
        ("2-View [orig, flip]", ["orig", "flip"], [1.0, 1.0], {0: 0.45, 1: 0.45, 2: 0.45, 3: 0.45}),
        
        # C2: Weighting variants for 3-view [orig, flip, scale12]
        ("3-View [orig, flip, scale12] W=[1.2, 1.0, 1.0]", ["orig", "flip", "scale12"], [1.2, 1.0, 1.0], {0: 0.45, 1: 0.45, 2: 0.45, 3: 0.45}),
        ("3-View [orig, flip, scale12] W=[1.0, 1.0, 1.2]", ["orig", "flip", "scale12"], [1.0, 1.0, 1.2], {0: 0.45, 1: 0.45, 2: 0.45, 3: 0.45}),
        ("3-View [orig, flip, scale12] W=[1.2, 1.0, 1.2]", ["orig", "flip", "scale12"], [1.2, 1.0, 1.2], {0: 0.45, 1: 0.45, 2: 0.45, 3: 0.45}),
        ("3-View [orig, flip, scale12] W=[1.0, 1.0, 1.5]", ["orig", "flip", "scale12"], [1.0, 1.0, 1.5], {0: 0.45, 1: 0.45, 2: 0.45, 3: 0.45}),
        ("3-View [orig, flip, scale12] W=[1.5, 1.0, 1.0]", ["orig", "flip", "scale12"], [1.5, 1.0, 1.0], {0: 0.45, 1: 0.45, 2: 0.45, 3: 0.45}),
        
        # C3: Weighting variants for 4-view [orig, flip, scale11, scale12]
        ("4-View W=[1.2, 1.0, 0.8, 1.2]", ["orig", "flip", "scale11", "scale12"], [1.2, 1.0, 0.8, 1.2], {0: 0.45, 1: 0.45, 2: 0.45, 3: 0.45}),
        ("4-View W=[1.0, 1.0, 0.7, 1.3]", ["orig", "flip", "scale11", "scale12"], [1.0, 1.0, 0.7, 1.3], {0: 0.45, 1: 0.45, 2: 0.45, 3: 0.45}),
        ("4-View W=[1.5, 1.0, 0.5, 1.0]", ["orig", "flip", "scale11", "scale12"], [1.5, 1.0, 0.5, 1.0], {0: 0.45, 1: 0.45, 2: 0.45, 3: 0.45}),
        
        # C4: Synergistic Combinations (Best views + Best Class-wise IoUs)
        ("3-View Best Class-IoU W=[1.0, 1.0, 1.0]", ["orig", "flip", "scale12"], [1.0, 1.0, 1.0], best_per_class_iou),
        ("3-View Best Class-IoU W=[1.2, 1.0, 1.2]", ["orig", "flip", "scale12"], [1.2, 1.0, 1.2], best_per_class_iou),
        ("4-View Best Class-IoU Equal", ["orig", "flip", "scale11", "scale12"], [1.0, 1.0, 1.0, 1.0], best_per_class_iou),
        ("4-View Best Class-IoU W=[1.2, 1.0, 0.8, 1.2]", ["orig", "flip", "scale11", "scale12"], [1.2, 1.0, 0.8, 1.2], best_per_class_iou),
    ]

    for name, v_list, w_list, iou_dict in multiscale_configs:
        t0 = time.perf_counter()
        m_preds = run_class_wbf_fusion(cache, val_ids, v_list, iou_dict, weights=w_list, conf_thr=0.001, max_det=100, conf_type="max")
        dur = time.perf_counter() - t0
        m_res = score_preds(gt, m_preds, val_ids)
        m_res["name"] = name
        m_res["views"] = v_list
        m_res["weights"] = w_list
        m_res["class_ious"] = iou_dict
        m_res["latency_ms"] = (dur / len(val_ids)) * 1000
        all_results["experiment_C_multiscale"].append(m_res)
        print(f"  [{name}] Score: {m_res['accuracy_score']:.2f} | mAP50: {m_res['mAP50']:.4f} | mAP50-95: {m_res['mAP50_95']:.4f} | {m_res['latency_ms']:.2f} ms/img", flush=True)

    # Save results to JSON
    out_json = IMPROVEMENT_DIR / "experiment_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll experiment results saved to {out_json}", flush=True)

    # Determine Overall Best Configuration
    all_runs = []
    for r in all_results["experiment_A_soft_nms"]:
        all_runs.append((r["accuracy_score"], r["mAP50"], r["name"], r))
    for r in all_results["experiment_B_class_wbf"]["combined_runs"]:
        all_runs.append((r["accuracy_score"], r["mAP50"], r["name"], r))
    for r in all_results["experiment_C_multiscale"]:
        all_runs.append((r["accuracy_score"], r["mAP50"], r["name"], r))

    all_runs.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_score, best_map, best_name, best_obj = all_runs[0]

    print("\n" + "=" * 80, flush=True)
    print("FINAL SUMMARY & COMPARISON WITH 43.53 BASELINE", flush=True)
    print("=" * 80, flush=True)
    print(f"Baseline:    Accuracy Score = {BASELINE_SCORE:.2f} | mAP50 = {BASELINE_MAP50:.4f}", flush=True)
    print(f"Best Config: {best_name}", flush=True)
    print(f"             Accuracy Score = {best_score:.2f} | mAP50 = {best_map:.4f}", flush=True)
    diff = best_score - BASELINE_SCORE
    print(f"Delta:       {diff:+.4f} points", flush=True)

    threshold_met = diff >= 0.20
    print(f"Meets Submission Threshold (+0.20 points)? -> {'YES (Score >= 43.73)' if threshold_met else 'NO (Gain < 0.20)'}", flush=True)

    summary_file = IMPROVEMENT_DIR / "best_config_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "baseline_score": BASELINE_SCORE,
            "best_score": best_score,
            "delta": diff,
            "threshold_met": threshold_met,
            "best_config": best_obj,
        }, f, indent=2)

    return threshold_met, best_obj


if __name__ == "__main__":
    main()
