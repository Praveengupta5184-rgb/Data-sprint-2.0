#!/usr/bin/env python3
"""
Class-wise prediction optimization using cached validation predictions.
Phase 1: Class-specific confidence thresholds
Phase 2: Class-specific NMS IoU
Phase 3: Combined best config
"""
import os
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
val_labels_csv = Path("val_labels.csv").resolve()
val_image_ids_csv = Path("val_image_ids.csv").resolve()
cached_pred_csv = Path("opt_results/val_pred_conf0.001_iou0.5_maxdet100.csv").resolve()

# Output directory
opt_dir = Path("classopt_results")
opt_dir.mkdir(exist_ok=True)

# Load cached predictions
print("Loading cached predictions...")
df = pd.read_csv(cached_pred_csv)
print(f"  Total predictions: {len(df)}")
print(f"  Images covered: {df['image_id'].nunique()}")
print(f"  Classes: {sorted(df['class_id'].unique())}")

# Load validation image IDs to ensure we score all 2150
val_ids = pd.read_csv(val_image_ids_csv)['image_id'].tolist()
print(f"  Validation images to score: {len(val_ids)}")


def box_iou_numpy(boxes1, boxes2):
    """Compute IoU between two sets of boxes."""
    # boxes: [N, 4] format [x1, y1, x2, y2]
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-12)


def apply_nms_per_class(df, iou_thresholds):
    """Apply NMS per image per class with class-specific IoU thresholds."""
    results = []

    for img_id, group in df.groupby('image_id'):
        for cls_id in range(4):
            cls_df = group[group['class_id'] == cls_id].copy()
            if len(cls_df) == 0:
                continue

            iou_thr = iou_thresholds.get(cls_id, 0.5)

            # Sort by score descending
            cls_df = cls_df.sort_values('score', ascending=False).reset_index(drop=True)

            boxes = cls_df[['x_min', 'y_min', 'x_max', 'y_max']].values
            scores = cls_df['score'].values
            keep = []

            while len(boxes) > 0:
                best_idx = 0
                keep.append(cls_df.iloc[best_idx])
                if len(boxes) == 1:
                    break

                ious = box_iou_numpy(boxes[0:1], boxes[1:])[0]
                boxes = boxes[1:][ious < iou_thr]
                cls_df = cls_df.iloc[1:][ious < iou_thr].reset_index(drop=True)

            results.extend(keep)

    return pd.DataFrame(results) if results else pd.DataFrame(columns=df.columns)


def apply_conf_filter(df, conf_thresholds):
    """Apply class-specific confidence thresholds."""
    mask = np.ones(len(df), dtype=bool)
    for cls_id, thresh in conf_thresholds.items():
        mask &= ~((df['class_id'] == cls_id) & (df['score'] < thresh))
    return df[mask].copy()


def run_score(pred_df, output_json):
    """Run official scorer on predictions DataFrame."""
    # Save to CSV
    pred_csv = opt_dir / "temp_pred.csv"
    pred_df.to_csv(pred_csv, index=False, columns=['image_id', 'class_id', 'score', 'x_min', 'y_min', 'x_max', 'y_max'])

    cmd = [
        sys.executable, "score.py",
        "--pred", str(pred_csv),
        "--gt", str(val_labels_csv),
        "--image-ids", str(val_image_ids_csv),
        "--json-out", str(output_json)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Scoring failed: {result.stderr}")
        return None

    with open(output_json) as f:
        return json.load(f)


def evaluate_config(config_name, conf_thresholds, iou_thresholds):
    """Evaluate a single configuration."""
    print(f"\n--- {config_name} ---")
    print(f"  Conf: {conf_thresholds}")
    print(f"  IoU:  {iou_thresholds}")

    # Apply confidence filter
    filtered = apply_conf_filter(df, conf_thresholds)
    print(f"  After conf filter: {len(filtered)} predictions")

    # Apply NMS
    nms_result = apply_nms_per_class(filtered, iou_thresholds)
    print(f"  After NMS: {len(nms_result)} predictions")

    # Score
    json_out = opt_dir / f"score_{config_name}.json"
    score_result = run_score(nms_result, json_out)

    if score_result is None:
        return None

    m = score_result["main"]
    acc = score_result["combined"]["accuracy_score"]

    print(f"  Accuracy Score: {acc:.4f}")
    print(f"  mAP@50: {m['mAP50']:.4f}, mAP@50:95: {m['mAP50_95']:.4f}")
    print(f"  Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}")

    return {
        "config": config_name,
        "conf_thresholds": conf_thresholds,
        "iou_thresholds": iou_thresholds,
        "n_predictions": len(nms_result),
        "accuracy_score": acc,
        "mAP50": m["mAP50"],
        "mAP50_95": m["mAP50_95"],
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "confidence_threshold": m["confidence_threshold"],
        "per_class": m["per_class"],
    }


def main():
    all_results = []
    baseline_score = 42.219061
    best_score = baseline_score
    best_config = None

    # ============================================================
    # PHASE 1: Class-specific confidence thresholds
    # ============================================================
    print("\n" + "="*80)
    print("PHASE 1: CLASS-SPECIFIC CONFIDENCE THRESHOLDS")
    print("="*80)

    phase1_configs = [
        ("base", {0: 0.001, 1: 0.001, 2: 0.001, 3: 0.001}, {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}),
        ("C1_D00_strict", {0: 0.01, 1: 0.001, 2: 0.001, 3: 0.001}, {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}),
        ("C2_D10_strict", {0: 0.001, 1: 0.01, 2: 0.001, 3: 0.001}, {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}),
        ("C3_D20_strict", {0: 0.001, 1: 0.001, 2: 0.05, 3: 0.001}, {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}),
        ("C4_D40_strict", {0: 0.001, 1: 0.001, 2: 0.001, 3: 0.01}, {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}),
        ("C5_D00_D10_strict", {0: 0.01, 1: 0.01, 2: 0.001, 3: 0.001}, {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}),
        ("C6_uniform_005", {0: 0.005, 1: 0.005, 2: 0.005, 3: 0.005}, {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}),
        ("C7_tuned", {0: 0.01, 1: 0.005, 2: 0.01, 3: 0.005}, {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}),
    ]

    phase1_results = []
    for name, conf_t, iou_t in phase1_configs:
        result = evaluate_config(name, conf_t, iou_t)
        if result:
            phase1_results.append(result)
            all_results.append(result)
            if result["accuracy_score"] > best_score:
                best_score = result["accuracy_score"]
                best_config = result

    # ============================================================
    # PHASE 2: Class-specific NMS IoU (using best conf from Phase 1)
    # ============================================================
    print("\n" + "="*80)
    print("PHASE 2: CLASS-SPECIFIC NMS IoU")
    print("="*80)

    # Use best conf from Phase 1
    best_conf = best_config["conf_thresholds"] if best_config else {0: 0.001, 1: 0.001, 2: 0.001, 3: 0.001}
    print(f"Using best conf from Phase 1: {best_conf}")

    phase2_configs = [
        ("base_iou", best_conf, {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}),
        ("N1_D00_strict", best_conf, {0: 0.4, 1: 0.5, 2: 0.5, 3: 0.5}),
        ("N2_D10_strict", best_conf, {0: 0.5, 1: 0.4, 2: 0.5, 3: 0.5}),
        ("N3_D20_loose", best_conf, {0: 0.5, 1: 0.5, 2: 0.6, 3: 0.5}),
        ("N4_D40_strict", best_conf, {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.4}),
        ("N5_D00_D10_strict", best_conf, {0: 0.4, 1: 0.4, 2: 0.5, 3: 0.5}),
    ]

    for name, conf_t, iou_t in phase2_configs:
        result = evaluate_config(name, conf_t, iou_t)
        if result:
            all_results.append(result)
            if result["accuracy_score"] > best_score:
                best_score = result["accuracy_score"]
                best_config = result

    # ============================================================
    # PHASE 3: Combined best config
    # ============================================================
    print("\n" + "="*80)
    print("PHASE 3: COMBINED BEST CONFIG")
    print("="*80)

    # Already evaluated in Phase 2 if best was from Phase 1
    # Just verify the best overall
    if best_config:
        print(f"\n*** BEST OVERALL: {best_config['config']} ***")
        print(f"   Accuracy Score: {best_config['accuracy_score']:.4f} (delta: {best_config['accuracy_score'] - baseline_score:+.4f})")
        print(f"   mAP@50: {best_config['mAP50']:.4f}")
        print(f"   mAP@50:95: {best_config['mAP50_95']:.4f}")
        print(f"   Precision: {best_config['precision']:.4f}")
        print(f"   Recall: {best_config['recall']:.4f}")

    # Sort and save
    all_results.sort(key=lambda x: x["accuracy_score"], reverse=True)

    # Save full results
    with open(opt_dir / "comparison_report.json", "w") as f:
        json.dump({
            "baseline_score": baseline_score,
            "best_config": best_config["config"] if best_config else "base",
            "best_accuracy_score": best_score,
            "improvement": best_score - baseline_score,
            "all_results": all_results
        }, f, indent=2)

# Print summary
    print("\n" + "="*80)
    print("FINAL COMPARISON REPORT")
    print("="*80)
    print(f"{'Config':<25} {'AccScore':>8} {'Delta':>6} {'mAP50':>7} {'mAP50:95':>8} {'Prec':>6} {'Rec':>6} {'Preds':>6}")
    print("-"*80)
    for r in all_results:
        delta = r["accuracy_score"] - baseline_score
        print(f"{r['config']:<25} {r['accuracy_score']:>8.4f} {delta:>+6.4f} {r['mAP50']:>7.4f} {r['mAP50_95']:>8.4f} {r['precision']:>6.4f} {r['recall']:>6.4f} {r['n_predictions']:>6}")

    # Per-class for top 3
    print("\nPer-class AP@50 for top 3:")
    for r in all_results[:3]:
        print(f"\n  {r['config']}:")
        for name, v in r["per_class"].items():
            print(f"    {name:<28} AP50 {v['AP50']:.4f}   AP50:95 {v['AP50_95']:.4f}")

    print(f"\nResults saved to: {opt_dir / 'comparison_report.json'}")
    print(f"Best improvement: {best_score - baseline_score:+.4f}")

    if best_score - baseline_score >= 0.20:
        print(f"\n✓ MEETS THRESHOLD (+0.20): Recommend adopting {best_config['config']}")
    else:
        print(f"\n✗ BELOW THRESHOLD (+0.20): No configuration meets improvement target")

    return all_results


if __name__ == "__main__":
    main()