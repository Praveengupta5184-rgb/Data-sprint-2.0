#!/usr/bin/env python3
import csv
from collections import defaultdict

gt_by_img = defaultdict(list)
with open("val_labels.csv", newline="", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        gt_by_img[r["image_id"]].append((int(r["class_id"]), float(r["x_min"]), float(r["y_min"]), float(r["x_max"]), float(r["y_max"])))

pred_by_img = defaultdict(list)
with open("tta_optimization_v2/val_pred_best_optimized.csv", newline="", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        pred_by_img[r["image_id"]].append((int(r["class_id"]), float(r["score"]), float(r["x_min"]), float(r["y_min"]), float(r["x_max"]), float(r["y_max"])))

def calc_iou(b1, b2):
    xx1 = max(b1[0], b2[0])
    yy1 = max(b1[1], b2[1])
    xx2 = min(b1[2], b2[2])
    yy2 = min(b1[3], b2[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-6)

score_thr = 0.237
stats = {c: {"gt_total": 0, "pred_total": 0, "tp": 0, "fp_bg": 0, "fp_loc": 0, "fp_cls": 0, "fn": 0} for c in range(4)}
all_img_ids = set(list(gt_by_img.keys()) + list(pred_by_img.keys()))

dim_errors = {c: {"dw": [], "dh": [], "d_aspect": []} for c in range(4)}
aspect_by_class = {c: [] for c in range(4)}

for iid in all_img_ids:
    gts = gt_by_img[iid]
    preds = [p for p in pred_by_img[iid] if p[1] >= score_thr]
    preds.sort(key=lambda x: -x[1])
    matched_gt = set()
    for p in preds:
        pc, ps, px1, py1, px2, py2 = p
        p_box = (px1, py1, px2, py2)
        pw = px2 - px1
        ph = py2 - py1
        p_asp = pw / max(1.0, ph)
        aspect_by_class[pc].append(p_asp)
        stats[pc]["pred_total"] += 1
        best_iou = 0.0
        best_g_idx = -1
        for gi, g in enumerate(gts):
            gc, gx1, gy1, gx2, gy2 = g
            if gc == pc and gi not in matched_gt:
                iou = calc_iou(p_box, (gx1, gy1, gx2, gy2))
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = gi
        if best_iou >= 0.50:
            stats[pc]["tp"] += 1
            matched_gt.add(best_g_idx)
            gw = gts[best_g_idx][3] - gts[best_g_idx][1]
            gh = gts[best_g_idx][4] - gts[best_g_idx][2]
            g_asp = gw / max(1.0, gh)
            dim_errors[pc]["dw"].append(pw - gw)
            dim_errors[pc]["dh"].append(ph - gh)
            dim_errors[pc]["d_aspect"].append(p_asp - g_asp)
        elif 0.20 <= best_iou < 0.50:
            stats[pc]["fp_loc"] += 1
        else:
            cross_cls_iou = 0.0
            for gi, g in enumerate(gts):
                gc, gx1, gy1, gx2, gy2 = g
                if gc != pc:
                    c_iou = calc_iou(p_box, (gx1, gy1, gx2, gy2))
                    if c_iou > cross_cls_iou:
                        cross_cls_iou = c_iou
            if cross_cls_iou >= 0.35:
                stats[pc]["fp_cls"] += 1
            else:
                stats[pc]["fp_bg"] += 1

    for gi, g in enumerate(gts):
        gc = g[0]
        stats[gc]["gt_total"] += 1
        if gi not in matched_gt:
            stats[gc]["fn"] += 1

class_names = ["D00_longitudinal", "D10_transverse", "D20_alligator", "D40_pothole"]
print(f"Validation Error Breakdown at confidence {score_thr}:")
for c in range(4):
    st = stats[c]
    p_tot = max(1, st["pred_total"])
    print(f"=== Class {c}: {class_names[c]} ===")
    print(f"  GT: {st['gt_total']} | Preds: {st['pred_total']}")
    print(f"  TP (IoU>=0.50): {st['tp']} (Recall: {st['tp']/st['gt_total']*100:.1f}%)")
    print(f"  FN (Missed):    {st['fn']} ({st['fn']/st['gt_total']*100:.1f}%)")
    print(f"  FP Localization (0.2<=IoU<0.5): {st['fp_loc']} ({st['fp_loc']/p_tot*100:.1f}%)")
    print(f"  FP Class Confusion:             {st['fp_cls']} ({st['fp_cls']/p_tot*100:.1f}%)")
    print(f"  FP Background/False Alarm:      {st['fp_bg']} ({st['fp_bg']/p_tot*100:.1f}%)")
    if dim_errors[c]["dw"]:
        print(f"  Avg Box Dim Error: dw={sum(dim_errors[c]['dw'])/len(dim_errors[c]['dw']):+.1f}px, dh={sum(dim_errors[c]['dh'])/len(dim_errors[c]['dh']):+.1f}px")
