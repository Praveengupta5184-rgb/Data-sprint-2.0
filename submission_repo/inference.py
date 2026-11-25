"""
inference.py : Final submission for DataSprint 2.0 Computer Vision Track.

Model: YOLOv8n fine-tuned on competition data (weights/extended_best.pt)
Inference: 3-view TTA + Weighted Boxes Fusion (WBF)
- View 1: Original image at 640
- View 2: Horizontal flip at 640
- View 3: Scale 1.2x at 768
Fusion: WBF with IoU=0.45, conf_type='max', conf_thr=0.001, max_det=100
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

_HERE = Path(__file__).resolve().parent
WEIGHTS = _HERE / "weights" / "extended_best.pt"

# WBF configuration (validated on 2,150 validation images)
WBF_IOU = 0.45
WBF_CONF_TYPE = "max"
CONF_THR = 0.001
MAX_DET = 100

# TTA scales
SCALES = [640, 640, 768]  # orig, flip, 1.2x


def _nms_fusion(dets, iou_thr=0.5, max_det=MAX_DET):
    """Fallback NMS fusion if WBF unavailable."""
    if len(dets) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    t_boxes = torch.tensor(dets[:, 2:6], dtype=torch.float32, device="cuda:0")
    t_scores = torch.tensor(dets[:, 1], dtype=torch.float32, device="cuda:0")
    t_classes = torch.tensor(dets[:, 0], dtype=torch.int64, device="cuda:0")
    keep = torch.ops.torchvision.nms(t_boxes, t_scores, iou_thr)[:max_det].cpu().numpy()
    return dets[keep]


def _wbf_fusion(views_dets, img_w, img_h, iou_thr=WBF_IOU, conf_type=WBF_CONF_TYPE, weights=None):
    """Weighted Boxes Fusion across TTA views."""
    if not views_dets:
        return np.zeros((0, 6), dtype=np.float32)

    n_views = len(views_dets)
    if weights is None:
        weights = [1.0] * n_views

    boxes_list = []
    scores_list = []
    labels_list = []

    for dets in views_dets:
        if len(dets) == 0:
            boxes_list.append([])
            scores_list.append([])
            labels_list.append([])
            continue
        # Clip to image bounds
        d = dets.copy()
        d[:, 2] = np.clip(d[:, 2], 0.0, img_w - 1.0)
        d[:, 3] = np.clip(d[:, 3], 0.0, img_h - 1.0)
        d[:, 4] = np.clip(d[:, 4], d[:, 2] + 1.0, float(img_w))
        d[:, 5] = np.clip(d[:, 5], d[:, 3] + 1.0, float(img_h))

        # Normalize [0,1]
        b_norm = d[:, 2:6].copy()
        b_norm[:, [0, 2]] = np.clip(b_norm[:, [0, 2]] / img_w, 0.0, 1.0)
        b_norm[:, [1, 3]] = np.clip(b_norm[:, [1, 3]] / img_h, 0.0, 1.0)
        b_norm[:, 2] = np.maximum(b_norm[:, 2], b_norm[:, 0] + 1e-4)
        b_norm[:, 3] = np.maximum(b_norm[:, 3], b_norm[:, 1] + 1e-4)

        boxes_list.append(b_norm.tolist())
        scores_list.append(d[:, 1].tolist())
        labels_list.append(d[:, 0].astype(int).tolist())

    try:
        from ensemble_boxes import weighted_boxes_fusion
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=0.0,
            conf_type=conf_type
        )
        if len(fused_boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        # Denormalize
        out_boxes = np.array(fused_boxes).copy()
        out_boxes[:, [0, 2]] = out_boxes[:, [0, 2]] * img_w
        out_boxes[:, [1, 3]] = out_boxes[:, [1, 3]] * img_h
        out = np.column_stack([
            fused_labels.astype(np.int32),
            fused_scores.astype(np.float32),
            out_boxes.astype(np.float32)
        ])
        # Sort by score descending, limit
        order = np.argsort(-out[:, 1], kind="stable")[:100]
        return out[order].astype(np.float32)
    except Exception:
        # Fallback to NMS if ensemble_boxes not available
        all_dets = np.vstack(views_dets) if views_dets else np.zeros((0, 6), dtype=np.float32)
        return _nms_fusion(all_dets, iou_thr=0.5, max_det=100)


class _Model:
    def __init__(self, yolo, device):
        self.yolo, self.device = yolo, device


def load(device="cuda"):
    from ultralytics import YOLO
    return _Model(YOLO(str(WEIGHTS)), device)


def predict(model, image):
    """
    Run 3-view TTA + WBF inference.
    image: numpy array [H, W, 3], RGB, uint8
    Returns: list of (class_id, score, x_min, y_min, x_max, y_max)
    """
    H, W = image.shape[:2]
    bgr = np.ascontiguousarray(image[:, :, ::-1])

    views_dets = []

    # View 1: Original 640
    res = model.yolo.predict(bgr, imgsz=640, conf=CONF_THR, iou=0.7, max_det=300,
                             device=model.device, verbose=False)[0]
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy()
        views_dets.append(np.column_stack([cls, conf, xyxy]).astype(np.float32))
    else:
        views_dets.append(np.zeros((0, 6), dtype=np.float32))

    # View 2: Horizontal Flip 640
    bgr_flip = np.ascontiguousarray(bgr[:, ::-1, :])
    res = model.yolo.predict(bgr_flip, imgsz=640, conf=CONF_THR, iou=0.7, max_det=300,
                             device=model.device, verbose=False)[0]
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy()
        d = np.column_stack([cls, conf, xyxy]).astype(np.float32).copy()
        # Transform boxes back to original coordinates
        d[:, 2] = np.clip(W - d[:, 4], 0.0, float(W))
        d[:, 4] = np.clip(W - d[:, 2], 0.0, float(W))
        views_dets.append(d)
    else:
        views_dets.append(np.zeros((0, 6), dtype=np.float32))

    # View 3: Scale 1.2x (768)
    res = model.yolo.predict(bgr, imgsz=768, conf=CONF_THR, iou=0.7, max_det=300,
                             device=model.device, verbose=False)[0]
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy()
        views_dets.append(np.column_stack([cls, conf, xyxy]).astype(np.float32))
    else:
        views_dets.append(np.zeros((0, 6), dtype=np.float32))

    # Fuse with WBF
    fused = _wbf_fusion(views_dets, W, H, iou_thr=WBF_IOU, conf_type=WBF_CONF_TYPE)

    # Convert to required output format
    if len(fused) == 0:
        return []
    return [(int(c), float(s), float(x1), float(y1), float(x2), float(y2))
            for c, s, x1, y1, x2, y2 in fused]