"""
ROADVISION AI - Inference Engine
Reuses the exact verified 3-view TTA + Weighted Boxes Fusion (WBF) pipeline:
  - Model: YOLOv8n (weights/extended_best.pt)
  - Views: Original (640x640), Horizontal Flip (640x640), Scale 1.2x (768x768)
  - Fusion: WBF (iou_thr=0.45, conf_type='max', skip_box_thr=0.001, max_det=100)
"""
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_DEFAULT_WEIGHTS = _ROOT / "weights" / "extended_best.pt"
_FALLBACK_WEIGHTS = _ROOT.parent / "weights" / "extended_best.pt"

# Verified WBF + TTA configuration
WBF_IOU = 0.45
WBF_CONF_TYPE = "max"
CONF_THR = 0.001
YOLO_IOU = 0.7
YOLO_MAX_DET = 300
MAX_DET = 100
DEFAULT_DISPLAY_CONF = 0.237  # F1-optimal threshold from official scorer

CLASS_NAMES: Dict[int, str] = {
    0: "D00 - Longitudinal Crack",
    1: "D10 - Transverse Crack",
    2: "D20 - Alligator Crack",
    3: "D40 - Pothole",
}

CLASS_SHORT: Dict[int, str] = {
    0: "D00",
    1: "D10",
    2: "D20",
    3: "D40",
}

CLASS_TYPES: Dict[int, str] = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
}

# High-contrast hex colors for UI & bounding boxes
CLASS_COLORS: Dict[int, str] = {
    0: "#00D4FF",  # Cyan - D00 Longitudinal Crack
    1: "#FF8C42",  # Orange - D10 Transverse Crack
    2: "#10B981",  # Emerald - D20 Alligator Crack
    3: "#F43F5E",  # Rose Red - D40 Pothole
}


def _resolve_weights_path(weights_path: Optional[str] = None) -> Path:
    if weights_path:
        return Path(weights_path)
    if _DEFAULT_WEIGHTS.exists():
        return _DEFAULT_WEIGHTS
    return _FALLBACK_WEIGHTS


def _nms_fallback(dets: np.ndarray, iou_thr: float = 0.5, max_det: int = MAX_DET) -> np.ndarray:
    """CPU/GPU-safe NMS fallback if ensemble_boxes fails."""
    if len(dets) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    import torchvision.ops as ops
    t_boxes = torch.tensor(dets[:, 2:6], dtype=torch.float32)
    t_scores = torch.tensor(dets[:, 1], dtype=torch.float32)
    t_classes = torch.tensor(dets[:, 0], dtype=torch.int64)
    keep = ops.batched_nms(t_boxes, t_scores, t_classes, iou_thr)[:max_det].numpy()
    out = dets[keep]
    order = np.argsort(-out[:, 1], kind="stable")
    return out[order].astype(np.float32)


def _wbf_fusion(
    views_dets: List[np.ndarray],
    img_w: int,
    img_h: int,
    iou_thr: float = WBF_IOU,
    conf_type: str = WBF_CONF_TYPE,
    conf_thr: float = CONF_THR,
    max_det: int = MAX_DET,
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """Weighted Boxes Fusion across TTA views, matching generate_test_wbf_submission.py."""
    if not views_dets:
        return np.zeros((0, 6), dtype=np.float32)

    n_views = len(views_dets)
    if weights is None:
        weights = [1.0] * n_views

    boxes_list, scores_list, labels_list = [], [], []
    has_dets = False

    for d in views_dets:
        if len(d) > 0:
            mask = d[:, 1] >= conf_thr
            if np.any(mask):
                d_filt = d[mask].copy()
                d_filt[:, 2] = np.clip(d_filt[:, 2], 0.0, max(0.0, img_w - 1.0))
                d_filt[:, 3] = np.clip(d_filt[:, 3], 0.0, max(0.0, img_h - 1.0))
                d_filt[:, 4] = np.clip(d_filt[:, 4], d_filt[:, 2] + 1.0, float(img_w))
                d_filt[:, 5] = np.clip(d_filt[:, 5], d_filt[:, 3] + 1.0, float(img_h))

                b_norm = d_filt[:, 2:6].copy()
                b_norm[:, [0, 2]] = np.clip(b_norm[:, [0, 2]] / img_w, 0.0, 1.0)
                b_norm[:, [1, 3]] = np.clip(b_norm[:, [1, 3]] / img_h, 0.0, 1.0)
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
        return np.zeros((0, 6), dtype=np.float32)

    try:
        from ensemble_boxes import weighted_boxes_fusion
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=conf_thr,
            conf_type=conf_type,
        )
    except Exception:
        all_dets = np.vstack([d for d in views_dets if len(d) > 0])
        return _nms_fallback(all_dets, iou_thr=0.5, max_det=max_det)

    if len(fused_boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    out_boxes = np.asarray(fused_boxes, dtype=np.float32).copy()
    out_boxes[:, [0, 2]] *= float(img_w)
    out_boxes[:, [1, 3]] *= float(img_h)
    out = np.column_stack([
        np.asarray(fused_labels, dtype=np.float32),
        np.asarray(fused_scores, dtype=np.float32),
        out_boxes,
    ])
    order = np.argsort(-out[:, 1], kind="stable")[:max_det]
    return out[order].astype(np.float32)


class RoadVisionModel:
    """Thread-safe YOLOv8n wrapper with 3-view TTA + WBF inference."""

    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            if device.startswith("cuda") and not torch.cuda.is_available():
                self.device = "cpu"
            else:
                self.device = device
        self.weights_path = _resolve_weights_path(weights_path)
        self.model: Optional[YOLO] = None
        self._loaded = False
        self.load_time = 0.0
        self._lock = threading.Lock()

    def load(self) -> bool:
        """Load the YOLOv8n checkpoint once and warm up."""
        with self._lock:
            if self._loaded:
                return True
            if not self.weights_path.exists():
                self._loaded = False
                return False

            start = time.perf_counter()
            try:
                self.model = YOLO(str(self.weights_path))
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = self.model.predict(
                    dummy,
                    imgsz=640,
                    conf=CONF_THR,
                    iou=YOLO_IOU,
                    max_det=YOLO_MAX_DET,
                    device=self.device,
                    verbose=False,
                )
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()
                self._loaded = True
                self.load_time = round(time.perf_counter() - start, 3)
                return True
            except Exception:
                self._loaded = False
                self.model = None
                return False

    def _predict_view(self, bgr: np.ndarray, imgsz: int) -> np.ndarray:
        res = self.model.predict(
            bgr,
            imgsz=imgsz,
            conf=CONF_THR,
            iou=YOLO_IOU,
            max_det=YOLO_MAX_DET,
            device=self.device,
            verbose=False,
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy()
        return np.column_stack([cls, conf, xyxy]).astype(np.float32)

    def predict(self, image: np.ndarray) -> Tuple[List[dict], float]:
        """
        Run 3-view TTA + WBF inference on an RGB uint8 numpy image.
        Returns: (detections_list, total_inference_time_ms)
        """
        if not self._loaded or self.model is None:
            raise RuntimeError("Model is not loaded.")

        H, W = image.shape[:2]
        bgr = np.ascontiguousarray(image[:, :, ::-1])

        with self._lock:
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            # View 1: Original 640x640
            d_orig = self._predict_view(bgr, 640)

            # View 2: Horizontal Flip 640x640 + non-destructive coordinate reversal
            bgr_flip = np.ascontiguousarray(bgr[:, ::-1, :])
            d_flip_raw = self._predict_view(bgr_flip, 640)
            if len(d_flip_raw) > 0:
                d_flip = d_flip_raw.copy()
                d_flip[:, 2] = np.clip(W - d_flip_raw[:, 4], 0.0, float(W))
                d_flip[:, 4] = np.clip(W - d_flip_raw[:, 2], 0.0, float(W))
                d_flip[:, 3] = np.clip(d_flip_raw[:, 3], 0.0, float(H))
                d_flip[:, 5] = np.clip(d_flip_raw[:, 5], 0.0, float(H))
            else:
                d_flip = np.zeros((0, 6), dtype=np.float32)

            # View 3: Scale 1.2x (768x768)
            d_scale12 = self._predict_view(bgr, 768)

            # WBF across the 3 views
            fused = _wbf_fusion(
                [d_orig, d_flip, d_scale12],
                img_w=W,
                img_h=H,
                iou_thr=WBF_IOU,
                conf_type=WBF_CONF_TYPE,
                conf_thr=CONF_THR,
                max_det=MAX_DET,
            )

            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_time_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[dict] = []
        for c, s, x1, y1, x2, y2 in fused:
            cls_id = int(c)
            detections.append({
                "class_id": cls_id,
                "class_code": CLASS_SHORT.get(cls_id, f"C{cls_id}"),
                "damage_type": CLASS_TYPES.get(cls_id, f"Class {cls_id}"),
                "class_name": CLASS_NAMES.get(cls_id, f"Class {cls_id}"),
                "confidence": float(s),
                "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)],
            })

        return detections, round(inference_time_ms, 2)


def draw_detections(image: np.ndarray, detections: List[dict]) -> np.ndarray:
    """Draw crisp, high-contrast bounding boxes and class labels onto an RGB image."""
    annotated = image.copy()
    H, W = annotated.shape[:2]
    scale = max(0.45, min(1.0, min(W, H) / 720.0))
    thickness = max(2, int(round(2.5 * scale)))
    font_scale = max(0.5, 0.65 * scale)
    font_thick = max(1, int(round(1.8 * scale)))

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))

        cls_id = det["class_id"]
        conf = det["confidence"]
        hex_color = CLASS_COLORS.get(cls_id, "#FFFFFF").lstrip("#")
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), rgb, thickness)

        code = CLASS_SHORT.get(cls_id, f"C{cls_id}")
        label = f"{code} {conf * 100:.1f}%"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)

        box_y1 = max(th + baseline + 8, y1)
        cv2.rectangle(
            annotated,
            (x1, box_y1 - th - baseline - 6),
            (min(W - 1, x1 + tw + 10), box_y1),
            rgb,
            -1,
        )
        text_color = (10, 18, 36) if cls_id in (0, 1, 2) else (255, 255, 255)
        cv2.putText(
            annotated,
            label,
            (x1 + 5, box_y1 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            font_thick,
            cv2.LINE_AA,
        )

    return annotated
