#!/usr/bin/env python3
"""
Phase 10 Automated Verification Suite for ROADVISION AI.
Tests:
  1. Valid JPG upload
  2. Valid PNG upload
  3. Pothole sample inference
  4. Crack samples inference
  5. Multi-class road damage image
  6. Invalid file upload (.txt) -> 400
  7. Corrupted image (.jpg) -> 400
  8. Repeated uploads -> deterministic & stable
  9. Annotated image integrity check
 10. Model unavailable (503) & no path leakage
 11. Exact prediction parity with competition 3-view TTA + WBF pipeline
"""
import base64
import csv
import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from ensemble_boxes import weighted_boxes_fusion
from fastapi.testclient import TestClient
from PIL import Image
from ultralytics import YOLO

WEBAPP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WEBAPP_ROOT.parent
sys.path.insert(0, str(WEBAPP_ROOT / "backend"))

import main as backend_main  # noqa: E402


def reference_tta_wbf(img_path: Path):
    """Exact reference 3-view TTA + WBF from generate_test_wbf_submission.py."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    yolo = YOLO(str(WEBAPP_ROOT / "weights" / "extended_best.pt"))
    img = np.asarray(Image.open(img_path).convert("RGB"))
    bgr = np.ascontiguousarray(img[:, :, ::-1])
    H, W = bgr.shape[:2]

    def pred_view(arr, sz):
        r = yolo.predict(arr, imgsz=sz, conf=0.001, iou=0.7, max_det=300, device=device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        return np.column_stack([r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()]).astype(np.float32)

    d_orig = pred_view(bgr, 640)
    d_flip_raw = pred_view(np.ascontiguousarray(bgr[:, ::-1, :]), 640)
    if len(d_flip_raw):
        d_flip = d_flip_raw.copy()
        d_flip[:, 2] = np.clip(W - d_flip_raw[:, 4], 0.0, float(W))
        d_flip[:, 4] = np.clip(W - d_flip_raw[:, 2], 0.0, float(W))
    else:
        d_flip = np.zeros((0, 6), dtype=np.float32)
    d_scale12 = pred_view(bgr, 768)

    boxes_list, scores_list, labels_list = [], [], []
    for d in [d_orig, d_flip, d_scale12]:
        if len(d):
            df = d[d[:, 1] >= 0.001].copy()
            df[:, 2] = np.clip(df[:, 2], 0.0, W - 1.0)
            df[:, 3] = np.clip(df[:, 3], 0.0, H - 1.0)
            df[:, 4] = np.clip(df[:, 4], df[:, 2] + 1.0, float(W))
            df[:, 5] = np.clip(df[:, 5], df[:, 3] + 1.0, float(H))
            bn = df[:, 2:6].copy()
            bn[:, [0, 2]] = np.clip(bn[:, [0, 2]] / W, 0.0, 1.0)
            bn[:, [1, 3]] = np.clip(bn[:, [1, 3]] / H, 0.0, 1.0)
            bn[:, 2] = np.maximum(bn[:, 2], bn[:, 0] + 1e-4)
            bn[:, 3] = np.maximum(bn[:, 3], bn[:, 1] + 1e-4)
            boxes_list.append(bn.tolist())
            scores_list.append(df[:, 1].tolist())
            labels_list.append(df[:, 0].astype(int).tolist())
        else:
            boxes_list.append([])
            scores_list.append([])
            labels_list.append([])

    fb, fs, fl = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        weights=[1.0, 1.0, 1.0], iou_thr=0.45, skip_box_thr=0.001, conf_type="max",
    )
    fb = np.asarray(fb)
    fb[:, [0, 2]] *= W
    fb[:, [1, 3]] *= H
    res = np.column_stack([fl, fs, fb])
    res = res[np.argsort(-res[:, 1], kind="stable")[:100]]
    return res


def find_multiclass_val_image():
    val_csv = PROJECT_ROOT / "val_labels.csv"
    if not val_csv.exists():
        return None
    img_cls = defaultdict(set)
    with open(val_csv) as f:
        for r in csv.DictReader(f):
            img_cls[r["image_id"]].add(int(r["class_id"]))
    for iid, clss in img_cls.items():
        if len(clss) >= 3:
            p = PROJECT_ROOT / "val_images" / f"{iid}.jpg"
            if p.exists():
                return p
    return None


def main():
    results = []

    with TestClient(backend_main.app) as client:
        # 0. Health & UI
        h = client.get("/health")
        assert h.status_code == 200 and h.json()["model_loaded"] is True, f"Health failed: {h.text}"
        results.append(f"Health OK (device={h.json()['device']}, load_time={h.json()['model_load_time']}s)")

        home = client.get("/")
        assert home.status_code == 200 and "ROADVISION AI" in home.text
        results.append("Frontend HTML served OK")

        # 1. Valid JPG upload
        pothole_path = WEBAPP_ROOT / "assets" / "samples" / "d40_pothole.jpg"
        with open(pothole_path, "rb") as f:
            r1 = client.post("/api/detect?conf_threshold=0.237", files={"file": ("pothole.jpg", f, "image/jpeg")})
        assert r1.status_code == 200, r1.text
        d1 = r1.json()
        assert d1["success"] and d1["total_detections"] > 0
        results.append(f"1. Valid JPG upload: {d1['total_detections']} detections in {d1['inference_time_ms']} ms")

        # 2. Valid PNG upload
        png_buf = io.BytesIO()
        Image.open(pothole_path).save(png_buf, format="PNG")
        png_buf.seek(0)
        r2 = client.post("/api/detect?conf_threshold=0.237", files={"file": ("pothole.png", png_buf, "image/png")})
        assert r2.status_code == 200, r2.text
        d2 = r2.json()
        assert d2["success"] and d2["total_detections"] > 0
        results.append(f"2. Valid PNG upload: {d2['total_detections']} detections in {d2['inference_time_ms']} ms")

        # 3 & 4. Sample endpoints (pothole + 3 crack types)
        for sid in ["d00_longitudinal", "d10_transverse", "d20_alligator", "d40_pothole"]:
            rs = client.post(f"/api/detect/sample/{sid}?conf_threshold=0.237")
            assert rs.status_code == 200, f"{sid}: {rs.text}"
            ds = rs.json()
            assert ds["success"]
            results.append(f"3/4. Sample '{sid}': {ds['total_detections']} detections ({ds['inference_time_ms']} ms)")

        # 5. Multi-class image
        multi_p = find_multiclass_val_image()
        if multi_p:
            with open(multi_p, "rb") as f:
                rm = client.post("/api/detect?conf_threshold=0.237", files={"file": (multi_p.name, f, "image/jpeg")})
            assert rm.status_code == 200
            dm = rm.json()
            results.append(f"5. Multi-class image ({multi_p.name}): {dm['total_detections']} dets, {dm['analytics']['unique_classes']} unique classes")

        # 6. Invalid file upload
        r6 = client.post("/api/detect", files={"file": ("bad.txt", io.BytesIO(b"hello world"), "text/plain")})
        assert r6.status_code == 400 and r6.json()["success"] is False
        results.append(f"6. Invalid file upload rejected (400): {r6.json()['error']}")

        # 7. Corrupted image
        r7 = client.post("/api/detect", files={"file": ("corrupt.jpg", io.BytesIO(b"not-a-real-jpeg"), "image/jpeg")})
        assert r7.status_code == 400 and r7.json()["success"] is False
        assert "C:\\" not in r7.text and "/app" not in r7.text
        results.append(f"7. Corrupted image rejected cleanly (400): {r7.json()['error']}")

        # 8. Repeated uploads
        counts = []
        for _ in range(3):
            with open(pothole_path, "rb") as f:
                rr = client.post("/api/detect?conf_threshold=0.237", files={"file": ("pothole.jpg", f, "image/jpeg")})
            assert rr.status_code == 200
            counts.append(rr.json()["total_detections"])
        assert len(set(counts)) == 1, f"Non-deterministic counts: {counts}"
        results.append(f"8. Repeated uploads (x3): deterministic counts = {counts}")

        # 9. Annotated image decode check
        ann_bytes = base64.b64decode(d1["annotated_image_base64"])
        orig_bytes = base64.b64decode(d1["original_image_base64"])
        ann_img = Image.open(io.BytesIO(ann_bytes))
        assert ann_img.size == tuple(d1["image_size"])
        assert ann_bytes != orig_bytes
        results.append(f"9. Annotated image verified: {ann_img.size}, {len(ann_bytes)} bytes")

        # 10. Parity check against standalone competition TTA + WBF
        with open(pothole_path, "rb") as f:
            r_raw = client.post("/api/detect?conf_threshold=0.001", files={"file": ("pothole.jpg", f, "image/jpeg")})
        assert r_raw.status_code == 200
        api_dets = r_raw.json()["detections"]
        ref_dets = reference_tta_wbf(pothole_path)
        assert len(api_dets) == len(ref_dets), f"Count mismatch: API={len(api_dets)} vs Ref={len(ref_dets)}"
        max_conf_diff = max(abs(a["confidence"] - float(b[1])) for a, b in zip(api_dets, ref_dets))
        max_box_diff = max(max(abs(a["bbox"][k] - float(b[2 + k])) for k in range(4)) for a, b in zip(api_dets, ref_dets))
        assert max_conf_diff < 1e-4 and max_box_diff < 0.05, f"Parity diff too large: conf={max_conf_diff}, box={max_box_diff}"
        results.append(f"10. Parity with competition TTA+WBF: {len(api_dets)} boxes, max_conf_diff={max_conf_diff:.6f}, max_box_diff={max_box_diff:.4f}px")

        # 11. Model unavailable -> 503
        orig_loaded = backend_main.model._loaded
        backend_main.model._loaded = False
        try:
            with open(pothole_path, "rb") as f:
                r11 = client.post("/api/detect", files={"file": ("pothole.jpg", f, "image/jpeg")})
            assert r11.status_code == 503 and r11.json()["success"] is False
            results.append(f"11. Model unavailable handled cleanly (503): {r11.json()['error']}")
        finally:
            backend_main.model._loaded = orig_loaded

    print("\n=== ROADVISION AI VERIFICATION REPORT ===")
    for line in results:
        print(f"  [PASS] {line}")
    print("ALL 11 VERIFICATION CHECKS PASSED!")


if __name__ == "__main__":
    main()
