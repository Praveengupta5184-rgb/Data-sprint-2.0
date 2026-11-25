---
title: ROADVISION AI
emoji: 🛣️
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 7860
pinned: false
---

# ROADVISION AI — Intelligent Road Damage Detection System

**Anveshan DataSprint 2.0 — Computer Vision Track**

A full-stack AI web application running the verified **YOLOv8n + 3-View Test-Time Augmentation (TTA) + Weighted Boxes Fusion (WBF)** road damage detection pipeline.

## Model & Inference Pipeline

- **Architecture:** YOLOv8n (`3.0M` parameters, `8.1` GFLOPs, COCO pretrained)
- **Checkpoint:** `weights/extended_best.pt` (6.5 MB)
- **3-View TTA:**
  1. Original image (`640 × 640`)
  2. Horizontal flip (`640 × 640`) with coordinate reversal
  3. 1.2× scaled view (`768 × 768`)
- **Post-Processing:** Weighted Boxes Fusion (`iou_thr=0.45`, `conf_type="max"`, `conf_thr=0.001`, `max_det=100`)
- **Classes:** `D00` Longitudinal Crack, `D10` Transverse Crack, `D20` Alligator Crack, `D40` Pothole
- **Official Validation Set Metrics (2,150 images):**
  - Validation Score: **43.53 / 100**
  - mAP@50: **0.5230** | mAP@50:95: **0.2156**
  - Precision: **0.5427** | Recall: **0.5185**

## Local Setup & Startup

```bash
cd roadvision-webapp
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000` in your browser.

## API Endpoints

- `GET /` — Web Dashboard UI
- `GET /health` — Backend & model health check
- `POST /api/detect?conf_threshold=0.237` — Run 3-view TTA + WBF inference on uploaded JPG/PNG
- `POST /api/detect/sample/{sample_id}` — Run inference on curated RDD2020 validation sample
- `GET /api/sample-images` — List curated sample images
- `GET /api/model/info` — Model specs & validation metrics
- `GET /api/docs` — Interactive Swagger API documentation

## Public Deployment (Hugging Face Spaces — Docker SDK)

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces) and select **Docker** as the Space SDK (CPU Basic 16GB RAM is free and runs `extended_best.pt` cleanly; `weights/extended_best.pt` is only 6.5 MB so it uploads directly without LFS issues).
2. Clone the Space repo and copy the contents of `roadvision-webapp/` into it.
3. Push to the Space (`git add . && git commit -m "Deploy ROADVISION AI" && git push`).
4. Wait ~3–5 minutes for the Docker build and startup; verify `/health` shows `"model_loaded": true`.

## Attribution

Sample images are from the **RDD2020** dataset (CC BY-NC 3.0) used for the Anveshan DataSprint 2.0 competition.
