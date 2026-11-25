"""
ROADVISION AI - FastAPI Backend
Serves the YOLOv8n + 3-view TTA + WBF inference API and the ROADVISION AI web UI.
"""
import asyncio
import base64
import io
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from inference_engine import (  # noqa: E402
    CLASS_COLORS,
    CLASS_NAMES,
    CLASS_SHORT,
    CLASS_TYPES,
    CONF_THR,
    DEFAULT_DISPLAY_CONF,
    MAX_DET,
    WBF_CONF_TYPE,
    WBF_IOU,
    RoadVisionModel,
    draw_detections,
)

FRONTEND_DIR = _ROOT / "frontend"
SAMPLES_DIR = _ROOT / "assets" / "samples"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

SAMPLE_METADATA: Dict[str, dict] = {
    "d00_longitudinal": {
        "id": "d00_longitudinal",
        "filename": "d00_longitudinal.jpg",
        "class_code": "D00",
        "title": "Longitudinal Crack",
        "subtitle": "Validation Sample (RDD2020)",
    },
    "d10_transverse": {
        "id": "d10_transverse",
        "filename": "d10_transverse.jpg",
        "class_code": "D10",
        "title": "Transverse Crack",
        "subtitle": "Validation Sample (RDD2020)",
    },
    "d20_alligator": {
        "id": "d20_alligator",
        "filename": "d20_alligator.jpg",
        "class_code": "D20",
        "title": "Alligator Crack",
        "subtitle": "Validation Sample (RDD2020)",
    },
    "d40_pothole": {
        "id": "d40_pothole",
        "filename": "d40_pothole.jpg",
        "class_code": "D40",
        "title": "Pothole",
        "subtitle": "Validation Sample (RDD2020)",
    },
}

model = RoadVisionModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = model.load()
    status_str = f"ONLINE on {model.device} ({model.load_time:.2f}s)" if loaded else "OFFLINE (weights failed to load)"
    print(f"[ROADVISION AI] Model status: {status_str}")
    yield


app = FastAPI(
    title="ROADVISION AI",
    description="Intelligent Road Damage Detection System (YOLOv8n + 3-View TTA + WBF)",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")


class Detection(BaseModel):
    class_id: int
    class_code: str
    damage_type: str
    class_name: str
    confidence: float
    bbox: List[float]


class DetectionResponse(BaseModel):
    success: bool
    detections: List[Detection]
    total_detections: int
    raw_detections_count: int
    conf_threshold: float
    inference_time_ms: float
    image_size: List[int]
    annotated_image_base64: str
    original_image_base64: str
    analytics: dict
    model_info: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_load_time: float
    device: str


def _encode_rgb_to_base64(rgb: np.ndarray) -> str:
    bgr = rgb[:, :, ::-1]
    ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise ValueError("Failed to encode output image")
    return base64.b64encode(buffer).decode("utf-8")


def _compute_analytics(detections: List[dict]) -> dict:
    counts = {str(cid): 0 for cid in range(4)}
    confs = []
    for det in detections:
        cid = str(det["class_id"])
        if cid in counts:
            counts[cid] += 1
        confs.append(det["confidence"])

    unique_classes = sum(1 for v in counts.values() if v > 0)
    avg_conf = round(float(np.mean(confs)), 4) if confs else 0.0

    distribution = [
        {
            "class_id": cid,
            "class_code": CLASS_SHORT[cid],
            "damage_type": CLASS_TYPES[cid],
            "color": CLASS_COLORS[cid],
            "count": counts[str(cid)],
        }
        for cid in range(4)
    ]
    return {
        "total_objects": len(detections),
        "unique_classes": unique_classes,
        "average_confidence": avg_conf,
        "distribution": distribution,
    }


def _run_pipeline_on_bytes(raw_bytes: bytes, conf_threshold: float) -> DetectionResponse:
    if not model._loaded:
        raise HTTPException(status_code=503, detail="AI model is currently unavailable.")

    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File size exceeds 10 MB limit.")

    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img.verify()
        with Image.open(io.BytesIO(raw_bytes)) as img:
            rgb_img = img.convert("RGB")
            image_np = np.asarray(rgb_img)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file. Please upload a valid JPG or PNG.")

    if image_np.ndim != 3 or image_np.shape[0] < 16 or image_np.shape[1] < 16:
        raise HTTPException(status_code=400, detail="Image dimensions are too small.")

    try:
        raw_detections, inference_time_ms = model.predict(image_np)
    except Exception:
        raise HTTPException(status_code=500, detail="Inference failed while processing the image.")

    filtered = [d for d in raw_detections if d["confidence"] >= conf_threshold]
    annotated = draw_detections(image_np, filtered)

    orig_b64 = _encode_rgb_to_base64(image_np)
    ann_b64 = _encode_rgb_to_base64(annotated)
    analytics = _compute_analytics(filtered)

    return DetectionResponse(
        success=True,
        detections=[Detection(**d) for d in filtered],
        total_detections=len(filtered),
        raw_detections_count=len(raw_detections),
        conf_threshold=conf_threshold,
        inference_time_ms=inference_time_ms,
        image_size=[int(image_np.shape[1]), int(image_np.shape[0])],
        annotated_image_base64=ann_b64,
        original_image_base64=orig_b64,
        analytics=analytics,
        model_info={
            "name": "YOLOv8n + 3-View TTA + WBF",
            "architecture": "YOLOv8n",
            "parameters": "3.0M",
            "gflops": 8.1,
            "input_resolution": "640x640",
            "pretraining": "COCO",
            "inference_strategy": "3-view TTA (Original 640, H-Flip 640, Scale 1.2x 768)",
            "post_processing": "Weighted Boxes Fusion",
            "wbf_iou": WBF_IOU,
            "wbf_conf_type": WBF_CONF_TYPE,
            "device": model.device,
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model._loaded else "degraded",
        model_loaded=model._loaded,
        model_load_time=model.load_time,
        device=model.device,
    )


@app.post("/api/detect", response_model=DetectionResponse)
async def detect_damage(
    file: UploadFile = File(...),
    conf_threshold: float = Query(DEFAULT_DISPLAY_CONF, ge=0.001, le=0.99),
):
    allowed_types = {"image/jpeg", "image/jpg", "image/png", "application/octet-stream"}
    ext = Path(file.filename or "").suffix.lower()
    if file.content_type not in allowed_types and ext not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=400, detail="Unsupported file format. Only JPG, JPEG, and PNG images are allowed.")
    if ext and ext not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=400, detail="Unsupported file extension. Only JPG, JPEG, and PNG images are allowed.")

    content = await file.read()
    return await asyncio.to_thread(_run_pipeline_on_bytes, content, conf_threshold)


@app.post("/api/detect/sample/{sample_id}", response_model=DetectionResponse)
async def detect_sample(
    sample_id: str,
    conf_threshold: float = Query(DEFAULT_DISPLAY_CONF, ge=0.001, le=0.99),
):
    meta = SAMPLE_METADATA.get(sample_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Sample image not found.")
    sample_path = SAMPLES_DIR / meta["filename"]
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample image file is unavailable.")
    raw_bytes = sample_path.read_bytes()
    return await asyncio.to_thread(_run_pipeline_on_bytes, raw_bytes, conf_threshold)


@app.get("/api/sample-images")
async def list_sample_images():
    samples = []
    for meta in SAMPLE_METADATA.values():
        p = SAMPLES_DIR / meta["filename"]
        if p.exists() and p.is_file():
            samples.append({
                **meta,
                "url": f"/api/sample-image/{meta['filename']}",
            })
    return {"samples": samples}


@app.get("/api/sample-image/{filename}")
async def get_sample_image(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    file_path = (SAMPLES_DIR / filename).resolve()
    if not str(file_path).startswith(str(SAMPLES_DIR.resolve())) or not file_path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found.")
    return FileResponse(file_path)


@app.get("/api/model/info")
async def model_info():
    return {
        "architecture": "YOLOv8n",
        "parameters": "3.0M",
        "gflops": 8.1,
        "input_resolution": "640x640",
        "pretraining": "COCO",
        "inference_strategy": "3-view TTA",
        "tta_views": ["Original (640x640)", "Horizontal Flip (640x640)", "Scale 1.2x (768x768)"],
        "post_processing": "Weighted Boxes Fusion",
        "wbf_config": {
            "iou_threshold": WBF_IOU,
            "conf_type": WBF_CONF_TYPE,
            "conf_threshold": CONF_THR,
            "max_detections": MAX_DET,
            "default_display_conf": DEFAULT_DISPLAY_CONF,
        },
        "classes": CLASS_NAMES,
        "class_colors": CLASS_COLORS,
        "validation_metrics": {
            "note": "Official validation set results (2,150 images), not Kaggle leaderboard results",
            "accuracy_score": 43.53,
            "mAP50": 0.5230,
            "mAP50_95": 0.2156,
            "precision": 0.5427,
            "recall": 0.5185,
            "f1": 0.5293,
            "median_latency_ms": 34.89,
        },
        "device": model.device,
        "model_loaded": model._loaded,
        "load_time_seconds": model.load_time,
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>ROADVISION AI</h1><p>Frontend assets not found.</p>")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"success": False, "error": exc.detail})


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"success": False, "error": "Internal server error."})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
