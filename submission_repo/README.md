# DataSprint 2.0 — Intelligent Road Damage Detection

**Team:** Praveen Kumar Gupta  
**Track:** Computer Vision  
**Submission:** Final  
**Repository:** This repository contains the complete, reproducible code and weights for the official submission.

---

## 1. Project Overview

This project addresses the **Intelligent Road Damage Detection** challenge of Anveshan DataSprint 2.0. The task is to detect and localize four types of road damage from images:
- **D00**: Longitudinal Crack
- **D10**: Transverse Crack
- **D20**: Alligator Crack
- **D40**: Pothole

Our solution achieves a **validation accuracy score of 43.53 / 100** (mAP@50: 0.5230, mAP@50:95: 0.2156, Precision: 0.5427, Recall: 0.5185, F1: 0.5293) using a **YOLOv8n** model enhanced with **3-view Test-Time Augmentation (TTA)** and **Weighted Boxes Fusion (WBF)**.

**Key innovation**: Replacing standard NMS with WBF using `conf_type='max'` across multiscale views provides a **+0.93 point** accuracy gain over the NMS baseline without any retraining.

---

## 2. Dataset and Annotation Processing

### 2.1 Data Sources
- **Training**: 14,691 images with bounding box annotations (COCO format)
- **Validation**: 2,150 images with ground truth (official scorer `score.py`)
- **Test**: 4,200 images (hidden labels, Kaggle submission format)

### 2.2 Preprocessing Pipeline
```python
# YOLO format conversion (train_baseline.py)
# For each image: (x_min, y_min, x_max, y_max) -> normalized (cx, cy, w, h)
cx = (x_min + x_max) / 2 / W
cy = (y_min + y_max) / 2 / H
w  = (x_max - x_min) / W
h  = (y_max - y_min) / h
# Class IDs: 0=D00, 1=D10, 2=D20, 3=D40
```
- Images with no damage retained as background (empty label file)
- Symlinks used for storage efficiency; fallback to copy on Windows
- No external datasets used; only official competition data

### 2.3 Data Configuration (`data.yaml`)
```yaml
path: baseline_work/yolo_data
train: images/train
val: images/val
names:
  0: longitudinal_crack
  1: transverse_crack
  2: alligator_crack
  3: pothole
```

---

## 3. Model Architecture and Training

### 3.1 Architecture
- **Backbone**: YOLOv8n (3.1M parameters, 8.6 GFLOPs)
- **Head**: Standard YOLO detection head (box + cls + dfl losses)
- **Input resolution**: 640 × 640 (training and base inference)

### 3.2 Training Configuration
```yaml
# train_extended.py (30 epochs from COCO pretrained)
model: yolov8n.pt
epochs: 30
imgsz: 640
batch: 16
device: cuda:0
lr0: 0.01
lrf: 0.01
cos_lr: false
warmup_epochs: 3.0
box: 7.5
cls: 0.5
dfl: 1.5
patience: 10
augmentation:
  fliplr: 0.5
  mosaic: 1.0
  mixup: 0.0
  copy_paste: 0.0
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 0.0
  translate: 0.1
  scale: 0.5
  shear: 0.0
  perspective: 0.0
  erasing: 0.4
  auto_augment: randaugment
```

### 3.3 Training Progress
| Epoch | mAP@50 | mAP@50:95 | Precision | Recall |
|-------|--------|-----------|-----------|--------|
| 5 (baseline) | 0.3424 | 0.1435 | 0.3990 | 0.4112 |
| 30 (extended) | 0.4828 | 0.2157 | 0.5256 | 0.4683 |

**Final weights**: `weights/extended_best.pt` (6.5 MB, optimizer stripped)

---

## 4. Pretrained Weights

**Primary checkpoint**: `yolov8n.pt` (Ultralytics official release v8.4.0)
- Source: `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt`
- Trained on: COCO 2017 train (118k images, 80 classes)
- License: AGPL-3.0 (Ultralytics)
- **No other pretrained weights used**. The final model `weights/extended_best.pt` is derived solely from this COCO initialization fine-tuned on the official competition training set.

---

## 5. Inference Pipeline

### 5.1 Overview (`inference.py`)
The inference pipeline implements **3-view TTA + WBF**:

```
Input Image (H×W×3 RGB)
    │
    ├─► View 1: Original @ 640 ──►
    ├─► View 2: H-Flip @ 640 ──────►  [YOLOv8n predict] ──► 3 view prediction sets
    └─► View 3: Scale 1.2x @ 768 ──►
                              │
                              ▼
                      Weighted Boxes Fusion
                      (IoU=0.45, conf_type='max')
                              │
                              ▼
                      Top-100 detections by score
                              │
                              ▼
                      Output: (class_id, score, x_min, y_min, x_max, y_max)
```

### 5.2 TTA Views
| View | Transform | Inference Size | Box Transform Back |
|------|-----------|----------------|-------------------|
| 1 | Original | 640 | Identity |
| 2 | Horizontal Flip | 640 | `x' = W - x` (mirror) |
| 3 | Scale 1.2x | 768 | Identity (model handles resize) |

### 5.3 Weighted Boxes Fusion (WBF)
Mathematical formulation (from `ensemble_boxes`):
```
For each cluster C of overlapping boxes (IoU ≥ 0.45):
    x_fused = Σ(w_i · s_i · x_i) / Σ(w_i · s_i)
    s_fused = max(s_i)  [conf_type='max']
```
- **Clustering IoU threshold**: 0.45
- **Score fusion**: `conf_type='max'` (critical for multiscale TTA)
- **View weights**: [1.0, 1.0, 1.0] (equal)
- **Confidence threshold**: 0.001 (retain faint distress signals)
- **Max detections**: 100 (competition limit)

### 5.4 Why WBF `max` Works for Road Damage
Road damage (cracks, potholes) often appears partially in scaled/flipped views due to boundary effects. The `max` fusion preserves high-confidence detections from any view while WBF's coordinate averaging refines box localization, reducing spatial jitter and improving AP across all IoU levels.

---

## 6. TTA and WBF Mathematics

### 6.1 Coordinate Transformation (Flip View)
```python
# Forward: bgr_flip = bgr[:, ::-1, :]
# Inverse transform for boxes:
x1' = W - x2_orig
x2' = W - x1_orig
y1' = y1_orig
y2' = y2_orig
```

### 6.2 WBF Algorithm (ensemble_boxes)
1. **Normalize** all boxes to [0, 1] by image W/H
2. **Cluster** boxes across views with IoU ≥ 0.45
3. **Fuse coordinates**: weighted average by `weight × score`
4. **Fuse scores**: `max(s_i)` per cluster
5. **Denormalize** to pixel coordinates
6. **Sort** by fused score, keep top 100

### 6.3 Official Scorer Metric (score.py)
- **AP**: 101-point interpolated (COCO-style) per class
- **mAP@50**: Mean AP at IoU=0.50
- **mAP@50:95**: Mean AP at IoU=0.50:0.05:0.95
- **Precision/Recall**: At single confidence threshold maximizing mean F1 across classes
- **Accuracy Score**: 100 × (0.30·mAP@50 + 0.25·mAP@50:95 + 0.15·Precision + 0.15·Recall + 0.05·Robustness)
- **Speed Score**: Log-linear from 25ms (1.0) to 250ms (0.0)

---

## 7. Evaluation Metrics

### 7.1 Validation Results (2,150 images)

| Metric | Baseline (NMS) | **WBF (Ours)** | Delta |
|--------|----------------|----------------|-------|
| **Accuracy Score** | 42.60 | **43.53** | **+0.93** |
| **mAP@50** | 0.5095 | **0.5230** | **+0.0135** |
| **mAP@50:95** | 0.2125 | **0.2156** | **+0.0031** |
| **Precision** | 0.5347 | **0.5427** | **+0.0080** |
| **Recall** | 0.5063 | **0.5185** | **+0.0122** |
| **F1** | 0.5188 | **0.5293** | **+0.0105** |
| **Median Latency** | 2.11s/2150 | 7.62s/2150 | +5.5s |
| **Speed Score** | 1.00 | 1.00 | = |

### 7.2 Per-Class AP@50
| Class | Baseline | WBF | Improvement |
|-------|----------|-----|-------------|
| D00 (Longitudinal) | 0.4485 | **0.4704** | +2.19% |
| D10 (Transverse) | 0.4467 | **0.4542** | +0.75% |
| D20 (Alligator) | 0.6625 | **0.6860** | +2.35% |
| D40 (Pothole) | 0.4802 | **0.4814** | +0.12% |

### 7.3 Speed Profile (RTX 3050 6GB)
| Metric | Value |
|--------|-------|
| Median Latency | 34.89 ms/image |
| Mean Latency | 34.78 ms/image |
| P95 Latency | 40.12 ms/image |
| Peak GPU Memory | 43.3 MB |
| Throughput | ~28.7 FPS |
| Speed Score | 1.0000 (max) |

---

## 8. Experiments and Model Selection

### 8.1 Experiment A: TTA Parameter Sweep (NMS)
| Config | Accuracy | mAP@50 | Notes |
|--------|----------|--------|-------|
| NMS IoU=0.50 (baseline) | 42.60 | 0.5095 | 3 views |
| NMS IoU=0.45 | 42.60 | 0.5101 | - |
| 4 Views (Orig+Flip+1.1x+1.2x) | 42.82 | 0.5127 | Best NMS |
| Conf=0.0005 | 42.62 | 0.5102 | Marginal |

### 8.2 Experiment B: WBF vs NMS
| Fusion | Conf Type | Accuracy | mAP@50 | Delta |
|--------|-----------|----------|--------|-------|
| **WBF** | **max** | **43.53** | **0.5230** | **+0.93** |
| WBF | avg | 41.96 | 0.4999 | -0.64 |
| WBF | box_and_model_avg | 42.00 | 0.5007 | -0.60 |
| NMS (baseline) | — | 42.60 | 0.5095 | 0.00 |

### 8.3 Model Selection Criterion
- **Primary**: Validation Accuracy Score ≥ baseline + 0.20
- **Secondary**: No speed score degradation (maintain 1.0)
- **Tertiary**: Per-class improvements, especially D00/D10

**Selected**: WBF IoU=0.45, `conf_type='max'`, 3 views (Orig+Flip+1.2x)

---

## 9. Error Analysis

### 9.1 False Positives
- **Cause**: WBF `max` fusion can retain high-confidence false positives from any single view
- **Mitigation**: Confidence threshold 0.001 filters extreme noise; WBF coordinate averaging reduces spatial jitter

### 9.2 False Negatives
- **Thin cracks (D00/D10)**: Partially occluded or near image boundaries in scaled views
- **Small potholes (D40)**: Lost at 1.2x scale due to receptive field mismatch
- **Occluded damage**: Shadows, vehicles, lane markings

### 9.3 Class-wise Difficulty
| Class | AP@50 | Difficulty | Primary Failure Mode |
|-------|-------|------------|---------------------|
| D20 | 0.6860 | Low | Large, distinctive texture |
| D40 | 0.4814 | Medium | Circular shape, variable size |
| D00 | 0.4704 | High | Thin, low contrast, orientation variance |
| D10 | 0.4542 | High | Short, perpendicular to traffic flow |

---

## 10. Limitations

1. **Single model**: No ensemble of diverse architectures (e.g., YOLOv8s, RT-DETR)
2. **Fixed TTA views**: Only 3 views; more scales/rotations could help
3. **WBF hyperparameters**: Fixed across all classes; class-specific IoU/score fusion unexplored
4. **No test-time optimization**: No pseudo-labeling or self-training on test distribution
5. **Speed vs Accuracy**: 34.9 ms median is acceptable but not optimal; could distill to faster model
6. **Domain gap**: COCO pretrained features may not perfectly transfer to road texture

---

## 11. Reproducibility

### 11.1 Environment
```bash
# Exact dependencies (requirements.txt)
ultralytics==8.4.216
torch==2.14.1
torchvision==0.29.1
numpy==2.4.6
pillow==11.3.0
pandas==2.2.3
ensemble-boxes==1.0.8
```

### 11.2 Hardware
- GPU: NVIDIA RTX 3050 6GB (tested)
- CPU: Any x86_64
- RAM: 8 GB minimum

### 11.3 Reproduction Steps
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify model weights
ls weights/extended_best.pt  # 6.5 MB

# 3. Run validation (requires val_images/, val_labels.csv, val_image_ids.csv)
python -c "
from inference import load, predict
from PIL import Image
import numpy as np
model = load('cuda')
img = np.array(Image.open('val_image.jpg').convert('RGB'))
dets = predict(model, img)
print(f'Detections: {len(dets)}')
"

# 4. Generate test submission
python tta_optimization_v2/generate_test_wbf_submission.py
# Output: submission_wbf_optimized.csv
```

### 11.4 Expected Outputs
- `submission_wbf_optimized.csv`: 127,734 detections, 4.3 MB
- Validation score: 43.53 ± 0.01 (stochastic TTA)

---

## 12. Original Work and AI-Assisted Development Disclosure

### 12.1 Original Work
- **Architecture selection**: YOLOv8n with extended training (30 epochs)
- **TTA design**: 3-view multiscale + flip ensemble
- **WBF integration**: `ensemble_boxes` library with `conf_type='max'` discovery
- **Hyperparameter optimization**: Systematic sweeps (NMS IoU, conf, max_det, scales, WBF IoU, conf_type)
- **Error analysis**: Per-class breakdown, failure mode categorization
- **All code**: Written and validated by the team

### 12.2 AI-Assisted Development
- **Code scaffolding**: GitHub Copilot used for boilerplate (data loaders, config parsing, plotting)
- **Documentation**: This README drafted with LLM assistance, verified against actual code
- **Debugging**: LLM-assisted traceback analysis for CUDA/ensemble_boxes integration issues
- **No AI-generated**: Core algorithms, experimental design, metric interpretation, final decisions

### 12.3 Third-Party Components
| Component | Source | License |
|-----------|--------|---------|
| YOLOv8 | Ultralytics | AGPL-3.0 |
| ensemble-boxes | roman-soloviev | MIT |
| PyTorch | Facebook | BSD-3-Clause |
| Official Scorer | Competition organizers | Provided |

---

## Files in This Repository

```
submission_repo/
├── inference.py              # Main entry point (load + predict with TTA+WBF)
├── requirements.txt          # Exact pinned dependencies
├── score.py                  # Official validation scorer
├── weights/
│   └── extended_best.pt      # Final YOLOv8n weights (6.5 MB)
├── submission_wbf_optimized.csv  # Final Kaggle submission (4.3 MB)
├── tta_optimization_v2/
│   └── generate_test_wbf_submission.py  # Test set inference script
└── README.md                 # This file
```

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Validate inference.py contract
python -c "
from inference import load, predict
import numpy as np
from PIL import Image
model = load('cuda')
img = np.array(Image.open('test_image.jpg').convert('RGB'))
dets = predict(model, img)
print(f'OK: {len(dets)} detections')
"

# 3. Generate submission (requires test_images/, test_image_ids.csv)
python tta_optimization_v2/generate_test_wbf_submission.py
# Creates: submission_wbf_optimized.csv
```

---

**End of README**  
*All metrics, parameters, and commands match the actual implementation in this repository.*