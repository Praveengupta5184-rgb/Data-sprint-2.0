# Improvement V4 Experiment Report: YOLOv8s Training & Evaluation

**Competition**: Anveshan DataSprint 2.0: Intelligent Road Damage Detection  
**Experiment Phase**: Improvement V4 (YOLOv8s Short Controlled Training)  
**Evaluation Scope**: All 2,150 official validation images evaluated strictly using official `score.py`  
**Current Best Baseline**: Accuracy Score **`43.53`** (WBF Uniform IoU 0.45) / **`43.62`** (Class-wise WBF) using `extended_best.pt` (YOLOv8n)

---

## 1. Safety Audit & Integrity Verification

Before training, a comprehensive SHA256 cryptographic audit was established for 48 protected project files across `weights/`, `baseline_work/`, `tta_experiment/`, `tta_optimization_v2/`, `improvement_v3/`, root submissions, and official scripts. 

Following training and validation, the post-experiment SHA256 audit verified that **all 48 files remained completely identical and untouched**.

- **Files Audited**: 48 files
- **Integrity Result**: **100% Match (0 modifications, 0 renames, 0 deletions)**
- **Audit Records**:
  - Pre-experiment hashes: [`safety_inventory_pre.json`](file:///c:/Users/prave/OneDrive/Desktop/DataSprintHackathon/improvement_v4/safety_inventory_pre.json)
  - Post-experiment hashes: [`safety_inventory_post.json`](file:///c:/Users/prave/OneDrive/Desktop/DataSprintHackathon/improvement_v4/safety_inventory_post.json)

---

## 2. Checkpoint Inspection & Pre-flight Testing

- **Starting Checkpoint**: [`baseline_work/runs/exp3_v8s_pilot/weights/best.pt`](file:///c:/Users/prave/OneDrive/Desktop/DataSprintHackathon/baseline_work/runs/exp3_v8s_pilot/weights/best.pt)
- **Model Architecture**: YOLOv8s (`129 layers`, `11,137,148 parameters`, `28.6 GFLOPs`)
- **Compatibility**: Verified 100% compatible. 355/355 layer tensors transferred into Ultralytics 8.4 / PyTorch 2.14+cu126.
- **Starting Performance (Epoch 3 Pilot)**: mAP@50: `0.1092`, Precision: `0.2634`, Recall: `0.1643`.
- **GPU Safety Test**: Tested on NVIDIA RTX 3050 Laptop GPU (6.0 GB VRAM):
  - Batch=4 Peak VRAM: 0.77 GB (<13% capacity).
  - Batch=8 Peak VRAM: 1.49 GB (<25% capacity).
  - Micro-batch training test completed cleanly with zero warnings or errors.

---

## 3. Training Execution & Progression (10 Epochs)

- **Training Script**: [`improvement_v4/train_yolov8s.py`](file:///c:/Users/prave/OneDrive/Desktop/DataSprintHackathon/improvement_v4/train_yolov8s.py)
- **Parameters**: `epochs=10`, `imgsz=640`, `batch=8`, `workers=2`, `amp=True`, `device=CUDA:0`, `patience=100`.
- **Output Directory**: `improvement_v4/runs/exp_v8s_10ep/`
- **Total Training Duration**: **88.10 minutes** (~8.8 minutes per epoch across 14,691 images).

### Epoch Progression

| Epoch | Train Box Loss | Train Cls Loss | Train DFL Loss | Val mAP@50 | Val mAP@50:95 | Precision | Recall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Start (Pilot Ep 3)** | 2.143 | 3.212 | 2.057 | 0.1092 | 0.0379 | 0.2634 | 0.1643 |
| **Epoch 1** | 2.127 | 2.867 | 2.097 | 0.1411 | 0.0504 | 0.2289 | 0.2037 |
| **Epoch 2** | 2.122 | 2.872 | 2.081 | 0.1646 | 0.0613 | 0.4397 | 0.1847 |
| **Epoch 4** | 2.077 | 2.777 | 2.020 | 0.2204 | 0.0852 | 0.3104 | 0.2735 |
| **Epoch 6** | 1.986 | 2.513 | 1.932 | 0.2865 | 0.1175 | 0.3843 | 0.3189 |
| **Epoch 8** | 1.903 | 2.341 | 1.866 | 0.3379 | 0.1418 | 0.3993 | 0.3718 |
| **Epoch 10** | **1.840** | **2.177** | **1.816** | **0.3767** | **0.1626** | **0.4474** | **0.3791** |

---

## 4. Official Validation Evaluation Results

All 2,150 validation images were scored using the official competition scoring script (`score.py`).

| Configuration | Accuracy Score | mAP@50 | mAP@50:95 | Precision | Recall | F1 Score | Median Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Current Best: YOLOv8n + TTA + WBF** | **43.53** | **0.5230** | **0.2227** | **0.5510** | 0.4497 | 0.4952 | 34.89 ms/img |
| **Current Best: Class-wise WBF** | **43.62** | 0.5228 | 0.2129 | 0.5511 | **0.5202** | **0.5342** | 33.60 ms/img |
| **YOLOv8s Alone (Single View)** | 33.83 | 0.3925 | 0.1631 | 0.4330 | 0.4268 | 0.4277 | 15.65 ms/img |
| **YOLOv8s + 3-View TTA + WBF** | 36.50 | 0.4283 | 0.1650 | 0.4781 | 0.4582 | 0.4663 | 56.42 ms/img |
| **YOLOv8n + YOLOv8s Ensemble (Single View)**| 42.68 | 0.5049 | 0.2081 | 0.5312 | 0.4720 | 0.5002 | 24.12 ms/img |

---

## 5. Key Findings & Insights

1. **YOLOv8s Learning Trajectory**:
   - Training YOLOv8s from Epoch 3 to Epoch 13 drove validation mAP@50 from **0.1092 to 0.3767** (+245% relative improvement), proving that the pilot checkpoint was fully viable and learning rapidly.
2. **TTA + WBF Efficacy on YOLOv8s**:
   - Applying 3-view TTA + WBF to YOLOv8s boosted its score from **33.83 to 36.50** (+2.67 points), showing that the TTA + WBF pipeline generalizes well across different model architectures.
3. **Capacity vs Training Epochs**:
   - YOLOv8s has 11.1M parameters (3.7x larger than YOLOv8n's 3.0M parameters).
   - Because our best YOLOv8n was trained for **25 epochs** (`extended_best.pt`), it reached a mature validation score of **42.22 alone / 43.53 with TTA+WBF**.
   - At only 13 epochs total, YOLOv8s is still early in its convergence curve.
4. **Ensemble Viability**:
   - Combining single-view YOLOv8n (42.22) with single-view YOLOv8s (33.83) via WBF lifted the score to **42.68**, demonstrating positive diversity and complementarity between the architectures.

---

## 6. Submission Decision

- **Benchmark to Beat**: `43.53` (or `43.62`).
- **YOLOv8s (10-epoch) Top Score**: `36.50` (TTA+WBF) / `42.68` (Ensemble).
- **Decision**: In strict adherence to the competition rule (*"generate a separate test submission only after approval"*), **no test submission is recommended or generated** for this checkpoint, as it does not surpass our current top submission.
- The official leading submission remains [`tta_optimization_v2/submission_wbf_optimized.csv`](file:///c:/Users/prave/OneDrive/Desktop/DataSprintHackathon/tta_optimization_v2/submission_wbf_optimized.csv) (**43.53**).
