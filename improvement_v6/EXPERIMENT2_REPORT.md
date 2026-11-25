# Experiment 2 Report: Geometry-Aware False Alarm Suppression

**Competition**: Anveshan DataSprint 2.0: Intelligent Road Damage Detection  
**Experiment**: Experiment 2 (Geometry-Aware False Alarm Suppression)  
**Directory**: `improvement_v6/`  
**Execution Time**: **39.52 seconds** (0.66 minutes)  
**Baseline Evaluated**: `tta_optimization_v2/val_pred_best_optimized.csv` (Score: **`43.5267`** / **`43.53`**)  
**Submission Requirement**: Generate new test submission ONLY IF complete validation improves by $\ge +0.20$ points (score $\ge 43.73$).

---

## 1. Safety Audit & Integrity Verification

- **Protected Files Audited**: 73 files (across `weights/`, `baseline_work/`, `tta_experiment/`, `tta_optimization_v2/`, `improvement_v3/`, `improvement_v4/`, root CSV submissions, and official scripts).
- **Audit Files**:
  - Pre-experiment hashes: [`safety_inventory_pre.json`](file:///c:/Users/prave/OneDrive/Desktop/DataSprintHackathon/improvement_v6/safety_inventory_pre.json)
  - Post-experiment hashes: [`safety_inventory_post.json`](file:///c:/Users/prave/OneDrive/Desktop/DataSprintHackathon/improvement_v6/safety_inventory_post.json)
- **Result**: **100% Match (73/73 files verified identical; zero modifications, zero renames, zero deletions)**.

---

## 2. Confidence Threshold Clarification: 0.001 vs. 0.237

The user requested an investigation into why the error audit used confidence threshold `0.237` while earlier experiments used `0.001`:

- **In Prediction Generation (`conf = 0.001`)**:
  - `0.001` is the **candidate generation floor**. It acts as a wide filter to ensure all weak detections with potential overlap are retained for non-maximum suppression (NMS) and weighted boxes fusion (WBF), keeping file sizes manageable while preserving maximum recall.
- **In Official Scorer (`score.py`) Evaluation (`conf = 0.237`)**:
  - As defined in `score.py` lines 23–25:
    > *"Precision / Recall: at IoU 0.50, read at ONE confidence threshold shared by all classes: the threshold (grid step 0.001) that maximises the mean F1 over the classes. The scorer finds it; teams do not pick it."*
  - When `score.py` evaluates the 101-point COCO-style average precision curve, it evaluates across all thresholds from $0.0$ to $1.0$.
  - To report the scalar Precision, Recall, and F1 metrics, `score.py` automatically sweeps 1,001 candidate thresholds (`np.linspace(0.0, 1.0, 1001)`) to locate the exact threshold that maximizes mean F1 across the 4 classes.
  - For our baseline predictions, that evaluator-selected optimal threshold was **0.237** (and **0.241** on the holdout split).
  - Therefore, `0.001` is the candidate generation threshold, while `0.237` is the evaluator's internal F1-maximizing operating point.

---

## 3. Aspect Ratio Distribution Analysis (Ground Truth vs. Predictions)

Aspect ratios $R = w / h$ were computed across all validation annotations and baseline predictions:

| Damage Category | GT 5th %ile | GT Median | GT 95th %ile | Pred 5th %ile | Pred Median | Pred 95th %ile | Geometric Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **D00 (Longitudinal)** | 0.29 | 0.72 | 1.77 | 0.28 | 0.83 | 2.26 | Lane perspective ($h > w$ in >85% of damage) |
| **D10 (Transverse)** | 2.70 | 5.11 | 10.11 | 1.09 | 4.27 | 10.99 | Cross-road propagation ($w \gg h$, median 5.1) |
| **D20 (Alligator)** | 0.64 | 1.49 | 3.41 | 0.44 | 1.44 | 5.11 | Moderately isotropic / network cracks |
| **D40 (Pothole)** | 0.85 | 1.63 | 3.32 | 0.66 | 1.61 | 3.57 | Compact circular / oval depressions |

**Observation**: Ground truth D10 transverse cracks have a 5th percentile of 2.70 and median of 5.11. In contrast, predicted D10 detections have a 5th percentile down to 1.09, showing that the model occasionally outputs nearly square or vertical boxes for transverse cracks—many of which are background noise.

---

## 4. Calibration & Holdout Validation Protocol

To strictly prevent data leakage and avoid tuning on the evaluation target:
- The 2,150 validation images were deterministically split 50/50 (`seed=42`):
  - **Calibration Set**: 1,075 images (used exclusively for tuning soft penalty parameters).
  - **Holdout Set**: 1,075 images (untouched during tuning; evaluated single-shot).

### Calibration Set Tuning Results (1,075 Images)
Baseline Calibration Score: **`43.8703`** (mAP@50: 0.5278, F1: 0.5276).

| Candidate Configuration | Calibration Score | Delta | mAP@50 | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (No penalty) | 43.8703 | 0.0000 | 0.5278 | 0.5738 | 0.4930 | 0.5276 |
| D10 only ($R_{\text{min}}=1.0, \lambda=0.2$) | 43.8703 | +0.0001 | 0.5278 | 0.5738 | 0.4930 | 0.5276 |
| D10 only ($R_{\text{min}}=1.2, \lambda=0.3$) | 43.8705 | +0.0002 | 0.5278 | 0.5738 | 0.4930 | 0.5276 |
| D10 only ($R_{\text{min}}=1.5, \lambda=0.2$) | 43.8708 | +0.0005 | 0.5279 | 0.5738 | 0.4930 | 0.5276 |
| D00 only ($R_{\text{max}}=2.5, \lambda=0.2$) | 43.8714 | +0.0011 | 0.5279 | 0.5738 | 0.4930 | 0.5276 |
| D00 only ($R_{\text{max}}=2.0, \lambda=0.1$) | 43.8696 | -0.0006 | 0.5278 | 0.5738 | 0.4930 | 0.5276 |
| **Combined D10($R_{\text{min}}=1.2, \lambda=0.3$) + D00($R_{\text{max}}=2.5, \lambda=0.2$)** | **43.8716** | **+0.0014** | **0.5279** | **0.5738** | **0.4930** | **0.5276** |

**Winning Configuration Selected from Calibration**:
`D10: r_min=1.2, lambda=0.3` + `D00: r_max=2.5, lambda=0.2`.

---

## 5. Untouched Holdout Set Evaluation (Single Shot)

The winning configuration was applied to the untouched 1,075 holdout images and evaluated against the holdout baseline:

| Metric | Holdout Baseline | Holdout Refined | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy Score** | **43.6860** | **43.6871** | **+0.0011 points** |
| **mAP@50** | 0.5218 | 0.5218 | 0.0000 |
| **mAP@50:95** | 0.2246 | 0.2246 | 0.0000 |
| **Precision** | 0.5489 | 0.5489 | 0.0000 |
| **Recall** | 0.5238 | 0.5238 | 0.0000 |
| **F1 Score** | 0.5349 | 0.5349 | 0.0000 |
| **Optimal Conf Cutoff** | 0.2410 | 0.2410 | — |

### Holdout Per-Class True Positives & False Positives (Cutoff 0.241)
- **D00 (Longitudinal)**: Baseline TP = 139, Refined TP = 139 (**0 genuine detections suppressed**); FP = 141, Refined FP = 141.
- **D10 (Transverse)**: Baseline TP = 98, Refined TP = 98 (**0 genuine detections suppressed**); FP = 96, Refined FP = 96.
- **D20 (Alligator)**: Baseline TP = 291, Refined TP = 291; FP = 147, Refined FP = 147.
- **D40 (Pothole)**: Baseline TP = 152, Refined TP = 152; FP = 135, Refined FP = 135.
- **Genuine Detections Suppressed**: **0** (the soft exponential decay strictly preserved all valid ground truth matches).

---

## 6. Complete Validation Set Evaluation (All 2,150 Images)

Evaluating once on the full 2,150 images confirmed the slight positive gain observed on calibration and holdout:
- **Baseline Full Score**: `43.5267`
- **Refined Full Score**: `43.5281`
- **Net Delta**: **`+0.0014` points**

---

## 7. Submission Decision

- **Baseline Score to Beat**: `43.53`
- **Refined Score**: `43.5281`
- **Net Delta**: **`+0.0014` points**
- **Decision**: In strict accordance with the competition instructions (*"Generate a separate test submission only if the complete validation score improves by at least 0.20 over 43.53"*), **no test submission was generated**.
- The current top submission remains [`tta_optimization_v2/submission_wbf_optimized.csv`](file:///c:/Users/prave/OneDrive/Desktop/DataSprintHackathon/tta_optimization_v2/submission_wbf_optimized.csv) (**Validation Score: 43.53**).
