#!/usr/bin/env python3
"""
Compute and record SHA256 hashes and file sizes of all protected files.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROTECTED_PATHS = [
    # Model weights
    ROOT / "weights" / "extended_best.pt",
    ROOT / "weights" / "best.pt",
    
    # Official & baseline scripts
    ROOT / "inference.py",
    ROOT / "score.py",
    ROOT / "benchmark_speed.py",
    ROOT / "check_submission.py",
    ROOT / "make_kaggle_submission.py",
    ROOT / "train_baseline.py",
    
    # Existing experiment scripts
    ROOT / "benchmark_extended.py",
    ROOT / "evaluate_extended.py",
    ROOT / "evaluate_compare.py",
    ROOT / "make_kaggle_submission_extended.py",
    ROOT / "make_submission_iou05.py",
    ROOT / "classwise_optimize.py",
    ROOT / "optimize_predictions.py",
    ROOT / "train_extended.py",
    ROOT / "train_exp1_finetune.py",
    ROOT / "train_exp3_v8s_pilot.py",

    # Existing submission CSV files
    ROOT / "submission.csv",
    ROOT / "submission_extended.csv",
    ROOT / "submission_iou05.csv",
    ROOT / "speed_baseline_submission.csv",
    ROOT / "speed_extended_submission.csv",
    ROOT / "tta_experiment" / "submission_tta.csv",
    ROOT / "tta_optimization_v2" / "submission_wbf_optimized.csv",

    # Key existing reports
    ROOT / "tta_experiment" / "TTA_EXPERIMENT_REPORT.md",
    ROOT / "tta_experiment" / "score_tta.json",
    ROOT / "tta_experiment" / "tta_comparison_report.json",
    ROOT / "tta_optimization_v2" / "EXPERIMENT_REPORT_V2.md",
    ROOT / "tta_optimization_v2" / "comprehensive_comparison.json",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_inventory():
    inventory = {}
    missing = []
    for p in PROTECTED_PATHS:
        if p.exists():
            inventory[str(p.relative_to(ROOT))] = {
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
        else:
            missing.append(str(p.relative_to(ROOT)))
    return inventory, missing


if __name__ == "__main__":
    inv, miss = compute_inventory()
    out_file = ROOT / "improvement_v3" / "safety_inventory_pre.json"
    with open(out_file, "w") as f:
        json.dump(inv, f, indent=2)
    print(f"Safety Inventory Recorded: {len(inv)} protected files hashed.")
    if miss:
        print(f"Warning: {len(miss)} files listed were not found: {miss}")
    else:
        print("All specified protected files verified and hashed successfully.")
