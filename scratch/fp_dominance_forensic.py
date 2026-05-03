import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"
predictions_dir = f"{base_dir}/outputs/{run_id}/predictions"

def analyze_fp_dominance():
    print(f"--- [PHASE 2: FP DOMINANCE FORENSIC] ---")
    
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    oof_raw = np.load(f"{processed_dir}/oof_raw.npy", allow_pickle=True)
    oof_stable = np.load(f"{predictions_dir}/oof_stable.npy", allow_pickle=True)
    
    q90 = np.percentile(y_true, 90)
    is_tail = y_true >= q90
    
    # Identify samples where Stable predicted MUCH higher than Raw
    # This is a proxy for where the classifier gave high p
    diff = oof_stable - oof_raw
    threshold_diff = np.percentile(diff, 90) # Top 10% boosted samples
    is_boosted = diff > threshold_diff
    
    # Confusion Matrix (Proxy)
    # Predicted Tail = Boosted, Actual Tail = is_tail
    tp = np.sum(is_boosted & is_tail)
    fp = np.sum(is_boosted & (~is_tail))
    fn = np.sum((~is_boosted) & is_tail)
    tn = np.sum((~is_boosted) & (~is_tail))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"Tail Threshold (Q90): {q90:.4f}")
    print(f"Proxy Precision: {precision:.4f}")
    print(f"Proxy Recall:    {recall:.4f}")
    
    # Error Analysis
    error_stable = np.abs(y_true - oof_stable)
    error_raw = np.abs(y_true - oof_raw)
    
    mae_fp = np.mean(error_stable[is_boosted & (~is_tail)])
    mae_fp_raw = np.mean(error_raw[is_boosted & (~is_tail)])
    
    print(f"\n--- [ERROR IMPACT] ---")
    print(f"MAE on FP (Stable): {mae_fp:.4f}")
    print(f"MAE on FP (Raw):    {mae_fp_raw:.4f}")
    print(f"FP Error Increase:  {mae_fp - mae_fp_raw:+.4f}")
    
    total_increase = np.sum(error_stable[is_boosted & (~is_tail)]) - np.sum(error_raw[is_boosted & (~is_tail)])
    print(f"Total MAE contribution from FP boost: {total_increase / len(y_true):+.4f}")

if __name__ == "__main__":
    analyze_fp_dominance()
