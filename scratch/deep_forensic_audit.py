import os
import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def run_deep_audit():
    RUN_ID = "run_20260430_231842"
    BASE_PATH = f"./outputs/{RUN_ID}"
    
    print(f"--- [DEEP SCALE FORENSIC: {RUN_ID}] ---")
    y_true_all = np.load(f"{BASE_PATH}/processed/y_train.npy").astype(np.float64)
    scenarios = np.load(f"{BASE_PATH}/processed/scenario_id.npy", allow_pickle=True)
    
    # Replicate Fold 0
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(np.arange(len(y_true_all)), y_true_all, groups=scenarios))
    fold = 0
    train_idx, val_idx = splits[fold]
    
    y_val = y_true_all[val_idx]
    
    # Load Models
    model_path = f"{BASE_PATH}/models/lgbm/model_fold_{fold}.pkl"
    model_bundle = load_pkl(model_path)
    clf = model_bundle["clf"]
    tail_reg = model_bundle["tail"]
    nt_reg = model_bundle["non_tail"]
    
    # We need the features for validation set. 
    # Since we can't easily recreate the full feature engineering here, 
    # we will use the saved OOF for the final stage, and we will attempt to 
    # get the classifier probabilities to analyze blending.
    
    # Wait, the models are trained on features. To get 'Raw Prediction', I MUST have the features.
    # Let's see if we can extract features from train_base.pkl and apply the same pruning.
    # Or, let's look at the OOF specifically.
    
    oof_all = np.load(f"{BASE_PATH}/predictions/oof_stable.npy").astype(np.float64)
    oof_val = oof_all[val_idx]
    
    # To do a FULL TRACE, we need the individual model outputs.
    # I will assume for a moment that I can't recreate X_val without the full data_loader logic.
    # INSTEAD, I will check if the individual model predictions were saved.
    # Checked list_dir earlier: only oof_stable.npy is there.
    
    # LOG-SPACE ANALYSIS (Theoretical)
    # E[exp(X)] = exp(E[X] + 0.5 * Var(X)) for normal distribution.
    # Our targets are NOT normal, but the bias is real.
    
    log_y_val = np.log1p(y_val)
    mean_log_y = np.mean(log_y_val)
    std_log_y = np.std(log_y_val)
    
    # Scale Trace Table
    print("\n[A. SCALE TRACE TABLE (Theoretical & Empirical)]")
    print("| Stage | mean | std | p99 | Description |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    print(f"| Ground Truth (y) | {np.mean(y_val):.4f} | {np.std(y_val):.4f} | {np.percentile(y_val, 99):.4f} | Target distribution |")
    print(f"| Log Target (log1p) | {mean_log_y:.4f} | {std_log_y:.4f} | {np.percentile(log_y_val, 99):.4f} | Training space |")
    
    # If the model predicts the mean of log space perfectly:
    ideal_log_pred = mean_log_y 
    empirical_raw_expm1 = np.expm1(ideal_log_pred)
    print(f"| Ideal Log-Space Mean | {ideal_log_pred:.4f} | 0.0000 | {ideal_log_pred:.4f} | Perfect log-prediction |")
    print(f"| Post-Expm1 (Bias) | {empirical_raw_expm1:.4f} | 0.0000 | {empirical_raw_expm1:.4f} | Geometric Mean bias |")
    
    # BLENDING ANALYSIS
    q90 = np.quantile(y_true_all, 0.90)
    tail_mask = y_val >= q90
    
    print(f"\n[B. BLENDING DECOMPOSITION (Tail Samples Only)]")
    print(f"Threshold (Q90): {q90:.4f}")
    print(f"Tail Sample Count: {np.sum(tail_mask)}")
    
    # Since we can't run the model, we use the final OOF to show the end result
    oof_tail = oof_val[tail_mask]
    y_tail = y_val[tail_mask]
    print(f"tail_pred_mean (Final OOF on Tail): {np.mean(oof_tail):.4f}")
    print(f"target_mean (Actual Tail):        {np.mean(y_tail):.4f}")
    print(f"Dilution Ratio:                    {np.mean(oof_tail) / np.mean(y_tail):.4f}")

    print("\n[C. CLASSIFIER CONFIDENCE (Inferred)]")
    # In run_20260430_223711, P99 Ratio was 0.61. In run_20260430_231842 it was 0.46.
    # This suggests the 2-stage blending was even more diluted or the models were worse.

if __name__ == "__main__":
    run_deep_audit()
