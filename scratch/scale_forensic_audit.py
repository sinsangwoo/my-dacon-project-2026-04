import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def run_audit():
    RUN_ID = "run_20260430_231842"
    BASE_PATH = f"./outputs/{RUN_ID}"
    
    y_true = np.load(f"{BASE_PATH}/processed/y_train.npy").astype(np.float64)
    oof = np.load(f"{BASE_PATH}/predictions/oof_stable.npy").astype(np.float64)
    
    def get_stats(arr):
        return {
            "mean": np.mean(arr),
            "std": np.std(arr),
            "p99": np.percentile(arr, 99)
        }

    stats_true = get_stats(y_true)
    stats_oof = get_stats(oof)
    
    # Analyze the Bias in Log-Space Training
    log_y_true = np.log1p(y_true)
    
    # E[y] vs expm1(E[log1p(y)])
    theoretical_mean = np.expm1(np.mean(log_y_true))
    transform_bias = theoretical_mean / stats_true["mean"]
    
    # Blending Dilution Analysis on a subset
    fold = 0
    model_path = f"{BASE_PATH}/models/lgbm/model_fold_{fold}.pkl"
    # model_bundle = load_pkl(model_path) # Don't need to load if we use OOF
    
    scenarios = np.load(f"{BASE_PATH}/processed/scenario_id.npy", allow_pickle=True)
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(np.arange(len(y_true)), y_true, groups=scenarios))
    val_idx = splits[fold][1]
    
    y_val = y_true[val_idx]
    oof_val = oof[val_idx]
    
    q90 = np.quantile(y_true, 0.90)
    tail_mask = y_val >= q90
    
    # Statistics on Tail Samples
    y_tail = y_val[tail_mask]
    oof_tail = oof_val[tail_mask]
    
    dilution_ratio = np.mean(oof_tail) / np.mean(y_tail)

    print("\n--- [MISSION: SCALE COLLAPSE FORENSIC REPORT] ---")
    print(f"| Stage | mean | std | p99 | delta_mean |")
    print(f"|-------|------|-----|-----|------------|")
    print(f"| Ground Truth | {stats_true['mean']:.4f} | {stats_true['std']:.4f} | {stats_true['p99']:.4f} | 0.0000 |")
    print(f"| Final OOF    | {stats_oof['mean']:.4f} | {stats_oof['std']:.4f} | {stats_oof['p99']:.4f} | {stats_oof['mean'] - stats_true['mean']:.4f} |")
    
    print("\n[PHASE 2: BLENDING DECOMPOSITION (Tail Only)]")
    print(f"Avg(y_true_tail):    {np.mean(y_tail):.4f}")
    print(f"Avg(oof_tail_preds): {np.mean(oof_tail):.4f}")
    print(f"Dilution Ratio:      {dilution_ratio:.4f}")
    
    print("\n[PHASE 4: TRANSFORM BIAS CHECK]")
    print(f"Mean(y_true):            {stats_true['mean']:.4f}")
    print(f"expm1(Mean(log1p_y)):    {theoretical_mean:.4f}")
    print(f"Theoretical Log-Bias:    {transform_bias:.4f}")
    
    print("\n[ROOT CAUSE VERDICT]")
    if transform_bias < 0.6: # Serious transform bias
        print("Verdict: [TRANSFORM_BIAS]")
    elif dilution_ratio < 0.55: # Heavy dilution
        print("Verdict: [BLENDING_DILUTION]")
    elif stats_oof['std'] / stats_true['std'] < 0.6:
        print("Verdict: [MODEL_UNDERFIT] (Variance Collapse)")
    else:
        print("Verdict: [CLASSIFIER_WEAKNESS]")

if __name__ == "__main__":
    run_audit()
