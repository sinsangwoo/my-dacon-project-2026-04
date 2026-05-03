import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold

# Add src to path
sys.path.append(os.getcwd())
from src.config import Config
from src.data_loader import add_time_series_features
from src.schema import FEATURE_SCHEMA

def run_causal_validation_sanitized():
    RUN_ID = "run_20260430_231842"
    BASE_PATH = f"./outputs/{RUN_ID}"
    Config.rebuild_paths(RUN_ID)
    
    print("Loading data for SANITIZED causal validation...")
    y_all = np.load(f"{BASE_PATH}/processed/y_train.npy").astype(np.float64)
    scenarios = np.load(f"{BASE_PATH}/processed/scenario_id.npy", allow_pickle=True)
    
    with open(f"{BASE_PATH}/processed/train_base.pkl", "rb") as f:
        X_base_df = pickle.load(f)
    
    # DROP TARGET TO PREVENT LEAKAGE
    if Config.TARGET in X_base_df.columns:
        X_base_df = X_base_df.drop(columns=[Config.TARGET])
    
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(np.arange(len(y_all)), y_all, groups=scenarios))
    tr_idx, val_idx = splits[0]
    
    X_tr_base, X_val_base = X_base_df.iloc[tr_idx].copy(), X_base_df.iloc[val_idx].copy()
    y_tr_all, y_val_all = y_all[tr_idx], y_all[val_idx]
    
    q90_global = np.quantile(y_all, 0.90)
    y_tr_bin = (y_tr_all >= q90_global).astype(int)
    y_val_bin = (y_val_all >= q90_global).astype(int)
    
    print("Generating full feature set (761) - Sanitized...")
    X_tr_fe = add_time_series_features(X_tr_base)
    X_val_fe = add_time_series_features(X_val_base)
    
    X_tr_numeric = X_tr_fe.select_dtypes(include=[np.number])
    X_val_numeric = X_val_fe.select_dtypes(include=[np.number])
    
    # Identify Pruned Features
    with open(f"{BASE_PATH}/processed/pruning_manifest.json", "r") as f:
        pruning = json.load(f)
    dropped_cols = pruning.get("cols_to_drop_nan", []) + pruning.get("cols_to_drop_corr", [])
    dropped_cols = [c for c in dropped_cols if c in X_tr_numeric.columns]
    
    # STAGE: BEFORE_PRUNING
    print("Training Classifier: BEFORE_PRUNING (Sanitized)...")
    clf_before = LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
    clf_before.fit(X_tr_numeric, y_tr_bin)
    p_before = clf_before.predict_proba(X_val_numeric)[:, 1]
    
    auc_before = roc_auc_score(y_val_bin, p_before)
    pr_before = average_precision_score(y_val_bin, p_before)
    p_tail_before = np.mean(p_before[y_val_bin == 1])
    
    # STAGE: AFTER_PRUNING
    print("Training Classifier: AFTER_PRUNING (Sanitized)...")
    X_tr_pruned = X_tr_numeric.drop(columns=dropped_cols)
    X_val_pruned = X_val_numeric.drop(columns=dropped_cols)
    
    clf_after = LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
    clf_after.fit(X_tr_pruned, y_tr_bin)
    p_after = clf_after.predict_proba(X_val_pruned)[:, 1]
    
    auc_after = roc_auc_score(y_val_bin, p_after)
    pr_after = average_precision_score(y_val_bin, p_after)
    p_tail_after = np.mean(p_after[y_val_bin == 1])
    
    print("\n--- [1. BEFORE vs AFTER PRUNING (SANITIZED)] ---")
    print(f"| Stage | AUC | PR AUC | p_tail |")
    print(f"| :--- | :--- | :--- | :--- |")
    print(f"| BEFORE_PRUNING | {auc_before:.4f} | {pr_before:.4f} | {p_tail_before:.4f} |")
    print(f"| AFTER_PRUNING  | {auc_after:.4f} | {pr_after:.4f} | {p_tail_after:.4f} |")
    
    print("\n--- [FINAL CAUSAL VERDICT (RE-EVALUATED)] ---")
    delta_p = p_tail_after - p_tail_before
    if delta_p < -0.05:
        print("[CONFIRMED_OVER_PRUNING]")
    elif p_tail_before < 0.45:
        print("[INCONCLUSIVE - INTRINSIC_LIMITATION]")
        print(f"Reason: p_tail was already low ({p_tail_before:.4f}) BEFORE pruning.")
    else:
        print("[REJECTED_OVER_PRUNING]")

if __name__ == "__main__":
    run_causal_validation_sanitized()
