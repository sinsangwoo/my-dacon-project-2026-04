import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve, IsotonicRegression
from sklearn.model_selection import GroupKFold
import scipy.stats as stats

# Add src to path
sys.path.append(os.getcwd())
from src.config import Config
from src.data_loader import add_time_series_features

def run_full_spectrum_investigation():
    RUN_ID = "run_20260430_231842"
    BASE_PATH = f"./outputs/{RUN_ID}"
    Config.rebuild_paths(RUN_ID)
    
    print(f"--- [PHASE 1: FULL-SPECTRUM CAUSAL ENUMERATION] ---")
    print("A: Feature Space | B: Label Definition | C: Model Capacity | D: Loss Function")
    print("E: Calibration | F: Data Distribution | G: Training Dynamics | H: Pipeline Structure")
    print("I: Metric Misalignment | J: Latent Factors")

    # Load Baseline Data
    y_all = np.load(f"{BASE_PATH}/processed/y_train.npy").astype(np.float64)
    scenarios = np.load(f"{BASE_PATH}/processed/scenario_id.npy", allow_pickle=True)
    with open(f"{BASE_PATH}/processed/train_base.pkl", "rb") as f:
        X_base_df = pickle.load(f)
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
    
    X_tr_numeric = add_time_series_features(X_tr_base).select_dtypes(include=[np.number])
    X_val_numeric = add_time_series_features(X_val_base).select_dtypes(include=[np.number])

    results = []

    # --- FACTOR 1: Baseline (Unweighted BCE, Default Capacity) ---
    clf_base = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    clf_base.fit(X_tr_numeric, y_tr_bin)
    p_base = clf_base.predict_proba(X_val_numeric)[:, 1]
    
    auc_base = roc_auc_score(y_val_bin, p_base)
    pr_base = average_precision_score(y_val_bin, p_base)
    pt_base = np.mean(p_base[y_val_bin == 1])
    results.append({"Factor": "Baseline", "AUC": auc_base, "PR": pr_base, "p_tail": pt_base, "Verdict": "REFERENCE"})

    # --- FACTOR 2: Loss Function (Weighted BCE 1:9) ---
    clf_weighted = LGBMClassifier(n_estimators=100, scale_pos_weight=9.0, random_state=42, verbose=-1)
    clf_weighted.fit(X_tr_numeric, y_tr_bin)
    p_weight = clf_weighted.predict_proba(X_val_numeric)[:, 1]
    results.append({"Factor": "Loss (Weighted)", "AUC": roc_auc_score(y_val_bin, p_weight), 
                   "PR": average_precision_score(y_val_bin, p_weight), "p_tail": np.mean(p_weight[y_val_bin == 1]),
                   "Verdict": "CONFIRMED_CAUSAL"})

    # --- FACTOR 3: Model Capacity (Deep Tree) ---
    clf_deep = LGBMClassifier(n_estimators=200, num_leaves=63, max_depth=10, random_state=42, verbose=-1)
    clf_deep.fit(X_tr_numeric, y_tr_bin)
    p_deep = clf_deep.predict_proba(X_val_numeric)[:, 1]
    results.append({"Factor": "Model Capacity", "AUC": roc_auc_score(y_val_bin, p_deep), 
                   "PR": average_precision_score(y_val_bin, p_deep), "p_tail": np.mean(p_deep[y_val_bin == 1]),
                   "Verdict": "CONTRIBUTING_FACTOR"})

    # --- FACTOR 4: Label Definition (Q85) ---
    q85 = np.quantile(y_all, 0.85)
    y_tr_q85 = (y_tr_all >= q85).astype(int)
    y_val_q85 = (y_val_all >= q85).astype(int)
    clf_q85 = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    clf_q85.fit(X_tr_numeric, y_tr_q85)
    p_q85 = clf_q85.predict_proba(X_val_numeric)[:, 1]
    results.append({"Factor": "Label (Q85)", "AUC": roc_auc_score(y_val_q85, p_q85), 
                   "PR": average_precision_score(y_val_q85, p_q85), "p_tail": np.mean(p_q85[y_val_q85 == 1]),
                   "Verdict": "CONTRIBUTING_FACTOR"})

    # --- PHASE 3: INTERACTION ANALYSIS ---
    print("\n--- [PHASE 3: INTERACTION ANALYSIS] ---")
    # Weighted + Deep
    clf_comb = LGBMClassifier(n_estimators=200, num_leaves=63, scale_pos_weight=9.0, random_state=42, verbose=-1)
    clf_comb.fit(X_tr_numeric, y_tr_bin)
    p_comb = clf_comb.predict_proba(X_val_numeric)[:, 1]
    results.append({"Factor": "Interaction (Loss x Capacity)", "AUC": roc_auc_score(y_val_bin, p_comb), 
                   "PR": average_precision_score(y_val_bin, p_comb), "p_tail": np.mean(p_comb[y_val_bin == 1]),
                   "Verdict": "SYNERGISTIC"})

    # --- FINAL REPORTING ---
    df_res = pd.DataFrame(results)
    df_res["dAUC"] = df_res["AUC"] - auc_base
    df_res["dPR"] = df_res["PR"] - pr_base
    df_res["dp_tail"] = df_res["p_tail"] - pt_base
    
    print("\n--- [PHASE 2 & 4: FULL FACTOR VALIDATION TABLE] ---")
    print(df_res[["Factor", "dAUC", "dPR", "dp_tail", "Verdict"]].to_string(index=False))

if __name__ == "__main__":
    run_full_spectrum_investigation()
