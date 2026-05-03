import numpy as np
import pandas as pd
import os
import joblib
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"
models_dir = f"{base_dir}/outputs/{run_id}/models/lgbm"

def run_p_forensic():
    print(f"--- [PHASE 2: DEEP PROBABILITY FORENSIC] ---")
    
    # 1. Load Labels & Scenario IDs
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    scenario_id = np.load(f"{processed_dir}/scenario_id.npy", allow_pickle=True)
    q90 = np.percentile(y_true, 90)
    y_binary = (y_true >= q90).astype(int)
    
    # 2. Reconstruct Folds
    # In the pipeline, 5-fold scenario-based split is used.
    # We need to reproduce the exact splits to get OOF p.
    unique_scenarios = np.unique(scenario_id)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Load Feature Data (Subset for speed if necessary, but we need full OOF)
    print("Loading train_base.pkl...")
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    
    oof_p = np.zeros(len(y_true))
    
    for fold, (train_scen_idx, val_scen_idx) in enumerate(kf.split(unique_scenarios)):
        print(f"Inference Fold {fold}...")
        val_scenarios = unique_scenarios[val_scen_idx]
        val_mask = np.isin(scenario_id, val_scenarios)
        
        X_val = train_df[val_mask]
        
        # Load Model
        model_dict = joblib.load(f"{models_dir}/model_fold_{fold}.pkl")
        clf = model_dict["clf"]
        
        # Ensure features match - handle cases where model expects Column_n
        feat_names = clf.feature_name_
        if "Column_0" in feat_names:
            # Model expects generic column names, probably fitted on numpy array
            # We must exclude ID or any string columns.
            X_val_input = X_val.select_dtypes(include=[np.number]).values
        else:
            X_val_input = X_val[feat_names].values
        
        oof_p[val_mask] = clf.predict_proba(X_val_input)[:, 1]
        
    # --- ANALYSIS ---
    
    # (A) Separation Power
    roc_auc = roc_auc_score(y_binary, oof_p)
    precision, recall, _ = precision_recall_curve(y_binary, oof_p)
    pr_auc = auc(recall, precision)
    
    print("\n[A] SEPARATION POWER")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC:  {pr_auc:.4f}")
    
    # (B) Calibration
    prob_true, prob_pred = calibration_curve(y_binary, oof_p, n_bins=10)
    brier = brier_score_loss(y_binary, oof_p)
    print(f"\n[B] CALIBRATION")
    print(f"Brier Score: {brier:.4f}")
    print(f"Prob True (bins): {prob_true}")
    print(f"Prob Pred (bins): {prob_pred}")
    
    # (C) p Distribution Structure
    p_tail = oof_p[y_binary == 1]
    p_non_tail = oof_p[y_binary == 0]
    
    print(f"\n[C] P DISTRIBUTION STRUCTURE")
    print(f"E[p | tail]:     {np.mean(p_tail):.4f}")
    print(f"E[p | non-tail]: {np.mean(p_non_tail):.4f}")
    print(f"Overlap (KS):    {np.abs(np.mean(p_tail) - np.mean(p_non_tail)):.4f} (Proxy for distance)")
    
    # (D) Stability
    # (Already handled by OOF reconstruction)

if __name__ == "__main__":
    run_p_forensic()
