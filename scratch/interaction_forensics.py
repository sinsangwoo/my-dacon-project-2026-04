import os
import pandas as pd
import numpy as np
import json
import logging
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from src.config import Config
from src.utils import get_logger, save_json, load_npy

logger = get_logger("INTERACTION_FORENSICS")

def run_forensics(run_id):
    proc_dir = f"outputs/{run_id}/processed"
    
    # 1. Load Data
    train_base = pd.read_pickle(f"{proc_dir}/train_base.pkl")
    y_train = load_npy(f"{proc_dir}/y_train.npy")
    
    with open(f"{proc_dir}/signal_validation_logs.json", 'r') as f:
        data = json.load(f)
    df_val = pd.DataFrame(data['val_logs'])
    
    # Identify interactions
    interactions = [f for f in train_base.columns if f.startswith('inter_')]
    df_inter = df_val[df_val['feature'].isin(interactions)]
    
    print(f"\n--- [TASK 1] INTERACTION ROOT CAUSE ANALYSIS ---")
    print(f"Total Interactions Created: {len(interactions)}")
    print(f"Interactions in Validation Logs: {len(df_inter)}")
    
    if len(df_inter) > 0:
        print("\nRejection Audit (Top 5):")
        print(df_inter.sort_values('perm_delta', ascending=False).head(5)[['feature', 'perm_delta', 'gain', 'rejection_reasons']])

    # Test H2: Interaction-only Model
    if len(interactions) > 0:
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        oof_inter = np.zeros(len(train_base))
        for tr, val in kf.split(train_base):
            model = LGBMRegressor(**Config.RAW_LGBM_PARAMS)
            model.fit(train_base.iloc[tr][interactions], y_train[tr])
            oof_inter[val] = model.predict(train_base.iloc[val][interactions])
        
        inter_mae = mean_absolute_error(y_train, oof_inter)
        print(f"\nInteraction-only Model MAE: {inter_mae:.4f}")
        # Compare with dummy baseline (mean)
        dummy_mae = mean_absolute_error(y_train, np.full_like(y_train, np.mean(y_train)))
        print(f"Dummy Baseline MAE: {dummy_mae:.4f}")
        
        if inter_mae < dummy_mae:
            print("Conclusion: Interactions DO have predictive power (H1 False).")
        else:
            print("Conclusion: Interactions are noise-level or generated incorrectly (H1/H4 True).")

    # Test H2/H5: Forced Inclusion vs Base
    from src.schema import BASE_COLS
    base_features = [f for f in train_base.columns if f in BASE_COLS]
    
    def get_mae(cols):
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        oof = np.zeros(len(train_base))
        for tr, val in kf.split(train_base):
            m = LGBMRegressor(**Config.RAW_LGBM_PARAMS)
            m.fit(train_base.iloc[tr][cols], y_train[tr])
            oof[val] = m.predict(train_base.iloc[val][cols])
        return mean_absolute_error(y_train, oof)

    base_mae = get_mae(base_features)
    print(f"\nBase-only MAE: {base_mae:.4f}")
    
    top_3_inter = df_inter.sort_values('perm_delta', ascending=False).head(3)['feature'].tolist()
    if top_3_inter:
        forced_mae = get_mae(base_features + top_3_inter)
        print(f"Base + Top 3 Interactions MAE: {forced_mae:.4f}")
        print(f"Delta: {base_mae - forced_mae:.6f}")
        if forced_mae < base_mae:
            print("Conclusion: Interactions add value over Base (H5 False). Gate is likely too strict (H2 True).")
        else:
            print("Conclusion: Interactions are redundant with Base (H5 True).")

    # [TASK 2] COVERAGE 59% VALIDATION (Resurrection)
    print(f"\n--- [TASK 2] COVERAGE 59% VALIDATION ---")
    rejected = df_val[df_val['passed'] == False].sort_values('perm_delta', ascending=False)
    
    # Re-identify survivors
    survivors = df_val[df_val['passed'] == True]['feature'].tolist()
    final_set = list(set(base_features) | set(survivors))
    baseline_mae = get_mae(final_set)
    print(f"Baseline (59% Coverage) MAE: {baseline_mae:.4f}")
    
    # Test top 20 rejected
    top_20_rejected = rejected.head(20)['feature'].tolist()
    resurrected_mae = get_mae(final_set + top_20_rejected)
    print(f"Resurrected (Top 20 Rejected) MAE: {resurrected_mae:.4f}")
    print(f"Delta: {baseline_mae - resurrected_mae:.6f}")
    
    # Classification
    # Truly Noise: Negative perm_delta
    # Weak Signal: Positive perm_delta < 0.001
    # Misclassified: Positive perm_delta > 0.001 but rejected
    
    print("\n--- REJECTED CLASSIFICATION ---")
    truly_noise = len(rejected[rejected['perm_delta'] <= 0])
    weak_signal = len(rejected[(rejected['perm_delta'] > 0) & (rejected['perm_delta'] <= 0.001)])
    misclassified = len(rejected[rejected['perm_delta'] > 0.001])
    
    print(f"Truly Noise (perm <= 0): {truly_noise} ({truly_noise/len(rejected):.1%})")
    print(f"Weak Signal (0 < perm <= 0.001): {weak_signal} ({weak_signal/len(rejected):.1%})")
    print(f"Misclassified Signal (perm > 0.001): {misclassified} ({misclassified/len(rejected):.1%})")

if __name__ == "__main__":
    run_forensics("run_20260427_102343")
