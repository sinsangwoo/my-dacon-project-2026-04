import os
import sys
import numpy as np
import pandas as pd
import pickle
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

# Add src to path
sys.path.append(os.getcwd())
from src.config import Config
from src.data_loader import add_time_series_features

def run_mae_impact_validation():
    RUN_ID = "run_20260430_231842"
    BASE_PATH = f"./outputs/{RUN_ID}"
    Config.rebuild_paths(RUN_ID)
    
    print("Loading data for MAE impact validation...")
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
    y_tr, y_val = y_all[tr_idx], y_all[val_idx]
    
    X_tr = add_time_series_features(X_tr_base).select_dtypes(include=[np.number])
    X_val = add_time_series_features(X_val_base).select_dtypes(include=[np.number])
    
    # 1. Train Regressors (Fixed)
    q90 = np.quantile(y_tr, 0.90)
    tail_mask_tr = y_tr >= q90
    
    print("Training Regressors...")
    reg_tail = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    reg_tail.fit(X_tr[tail_mask_tr], np.log1p(y_tr[tail_mask_tr]))
    
    reg_nt = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    reg_nt.fit(X_tr[~tail_mask_tr], np.log1p(y_tr[~tail_mask_tr]))
    
    preds_t = np.expm1(reg_tail.predict(X_val))
    preds_nt = np.expm1(reg_nt.predict(X_val))
    
    # 2. Compare Classifiers
    scenarios_test = [
        ("Baseline (Unweighted)", 1.0),
        ("Weighted (1:9)", 9.0)
    ]
    
    y_val_bin = (y_val >= q90).astype(int)
    y_tr_bin = (y_tr >= q90).astype(int)
    
    val_results = []
    for label, weight in scenarios_test:
        print(f"Testing Scenario: {label}...")
        clf = LGBMClassifier(n_estimators=100, scale_pos_weight=weight, random_state=42, verbose=-1)
        clf.fit(X_tr, y_tr_bin)
        p = clf.predict_proba(X_val)[:, 1]
        
        # Blending
        final_preds = p * preds_t + (1 - p) * preds_nt
        
        mae_global = mean_absolute_error(y_val, final_preds)
        mae_tail = mean_absolute_error(y_val[y_val_bin == 1], final_preds[y_val_bin == 1])
        mae_nt = mean_absolute_error(y_val[y_val_bin == 0], final_preds[y_val_bin == 0])
        p_tail_avg = np.mean(p[y_val_bin == 1])
        p_nt_avg = np.mean(p[y_val_bin == 0])
        
        val_results.append({
            "Scenario": label,
            "Global MAE": mae_global,
            "Tail MAE": mae_tail,
            "Non-Tail MAE": mae_nt,
            "E[p|tail]": p_tail_avg,
            "E[p|nt]": p_nt_avg
        })
        
    df_res = pd.DataFrame(val_results)
    print("\n--- [MAE IMPACT VALIDATION REPORT] ---")
    print(df_res.to_string(index=False))
    
    # Conclusion
    d_mae = df_res.iloc[1]['Global MAE'] - df_res.iloc[0]['Global MAE']
    if d_mae < 0:
        print(f"\nVerdict: [CONFIRMED] p-increase REDUCES MAE by {abs(d_mae):.4f}")
    else:
        print(f"\nVerdict: [REJECTED] p-increase INCREASES MAE by {d_mae:.4f} (False Positive Dominant)")

if __name__ == "__main__":
    run_mae_impact_validation()
