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

logger = get_logger("RIGOROUS_AUDIT")

def run_rigorous_audit(run_id, prev_run_id=None):
    proc_dir = f"outputs/{run_id}/processed"
    pred_dir = f"outputs/{run_id}/predictions"
    
    # 1. Load Data
    train_base = pd.read_pickle(f"{proc_dir}/train_base.pkl")
    y_train = load_npy(f"{proc_dir}/y_train.npy")
    oof_preds = load_npy(f"{pred_dir}/oof_stable.npy")
    
    with open(f"{proc_dir}/signal_validation_logs.json", 'r') as f:
        data = json.load(f)
    val_logs = pd.DataFrame(data['val_logs'])
    
    # 2. Category Analysis
    def get_cat(feat):
        if feat.startswith('inter_'): return 'Interaction'
        if any(s in feat for s in ['_slope_', '_rate_', '_diff_']): return 'Trend'
        if any(s in feat for s in ['_std_', '_volatility_']): return 'Volatility'
        from src.schema import BASE_COLS
        if feat in BASE_COLS: return 'Base'
        return 'Other'

    val_logs['category'] = val_logs['feature'].apply(get_cat)
    cat_summary = val_logs.groupby('category').agg(
        total=('feature', 'count'),
        survived=('passed', 'sum'),
        rejected=('passed', lambda x: (x == False).sum())
    )
    cat_summary['survival_rate'] = cat_summary['survived'] / cat_summary['total']
    
    print("\n--- [TASK 1] CATEGORY SURVIVAL ---")
    print(cat_summary.to_string())

    # 3. Distribution Metrics (Absolute)
    target_std = np.std(y_train)
    pred_std = np.std(oof_preds)
    std_ratio_abs = pred_std / (target_std + 1e-9)
    
    target_p99 = np.percentile(y_train, 99)
    pred_p99 = np.percentile(oof_preds, 99)
    p99_ratio_abs = pred_p99 / (target_p99 + 1e-9)
    
    # Fold Stability
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_std_ratios = []
    for tr, val in kf.split(y_train):
        f_std = np.std(oof_preds[val]) / (np.std(y_train[val]) + 1e-9)
        fold_std_ratios.append(f_std)
        
    print("\n--- [TASK 2] STD_RATIO ANALYSIS ---")
    print(f"std_ratio (global)     : {std_ratio_abs:.4f}")
    print(f"fold_std_ratios        : {fold_std_ratios}")
    print(f"std_ratio_variance     : {np.var(fold_std_ratios):.6f}")
    
    print("\n--- [TASK 3] P99_RATIO ANALYSIS ---")
    print(f"p99_ratio (global)     : {p99_ratio_abs:.4f}")
    residuals = np.abs(oof_preds - y_train)
    q99_mae = np.percentile(residuals, 99)
    print(f"Q99 MAE (Tail Error)   : {q99_mae:.4f}")

    # 4. Resurrection Test (Signal Completeness)
    print("\n--- [TASK 1] RESURRECTION TEST ---")
    baseline_features = [f for f in train_base.columns if f in val_logs[val_logs['passed'] == True]['feature'].tolist() or f in val_logs['category'].tolist()]
    # Re-identify from train_base to be sure
    from src.schema import BASE_COLS
    current_features = [f for f in train_base.columns if (f in val_logs[val_logs['passed'] == True]['feature'].tolist() or f in BASE_COLS)]
    
    baseline_mae = mean_absolute_error(y_train, oof_preds)
    
    res_results = []
    for cat in ['Trend', 'Volatility', 'Interaction']:
        rejected_cat = val_logs[(val_logs['category'] == cat) & (val_logs['passed'] == False)].sort_values('perm_delta', ascending=False).head(20)
        if rejected_cat.empty: continue
        
        test_features = current_features + rejected_cat['feature'].tolist()
        # Train & CV
        cv_mae = []
        for tr, val in kf.split(train_base):
            model = LGBMRegressor(**Config.RAW_LGBM_PARAMS)
            model.fit(train_base.iloc[tr][test_features], y_train[tr])
            cv_mae.append(mean_absolute_error(y_train[val], model.predict(train_base.iloc[val][test_features])))
        
        mae_delta = np.mean(cv_mae) - baseline_mae
        res_results.append({'category': cat, 'mae_delta': mae_delta})
        print(f"Resurrecting {cat} (top 20): mae_delta = {mae_delta:.6f}")

    # 5. Interaction Ablation (Strict)
    print("\n--- [TASK 4] INTERACTION ABLATION ---")
    inter_features = [f for f in current_features if f.startswith('inter_')]
    if not inter_features:
        print("No interaction features survived to be ablated.")
    else:
        for f in inter_features:
            test_features = [c for c in current_features if c != f]
            cv_mae = []
            for tr, val in kf.split(train_base):
                model = LGBMRegressor(**Config.RAW_LGBM_PARAMS)
                model.fit(train_base.iloc[tr][test_features], y_train[tr])
                cv_mae.append(mean_absolute_error(y_train[val], model.predict(train_base.iloc[val][test_features])))
            mae_delta = np.mean(cv_mae) - baseline_mae # Positive means feature was HELPFUL
            print(f"Interaction {f}: mae_delta = {mae_delta:.6f} | {'HELPFUL' if mae_delta > 0 else 'REDUNDANT'}")

if __name__ == "__main__":
    run_rigorous_audit("run_20260427_095615")
