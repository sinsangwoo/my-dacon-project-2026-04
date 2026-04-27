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

# Setup minimal logging
logger = get_logger("UNDERFITTING_TEST")

def run_stepwise_test(run_id):
    # 1. Load Data & Manifests
    proc_dir = f"outputs/{run_id}/processed"
    log_path = f"{proc_dir}/signal_validation_logs.json"
    
    with open(log_path, 'r') as f:
        data = json.load(f)
    val_logs = data['val_logs']
    df_val = pd.DataFrame(val_logs)
    
    train_base = pd.read_pickle(f"{proc_dir}/train_base.pkl")
    y_train = load_npy(f"{proc_dir}/y_train.npy")
    
    # 2. Identify Baseline (Survivors + Base Cols)
    # BASE_COLS are the 30 sensors
    from src.schema import BASE_COLS
    baseline_features = [f for f in train_base.columns if f in BASE_COLS]
    survivors = df_val[df_val['passed'] == True]['feature'].tolist()
    baseline_features = list(set(baseline_features) | set(survivors))
    
    # 3. Identify Expansion Candidates (Rejected, ranked by perm_delta)
    rejected = df_val[df_val['passed'] == False].sort_values('perm_delta', ascending=False)
    expansion_candidates = rejected['feature'].tolist()
    
    results = []
    
    # Step sizes: baseline, 60, 100, 150
    steps = [len(baseline_features), 60, 100, 150]
    
    for step in steps:
        if step < len(baseline_features): continue
        
        # Take unique features while preserving order from baseline then candidates
        raw_combined = baseline_features + expansion_candidates[:(step - len(baseline_features))]
        current_features = []
        seen = set()
        for f in raw_combined:
            if f in train_base.columns and f not in seen:
                current_features.append(f)
                seen.add(f)
        
        logger.info(f"Testing step: {len(current_features)} features...")
        
        # 3-Fold CV
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(train_base))
        
        for tr_idx, val_idx in kf.split(train_base):
            X_tr, X_val = train_base.iloc[tr_idx][current_features], train_base.iloc[val_idx][current_features]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            
            model = LGBMRegressor(**Config.RAW_LGBM_PARAMS)
            model.fit(X_tr, y_tr)
            oof_preds[val_idx] = model.predict(X_val)
            
        mae = mean_absolute_error(y_train, oof_preds)
        std_ratio = np.std(oof_preds) / (np.std(y_train) + 1e-9)
        
        # Tail Error (Q99)
        residuals = np.abs(oof_preds - y_train)
        q99_mae = np.percentile(residuals, 99)
        
        results.append({
            'count': len(current_features),
            'mae': float(mae),
            'std_ratio': float(std_ratio),
            'q99_mae': float(q99_mae)
        })
        
        logger.info(f"  Result: MAE={mae:.4f}, StdRatio={std_ratio:.4f}, Q99_MAE={q99_mae:.4f}")

    # Output results
    output_path = f"outputs/{run_id}/underfitting_curve.json"
    save_json(results, output_path)
    
    print("\n[PERFORMANCE_CURVE_SUMMARY]")
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    run_stepwise_test("run_20260427_092836")
