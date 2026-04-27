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

logger = get_logger("OPTIMALITY_AUDIT")

def run_optimality_audit(run_id):
    proc_dir = f"outputs/{run_id}/processed"
    
    # 1. Load Data
    train_base = pd.read_pickle(f"{proc_dir}/train_base.pkl")
    y_train = load_npy(f"{proc_dir}/y_train.npy")
    
    with open(f"{proc_dir}/signal_validation_logs.json", 'r') as f:
        data = json.load(f)
    df_val = pd.DataFrame(data['val_logs'])
    
    # 2. Categorize
    def get_cat(feat):
        if feat.startswith('inter_'): return 'Interaction'
        if any(s in feat for s in ['_slope_', '_rate_', '_diff_']): return 'Trend'
        if any(s in feat for s in ['_std_', '_volatility_']): return 'Volatility'
        from src.schema import BASE_COLS
        if feat in BASE_COLS: return 'Base'
        return 'Other'
    
    df_val['category'] = df_val['feature'].apply(get_cat)
    
    # 3. TASK 1: TREND/VOL OPTIMALITY CURVE
    trend_vol = df_val[df_val['category'].isin(['Trend', 'Volatility'])].sort_values('perm_delta', ascending=False)
    
    from src.schema import BASE_COLS
    base_features = [f for f in train_base.columns if f in BASE_COLS]
    
    results = []
    steps = [5, 10, 20, 40, 80, len(trend_vol)]
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    print("\n--- [TASK 1] TREND/VOL OPTIMALITY CURVE ---")
    for step in steps:
        if step > len(trend_vol): step = len(trend_vol)
        
        candidates = base_features + trend_vol['feature'].tolist()[:step]
        candidates = [f for f in candidates if f in train_base.columns]
        
        oof = np.zeros(len(train_base))
        for tr, val in kf.split(train_base):
            model = LGBMRegressor(**Config.RAW_LGBM_PARAMS)
            model.fit(train_base.iloc[tr][candidates], y_train[tr])
            oof[val] = model.predict(train_base.iloc[val][candidates])
            
        mae = mean_absolute_error(y_train, oof)
        std_ratio = np.std(oof) / (np.std(y_train) + 1e-9)
        p99_ratio = np.percentile(oof, 99) / (np.percentile(y_train, 99) + 1e-9)
        
        results.append({
            'n_trend_vol': step,
            'mae': mae,
            'std_ratio': std_ratio,
            'p99_ratio': p99_ratio
        })
        print(f"Features: {step} | MAE: {mae:.4f} | StdRatio: {std_ratio:.4f} | P99Ratio: {p99_ratio:.4f}")

    # 4. TASK 2: SIGNAL COVERAGE QUANTIFICATION
    true_signals = df_val[df_val['perm_delta'] > 0.001]
    survived_signals = true_signals[true_signals['passed'] == True]
    
    print("\n--- [TASK 2] SIGNAL COVERAGE ---")
    print(f"Total 'True' Signals (perm > 0.001): {len(true_signals)}")
    print(f"Recovered Signals: {len(survived_signals)}")
    print(f"Recovery Rate: {len(survived_signals)/len(true_signals):.2%}")
    
    cat_coverage = df_val.groupby('category').agg(
        total=('feature', 'count'),
        survived=('passed', 'sum')
    )
    cat_coverage['coverage_rate'] = cat_coverage['survived'] / cat_coverage['total']
    print(cat_coverage)

    # 5. TASK 6: INTERACTION VALIDATION
    print("\n--- [TASK 6] INTERACTION ABLATION ---")
    interactions = df_val[df_val['category'] == 'Interaction'].sort_values('perm_delta', ascending=False).head(10)
    if not interactions.empty:
        for f in interactions['feature'].tolist():
            if f not in train_base.columns: continue
            # Test WITH vs WITHOUT
            test_with = base_features + [f]
            test_without = base_features
            
            def get_mae(cols):
                oof_int = np.zeros(len(train_base))
                for tr, val in kf.split(train_base):
                    m = LGBMRegressor(**Config.RAW_LGBM_PARAMS)
                    m.fit(train_base.iloc[tr][cols], y_train[tr])
                    oof_int[val] = m.predict(train_base.iloc[val][cols])
                return mean_absolute_error(y_train, oof_int)
            
            mae_with = get_mae(test_with)
            mae_without = get_mae(test_without)
            delta = mae_without - mae_with
            print(f"Interaction {f}: delta = {delta:.6f} | {'REAL' if delta > 0.0001 else 'ACCIDENTAL'}")

if __name__ == "__main__":
    run_optimality_audit("run_20260427_100632")
