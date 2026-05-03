import os
import numpy as np
import pandas as pd
import pickle
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

def run_decomposition_study():
    RUN_ID = "run_20260430_231842"
    BASE_PATH = f"./outputs/{RUN_ID}"
    
    y_all = np.load(f"{BASE_PATH}/processed/y_train.npy").astype(np.float64)
    scenarios = np.load(f"{BASE_PATH}/processed/scenario_id.npy", allow_pickle=True)
    
    # Fast Sampling (10%)
    indices = np.arange(len(y_all))
    np.random.seed(42)
    sample_idx = np.random.choice(indices, size=int(len(indices)*0.1), replace=False)
    
    y = y_all[sample_idx]
    s = scenarios[sample_idx]
    
    print("Loading features...")
    with open(f"{BASE_PATH}/processed/train_base.pkl", "rb") as f:
        X_df = pickle.load(f)
    
    X_sample = X_df.iloc[sample_idx]
    X_numeric = X_sample.select_dtypes(include=[np.number])
    X_vals = X_numeric.values
    
    gkf = GroupKFold(n_splits=2)
    train_idx, val_idx = next(gkf.split(np.arange(len(y)), y, groups=s))
    
    X_tr, X_val = X_vals[train_idx], X_vals[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    results = []
    q90_global = np.quantile(y_all, 0.90)

    def evaluate(preds, y_true, name):
        tail_mask = y_true >= q90_global
        m_ratio = np.mean(preds) / np.mean(y_true)
        # Handle cases where there might be no tail samples in the small split
        if np.sum(tail_mask) > 0:
            t_ratio = np.mean(preds[tail_mask]) / np.mean(y_true[tail_mask])
            t_mae = mean_absolute_error(y_true[tail_mask], preds[tail_mask])
        else:
            t_ratio = 0
            t_mae = 0
        mae = mean_absolute_error(y_true, preds)
        return {"name": name, "mae": mae, "mean_ratio": m_ratio, "tail_ratio": t_ratio, "tail_mae": t_mae}

    # EXP 3: Transform Only
    print("EXP 3...")
    model_exp3 = LGBMRegressor(objective='regression_l1', n_estimators=100, verbose=-1)
    model_exp3.fit(X_tr, np.log1p(y_tr))
    preds_exp3 = np.expm1(model_exp3.predict(X_val))
    results.append(evaluate(preds_exp3, y_val, "EXP 3: Log-Transform Only"))

    # EXP 2: No Transform
    print("EXP 2...")
    model_exp2 = LGBMRegressor(objective='regression_l1', n_estimators=100, verbose=-1)
    model_exp2.fit(X_tr, y_tr)
    preds_exp2 = model_exp2.predict(X_val)
    results.append(evaluate(preds_exp2, y_val, "EXP 2: No Transform (Raw)"))

    # EXP 1: No Blending (Tail-only)
    print("EXP 1...")
    tail_mask_tr = y_tr >= q90_global
    tail_mask_val = y_val >= q90_global
    model_tail = LGBMRegressor(objective='regression', n_estimators=100, verbose=-1)
    model_tail.fit(X_tr[tail_mask_tr], np.log1p(y_tr[tail_mask_tr]))
    preds_tail = np.expm1(model_tail.predict(X_val[tail_mask_val]))
    
    t_ratio_exp1 = np.mean(preds_tail) / np.mean(y_val[tail_mask_val])
    results.append({
        "name": "EXP 1: No Blending (Tail-Log)", 
        "mae": 0, # Not comparable for full set
        "mean_ratio": 0, 
        "tail_ratio": t_ratio_exp1,
        "tail_mae": mean_absolute_error(y_val[tail_mask_val], preds_tail)
    })

    # Output Report
    df = pd.DataFrame(results)
    print("\n--- [ROOT CAUSE DECOMPOSITION RESULTS] ---")
    print(df.to_string())

if __name__ == "__main__":
    run_decomposition_study()
