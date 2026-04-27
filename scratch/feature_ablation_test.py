import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import os
import json

def run_ablation_test(run_id):
    print(f"=== [ABLATION TEST: {run_id}] ===")
    
    # 1. Load Data
    data_dir = f"outputs/{run_id}/processed"
    train_df = pd.read_pickle(f"{data_dir}/train_base.pkl")
    y = np.load(f"{data_dir}/y_train.npy")
    
    with open(f"{data_dir}/signal_validation_logs.json", 'r') as f:
        val_data = json.load(f)
    
    val_logs = pd.DataFrame(val_data['val_logs'])
    
    # 2. Define Feature Sets
    stable_features = val_logs[val_logs['passed']]['feature'].tolist()
    
    # Rejected but high-impact features (Perm Delta > 0.001)
    high_impact_rejected = val_logs[~val_logs['passed'] & (val_logs['perm_delta'] > 0.001)]['feature'].tolist()
    
    print(f"Stable features count: {len(stable_features)}")
    print(f"High-impact rejected features count: {len(high_impact_rejected)}")
    
    # 3. CV Function
    def evaluate(features, label):
        if not features:
            print(f"[{label}] No features to evaluate.")
            return np.inf
            
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        maes = []
        
        for tr_idx, val_idx in kf.split(train_df):
            X_tr, X_val = train_df.iloc[tr_idx][features], train_df.iloc[val_idx][features]
            y_tr, y_val = y[tr_idx], y[val_idx]
            
            model = LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42, verbose=-1)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            maes.append(mean_absolute_error(y_val, preds))
            
        avg_mae = np.mean(maes)
        print(f"[{label}] MAE: {avg_mae:.6f} (Features: {len(features)})")
        return avg_mae

    # 4. Execute Comparison
    mae_stable = evaluate(stable_features, "STABLE_ONLY")
    mae_expanded = evaluate(stable_features + high_impact_rejected, "STABLE + HIGH_IMPACT_REJECTED")
    
    diff = mae_stable - mae_expanded
    print(f"\nMAE Improvement by including rejected features: {diff:.6f}")
    
    if diff > 0:
        print("RESULT: Underfitting confirmed. Pruning is too aggressive.")
    else:
        print("RESULT: Stability holds. Pruning is safe (but possibly over-cautious).")

if __name__ == "__main__":
    run_ablation_test('run_20260426_235007')
