import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def architecture_comparison():
    print("--- [MISSION 10: TAIL SAFETY ARCHITECTURE COMPARISON] ---")
    
    # Load Data
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    q90_val = np.percentile(y_true, 90)
    y_binary = (y_true >= q90_val).astype(int)
    
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    X = train_df.select_dtypes(include=[np.number]).fillna(0)
    X = X.drop(columns=['avg_delay_minutes_next_30m', 'ID', 'scenario_id', 'layout_id'], errors='ignore').values
    
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    tr_idx, val_idx = next(kf.split(X))
    
    # Train Models
    clf = RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42).fit(X[tr_idx], y_binary[tr_idx])
    reg_tail = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42).fit(X[tr_idx][y_binary[tr_idx]==1], y_true[tr_idx][y_binary[tr_idx]==1])
    reg_base = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42).fit(X[tr_idx][y_binary[tr_idx]==0], y_true[tr_idx][y_binary[tr_idx]==0])
    
    p = clf.predict_proba(X[val_idx])[:, 1]
    y_tail_raw = reg_tail.predict(X[val_idx])
    y_base_raw = reg_base.predict(X[val_idx])
    y_gt = y_true[val_idx]
    
    base_mae = mean_absolute_error(y_gt, y_base_raw)
    
    # [ARCH A: Power Damping (n=1.5)]
    y_arch_a = (p**1.5) * y_tail_raw + (1 - p**1.5) * y_base_raw
    mae_a = mean_absolute_error(y_gt, y_arch_a)
    
    # [ARCH B: Cost-Aware Gate]
    # Penalty estimation (from Mission 1): FP penalty ~58, Gain ~28
    # Only activate if p * 28 > (1-p) * 58 => p > 58 / 86 approx 0.67
    y_arch_b = np.where(p > 0.67, y_tail_raw, y_base_raw)
    mae_b = mean_absolute_error(y_gt, y_arch_b)
    
    # [ARCH C: Safety-Valved Residuals]
    # Instead of full y_tail, use it as a delta but clip the delta
    residual_tail = y_tail_raw - y_base_raw
    max_safe_delta = np.percentile(y_true[tr_idx][y_binary[tr_idx]==0], 99) # Safe upper bound from base data
    clipped_delta = np.clip(residual_tail, 0, max_safe_delta)
    y_arch_c = y_base_raw + (p * clipped_delta) # Probability-weighted residual
    mae_c = mean_absolute_error(y_gt, y_arch_c)

    # [BENCHMARK TABLE]
    print("\n[TABLE: ARCHITECTURE BENCHMARK]")
    results = [
        {"Arch": "Baseline (Hard Switch 0.3)", "MAE": mean_absolute_error(y_gt, np.where(p >= 0.3, y_tail_raw, y_base_raw))},
        {"Arch": "A: Power Damping (n=1.5)", "MAE": mae_a},
        {"Arch": "B: Cost-Aware Gate", "MAE": mae_b},
        {"Arch": "C: Safety-Valved Residual", "MAE": mae_c},
        {"Arch": "No Tail (Base Only)", "MAE": base_mae}
    ]
    print(pd.DataFrame(results))

if __name__ == "__main__":
    architecture_comparison()
