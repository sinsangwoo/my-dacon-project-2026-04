import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def damping_validation():
    print("--- [MISSION 7: FP PENALTY MITIGATION & DAMPING VALIDATION] ---")
    
    # 1. Load Labels & Data
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    scenario_id = np.load(f"{processed_dir}/scenario_id.npy", allow_pickle=True)
    q90_val = np.percentile(y_true, 90)
    y_binary = (y_true >= q90_val).astype(int)
    
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    X = train_df.select_dtypes(include=[np.number]).fillna(0)
    X = X.drop(columns=['avg_delay_minutes_next_30m', 'ID', 'scenario_id', 'layout_id'], errors='ignore').values
    
    # Single fold split for simulation
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    tr_idx, val_idx = next(kf.split(X))
    
    # Train Models
    clf = RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42).fit(X[tr_idx], y_binary[tr_idx])
    reg_tail = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42).fit(X[tr_idx][y_binary[tr_idx]==1], y_true[tr_idx][y_binary[tr_idx]==1])
    reg_base = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42).fit(X[tr_idx][y_binary[tr_idx]==0], y_true[tr_idx][y_binary[tr_idx]==0])
    
    # Predictions on Val
    p = clf.predict_proba(X[val_idx])[:, 1]
    y_tail = reg_tail.predict(X[val_idx])
    y_base = reg_base.predict(X[val_idx])
    y_gt = y_true[val_idx]
    
    # Base Reference (No Tail at all)
    base_mae = mean_absolute_error(y_gt, y_base)
    print(f"Base MAE (No Tail): {base_mae:.4f}")

    # [EXP 1: Power Damping Sweep]
    print("\n[TABLE 1: POWER DAMPING MAE]")
    results_power = []
    for n in [1, 1.5, 2, 3, 5, 7, 10]:
        y_blend = (p**n) * y_tail + (1 - p**n) * y_base
        mae = mean_absolute_error(y_gt, y_blend)
        results_power.append({"Power_n": n, "MAE": mae, "Improvement": base_mae - mae})
    print(pd.DataFrame(results_power))

    # [EXP 2: Optimal Threshold + Power Search]
    print("\n[TABLE 2: THRESHOLD vs POWER HEATMAP (MAE)]")
    results_heat = []
    for t in [0.1, 0.2, 0.3, 0.5]:
        for n in [1, 2, 5]:
            # Blend only if p > t, else use base
            mask = p >= t
            y_blend = y_base.copy()
            y_blend[mask] = (p[mask]**n) * y_tail[mask] + (1 - p[mask]**n) * y_base[mask]
            mae = mean_absolute_error(y_gt, y_blend)
            results_heat.append({"Threshold": t, "Power": n, "MAE": mae})
    print(pd.DataFrame(results_heat).pivot(index="Threshold", columns="Power", values="MAE"))

if __name__ == "__main__":
    damping_validation()
