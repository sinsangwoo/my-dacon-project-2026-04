import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
import gc

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def mae_driven_mission_opt():
    print("--- [MISSION: MAE-DRIVEN ATTACK ZONE (OPT)] ---")
    
    # 1. Load Data
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    q90_val = np.percentile(y_true, 90)
    y_binary = (y_true >= q90_val).astype(int)
    
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    X = train_df.select_dtypes(include=[np.number]).fillna(0)
    X = X.drop(columns=['avg_delay_minutes_next_30m', 'ID', 'scenario_id', 'layout_id'], errors='ignore')
    
    # [OPTIMIZATION] Sample 20% for speed while keeping distribution integrity
    sample_mask = np.random.choice([True, False], len(y_true), p=[0.2, 0.8])
    X = X[sample_mask].values
    y_true = y_true[sample_mask]
    y_binary = y_binary[sample_mask]
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    oof_base = np.zeros(len(y_true))
    oof_tail = np.zeros(len(y_true))
    oof_p = np.zeros(len(y_true))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y_binary[tr_idx], y_binary[val_idx]
        y_gt_tr, y_gt_val = y_true[tr_idx], y_true[val_idx]
        
        clf = RandomForestClassifier(n_estimators=10, max_depth=6, n_jobs=-1).fit(X_tr, y_tr)
        reg_base = RandomForestRegressor(n_estimators=10, max_depth=6, n_jobs=-1).fit(X_tr[y_tr==0], y_gt_tr[y_tr==0])
        reg_tail = RandomForestRegressor(n_estimators=10, max_depth=6, n_jobs=-1).fit(X_tr[y_tr==1], y_gt_tr[y_tr==1])
        
        oof_p[val_idx] = clf.predict_proba(X_val)[:, 1]
        oof_base[val_idx] = reg_base.predict(X_val)
        oof_tail[val_idx] = reg_tail.predict(X_val)
        gc.collect()

    mae_gain = np.abs(y_true - oof_base) - np.abs(y_true - oof_tail)
    
    print("\n[TASK 1 & 2: GAIN STATS]")
    print(pd.Series({
        "Ratio_Gain_Positive": np.mean(mae_gain > 0),
        "Mean_Gain": np.mean(mae_gain),
        "P99_Gain": np.percentile(mae_gain, 99)
    }))

    print("\n[TASK 3: GAIN vs P]")
    print(f"Corr(p, Gain): {np.corrcoef(oof_p, mae_gain)[0,1]:.4f}")
    
    p_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i in range(len(p_bins)-1):
        mask = (oof_p >= p_bins[i]) & (oof_p < p_bins[i+1])
        if np.sum(mask) > 0:
            print(f"P {p_bins[i]}-{p_bins[i+1]}: Gain {np.mean(mae_gain[mask]):.4f} (Size {np.sum(mask)})")

    print("\n[TASK 5: CONTROL]")
    y_oracle = np.where(mae_gain > 0, oof_tail, oof_base)
    print(pd.DataFrame([
        {"Method": "No Tail", "MAE": mean_absolute_error(y_true, oof_base)},
        {"Method": "Oracle", "MAE": mean_absolute_error(y_true, y_oracle)}
    ]))

    # [FINAL QUESTION ANSWER DATA]
    print("\n[FINAL QUESTION DATA]")
    # What are the commonalities of samples we should attack? (Gain > 0)
    # Check actual y values for Gain > 0
    print(f"Mean y for Gain > 0: {np.mean(y_true[mae_gain > 0]):.4f}")
    print(f"Mean y for Gain <= 0: {np.mean(y_true[mae_gain <= 0]):.4f}")

if __name__ == "__main__":
    mae_driven_mission_opt()
