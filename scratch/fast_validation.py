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

def fast_critical_validation():
    print("--- [MISSION: FAST CRITICAL VALIDATION] ---")
    
    # 1. Load Data
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    q90 = np.percentile(y_true, 90)
    y_binary = (y_true >= q90).astype(int)
    
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    X = train_df.select_dtypes(include=[np.number]).fillna(0)
    X = X.drop(columns=['avg_delay_minutes_next_30m', 'ID', 'scenario_id', 'layout_id'], errors='ignore').values
    
    # [SAMPLING] 10% for extreme speed
    sample_mask = np.random.choice([True, False], len(y_true), p=[0.1, 0.9])
    X = X[sample_mask]
    y_true = y_true[sample_mask]
    y_binary = y_binary[sample_mask]
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    # [TASK 1 & 3: CONFIGURATIONS]
    configs = ["Full", "No_Validator", "No_EV"]
    results = []

    for cfg in configs:
        oof_base = np.zeros(len(y_true))
        oof_tail = np.zeros(len(y_true))
        oof_p = np.zeros(len(y_true))
        
        for tr, val in kf.split(X):
            clf = RandomForestClassifier(n_estimators=10, max_depth=5).fit(X[tr], y_binary[tr])
            reg_b = RandomForestRegressor(n_estimators=10, max_depth=5).fit(X[tr][y_binary[tr]==0], y_true[tr][y_binary[tr]==0])
            reg_t = RandomForestRegressor(n_estimators=10, max_depth=5).fit(X[tr][y_binary[tr]==1], y_true[tr][y_binary[tr]==1])
            
            p = clf.predict_proba(X[val])[:, 1]
            b = reg_b.predict(X[val])
            t = reg_t.predict(X[val])
            
            gain_est = b - t
            loss_est = 50.0
            ev = p * gain_est - (1 - p) * loss_est
            
            # Simulated Validator / Execution Layer
            if cfg == "Full":
                # With Validator (Consistency check simulation)
                confidence = 0.5 # Simplified
                strength = np.clip(ev / 50.0, 0, 1.0) * confidence
            elif cfg == "No_Validator":
                # Raw EV
                strength = np.clip(ev / 50.0, 0, 1.0)
            elif cfg == "No_EV":
                # Only p and damping
                strength = p ** 2
            
            oof_base[val] = b
            oof_tail[val] = b + strength * (t - b)
        
        mae = mean_absolute_error(y_true, oof_tail)
        # FP Cost: When tail activation made error worse
        fp_mask = (np.abs(y_true - oof_base) < np.abs(y_true - oof_tail))
        fp_cost = np.mean(np.abs(y_true[fp_mask] - oof_tail[fp_mask]))
        results.append({"Config": cfg, "MAE": mae, "FP_Cost": fp_cost})

    print("\n[TASK 1 & 3: RESULTS]")
    res_df = pd.DataFrame(results)
    print(res_df)

    # [TASK 2: LEAKAGE SHUFFLE TEST]
    print("\n[TASK 2: SHUFFLE TEST (LEAKAGE AUDIT)]")
    y_shuffled = y_true.copy()
    np.random.shuffle(y_shuffled)
    
    oof_shuffled = np.zeros(len(y_true))
    for tr, val in kf.split(X):
        reg_shuffled = RandomForestRegressor(n_estimators=10, max_depth=5).fit(X[tr], y_shuffled[tr])
        oof_shuffled[val] = reg_shuffled.predict(X[val])
    
    mae_shuffled = mean_absolute_error(y_shuffled, oof_shuffled)
    mae_baseline_shuffled = mean_absolute_error(y_shuffled, np.full_like(y_shuffled, np.mean(y_shuffled)))
    print(f"Shuffled Target MAE: {mae_shuffled:.4f}")
    print(f"Shuffled Baseline MAE: {mae_baseline_shuffled:.4f}")
    print(f"Leakage Detected: {mae_shuffled < mae_baseline_shuffled * 0.95}") # 5% margin

if __name__ == "__main__":
    fast_critical_validation()
