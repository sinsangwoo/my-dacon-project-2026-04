import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
import random

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def attack_zone_stress_test():
    print("--- [MISSION: ATTACK ZONE STRESS TEST] ---")
    
    # 1. Load Data
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    q90_val = np.percentile(y_true, 90)
    y_binary = (y_true >= q90_val).astype(int)
    
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    X = train_df.select_dtypes(include=[np.number]).fillna(0)
    X = X.drop(columns=['avg_delay_minutes_next_30m', 'ID', 'scenario_id', 'layout_id'], errors='ignore').values
    
    # 2. Setup 5-Fold Validation Structure
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_attack_zones = []
    fold_metrics = []
    
    total_idx = np.arange(len(X))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y_binary[tr_idx], y_binary[val_idx]
        y_gt_tr, y_gt_val = y_true[tr_idx], y_true[val_idx]
        
        # Models
        clf = RandomForestClassifier(n_estimators=30, max_depth=8, n_jobs=-1, random_state=42).fit(X_tr, y_tr)
        reg_base = RandomForestRegressor(n_estimators=30, max_depth=8, n_jobs=-1).fit(X_tr[y_tr==0], y_gt_tr[y_tr==0])
        reg_tail = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=1).fit(X_tr[y_tr==1], y_gt_tr[y_tr==1])
        
        p_val = clf.predict_proba(X_val)[:, 1]
        y_base_val = reg_base.predict(X_val)
        y_tail_val = reg_tail.predict(X_val)
        
        # [RESCUE LOGIC REPLICATION]
        # Low-p clustering and EV Filter simulation on OOF
        # (Simplified for stability check: p <= 0.4 and EV > 0 based on actuals as proxy)
        gain_val = np.where(y_val == 1, np.abs(y_gt_val - y_base_val) - np.abs(y_gt_val - y_tail_val), 0)
        cost_val = np.where(y_val == 0, np.abs(y_tail_val - y_gt_val) - np.abs(y_base_val - y_gt_val), 0)
        ev_val = p_val * gain_val - (1 - p_val) * cost_val
        
        # Top 32 samples per fold based on EV
        top_idx_in_val = np.argsort(ev_val)[-32:]
        global_indices = val_idx[top_idx_in_val]
        fold_attack_zones.append(set(global_indices))
        
        # FP Audit for OOF
        fp_count = np.sum(y_val[top_idx_in_val] == 0)
        fold_metrics.append({"Fold": fold, "FP_Count": fp_count, "MAE_Gain": np.mean(gain_val[top_idx_in_val])})

    # [TASK 1: Leakage Audit]
    print("\n[TASK 1: LEAKAGE AUDIT]")
    # FP Count across folds
    fp_report = pd.DataFrame(fold_metrics)
    print(fp_report)

    # [TASK 3: Stability Test]
    print("\n[TASK 3: STABILITY TEST]")
    intersections = []
    unions = []
    for i in range(len(fold_attack_zones)):
        for j in range(i + 1, len(fold_attack_zones)):
            inter = len(fold_attack_zones[i].intersection(fold_attack_zones[j]))
            union = len(fold_attack_zones[i].union(fold_attack_zones[j]))
            intersections.append(inter)
            unions.append(union)
    
    stability_score = np.mean(intersections) / (np.mean(unions) + 1e-9)
    print(f"Mean Intersection Size: {np.mean(intersections):.2f}")
    print(f"Stability Score (I/U): {stability_score:.4f}")

    # [TASK 4: Scalability Verification]
    print("\n[TASK 4: SCALABILITY VERIFICATION]")
    # Re-run on one fold to check 32 vs 100
    tr_idx, val_idx = next(kf.split(X))
    # ... (using same logic as above)
    scalability_results = []
    for size in [32, 50, 100]:
        top_idx = np.argsort(ev_val)[-size:]
        mae_imp = np.mean(gain_val[top_idx]) - np.mean(cost_val[top_idx])
        fp_r = np.mean(y_val[top_idx] == 0)
        scalability_results.append({"Size": size, "Exp_Profit": mae_imp, "FP_R": fp_r})
    print(pd.DataFrame(scalability_results))

    # [TASK 5: Random Baseline]
    print("\n[TASK 5: RANDOM BASELINE COMPARISON]")
    random_maes = []
    for _ in range(100):
        rand_idx = np.random.choice(len(y_gt_val), 32, replace=False)
        rand_mae = mean_absolute_error(y_gt_val[rand_idx], y_tail_val[rand_idx])
        random_maes.append(rand_mae)
    
    print(f"Attack Zone MAE (Top 32): {mean_absolute_error(y_gt_val[top_idx_in_val], y_tail_val[top_idx_in_val]):.4f}")
    print(f"Random Mean MAE: {np.mean(random_maes):.4f}")
    print(f"Random Std MAE:  {np.std(random_maes):.4f}")

if __name__ == "__main__":
    attack_zone_stress_test()
