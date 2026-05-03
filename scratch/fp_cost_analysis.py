import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def fp_cost_structural_analysis():
    print("--- [MISSION 1: FP COST STRUCTURAL ANALYSIS] ---")
    
    # 1. Load Labels
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    q90_val = np.percentile(y_true, 90)
    is_tail = y_true >= q90_val
    is_base = ~is_tail
    
    # 2. Reconstruct Individual Regressor Predictions (OOF Proxy)
    # We need the pure outputs of both models to see what happens when they are swapped.
    # Since we don't have separate OOF pkls for each regressor saved, 
    # we will use the training data to get a high-fidelity proxy of their performance.
    
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    X = train_df.select_dtypes(include=[np.number]).fillna(0)
    X = X.drop(columns=['avg_delay_minutes_next_30m', 'ID', 'scenario_id', 'layout_id'], errors='ignore')
    
    # Simple split to get unbiased predictions
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    tr_idx, val_idx = next(kf.split(X))
    
    X_tr, X_val = X.values[tr_idx], X.values[val_idx]
    y_tr, y_val = y_true[tr_idx], y_true[val_idx]
    
    # Train both on their respective targets (within TR fold)
    reg_tail = RandomForestRegressor(n_estimators=30, max_depth=8, n_jobs=-1, random_state=42).fit(X_tr[y_tr >= q90_val], y_tr[y_tr >= q90_val])
    reg_base = RandomForestRegressor(n_estimators=30, max_depth=8, n_jobs=-1, random_state=42).fit(X_tr[y_tr < q90_val], y_tr[y_tr < q90_val])
    
    y_tail_pred = reg_tail.predict(X_val)
    y_base_pred = reg_base.predict(X_val)
    y_true_val = y_val
    is_tail_val = y_true_val >= q90_val
    is_base_val = ~is_tail_val
    
    # [EXPERIMENT 1: FP vs FN Cost]
    # FP: Base sample predicted by Tail model
    error_base_normal = np.abs(y_true_val[is_base_val] - y_base_pred[is_base_val])
    error_base_fp = np.abs(y_true_val[is_base_val] - y_tail_pred[is_base_val])
    fp_cost_per_case = np.mean(error_base_fp - error_base_normal)
    
    # FN: Tail sample predicted by Base model
    error_tail_normal = np.abs(y_true_val[is_tail_val] - y_tail_pred[is_tail_val])
    error_tail_fn = np.abs(y_true_val[is_tail_val] - y_base_pred[is_tail_val])
    fn_cost_per_case = np.mean(error_tail_fn - error_tail_normal)
    
    print(f"\n[PHASE 1: COST PER CASE]")
    print(f"FP Mean MAE Increase: {fp_cost_per_case:.4f}")
    print(f"FN Mean MAE Increase: {fn_cost_per_case:.4f}")
    print(f"Cost Ratio (FP/FN):   {fp_cost_per_case / fn_cost_per_case:.4f}")

    # [EXPERIMENT 2: Global Sensitivity]
    print(f"\n[PHASE 2: GLOBAL SENSITIVITY]")
    results_sens = []
    total_samples = len(y_true_val)
    base_indices = np.where(is_base_val)[0]
    
    for fp_ratio in [0, 0.05, 0.10, 0.20, 0.50]:
        # Assume correct classification for all tail, but some base samples are FP
        y_sim = y_base_pred.copy()
        y_sim[is_tail_val] = y_tail_pred[is_tail_val] # Correct Tail
        
        num_fp = int(len(base_indices) * fp_ratio)
        fp_idx = np.random.choice(base_indices, num_fp, replace=False)
        y_sim[fp_idx] = y_tail_pred[fp_idx] # Injected FP
        
        mae = mean_absolute_error(y_true_val, y_sim)
        results_sens.append({"FP_Ratio": fp_ratio, "Global_MAE": mae})
    
    print(pd.DataFrame(results_sens))

if __name__ == "__main__":
    fp_cost_structural_analysis()
