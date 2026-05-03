import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def oracle_mae_validation():
    print("--- [MISSION 3: ORACLE MAE VALIDATION] ---")
    
    # Load Data
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    scenario_id = np.load(f"{processed_dir}/scenario_id.npy", allow_pickle=True)
    q90_val = np.percentile(y_true, 90)
    y_binary = (y_true >= q90_val).astype(int)
    
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    X = train_df.select_dtypes(include=[np.number]).fillna(0)
    X = X.drop(columns=['avg_delay_minutes_next_30m', 'ID', 'scenario_id', 'layout_id'], errors='ignore').values
    
    gkf = GroupKFold(n_splits=5)
    
    oracle_maes = []
    real_maes = []
    partial_results = []
    
    print("\n[PHASE 1-3: FOLD-WISE VALIDATION]")
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_binary, groups=scenario_id)):
        # Train Regressors
        reg_tail = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=fold).fit(X[tr_idx][y_binary[tr_idx]==1], y_true[tr_idx][y_binary[tr_idx]==1])
        reg_base = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=fold).fit(X[tr_idx][y_binary[tr_idx]==0], y_true[tr_idx][y_binary[tr_idx]==0])
        
        # Train Real Classifier
        clf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=fold).fit(X[tr_idx], y_binary[tr_idx])
        
        # Predictions
        p_real = clf.predict_proba(X[val_idx])[:, 1]
        y_tail_p = reg_tail.predict(X[val_idx])
        y_base_p = reg_base.predict(X[val_idx])
        
        # 1. Oracle MAE (Using GT y_binary)
        y_oracle_blend = np.where(y_binary[val_idx] == 1, y_tail_p, y_base_p)
        f_oracle_mae = mean_absolute_error(y_true[val_idx], y_oracle_blend)
        oracle_maes.append(f_oracle_mae)
        
        # 2. Real MAE (Using Static Threshold 0.3)
        y_real_blend = np.where(p_real >= 0.3, y_tail_p, y_base_p)
        f_real_mae = mean_absolute_error(y_true[val_idx], y_real_blend)
        real_maes.append(f_real_mae)
        
        # 4. Partial Oracle Simulation (Top K%)
        if fold == 0: # Performance optimization: run sweep on one fold
            for k in [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
                # Top k% of p_real are replaced by GT
                num_replace = int(len(p_real) * k)
                top_k_indices = np.argsort(p_real)[::-1][:num_replace]
                
                y_partial_binary = (p_real >= 0.3).astype(int)
                y_partial_binary[top_k_indices] = y_binary[val_idx][top_k_indices]
                
                y_p_blend = np.where(y_partial_binary == 1, y_tail_p, y_base_p)
                p_mae = mean_absolute_error(y_true[val_idx], y_p_blend)
                partial_results.append({"k": k, "MAE": p_mae})

    # [TABLE 1: ORACLE vs REAL]
    print("\n[TABLE 1: ORACLE vs REAL MAE]")
    df_comp = pd.DataFrame({"Fold": range(5), "Oracle_MAE": oracle_maes, "Real_MAE": real_maes})
    print(df_comp)
    
    # [RAW NUMBERS: STABILITY]
    print("\n[RAW NUMBERS: UPPER BOUND STABILITY]")
    print(f"Mean Oracle MAE: {np.mean(oracle_maes):.4f}")
    print(f"Std Oracle MAE:  {np.std(oracle_maes):.4f}")
    print(f"Mean Real MAE:   {np.mean(real_maes):.4f}")
    print(f"Performance Gap: {np.mean(real_maes) - np.mean(oracle_maes):.4f}")

    # [TABLE 2: PARTIAL ORACLE]
    print("\n[TABLE 2: PARTIAL ORACLE SWEEP]")
    print(pd.DataFrame(partial_results))

if __name__ == "__main__":
    oracle_mae_validation()
