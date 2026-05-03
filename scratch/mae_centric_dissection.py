import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def mae_centric_dissection():
    print("--- [MISSION: MAE-CENTRIC CAUSAL DISSECTION] ---")
    
    # Load Data
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    scenario_id = np.load(f"{processed_dir}/scenario_id.npy", allow_pickle=True)
    q90_val = np.percentile(y_true, 90)
    y_binary = (y_true >= q90_val).astype(int)
    
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    X_raw = train_df.select_dtypes(include=[np.number]).fillna(0)
    cols_to_drop = ['avg_delay_minutes_next_30m', 'ID', 'scenario_id', 'layout_id']
    X_raw = X_raw.drop(columns=[c for c in cols_to_drop if c in X_raw.columns])
    
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=2) # Fast 2-fold for sensitivity
    
    # Pre-train Stage 1 (Classifier) and Stage 2 (Regressors)
    tr, val = next(gkf.split(X_raw, y_binary, groups=scenario_id))
    X_tr, X_val = X_raw.values[tr], X_raw.values[val]
    y_tr, y_val = y_binary[tr], y_binary[val]
    y_true_tr, y_true_val = y_true[tr], y_true[val]
    
    clf = RandomForestClassifier(n_estimators=30, max_depth=8, n_jobs=-1, random_state=42).fit(X_tr, y_tr)
    p_val = clf.predict_proba(X_val)[:, 1]
    
    reg_tail = RandomForestRegressor(n_estimators=30, max_depth=8, n_jobs=-1, random_state=42).fit(X_tr[y_tr==1], y_true_tr[y_tr==1])
    reg_base = RandomForestRegressor(n_estimators=30, max_depth=8, n_jobs=-1, random_state=42).fit(X_tr[y_tr==0], y_true_tr[y_tr==0])
    
    pred_tail = reg_tail.predict(X_val)
    pred_base = reg_base.predict(X_val)
    
    # [EXP 1: MAE vs Threshold Sweep]
    print("\n[TABLE: MAE vs THRESHOLD]")
    results_thresh = []
    for t in np.linspace(0.05, 0.95, 10):
        # 2-Stage Blend: if p > t then tail else base
        # Using current blending logic (p^2) vs Hard switch
        # Let's test hard switch first to see the impact of classification boundary
        y_blend = np.where(p_val > t, pred_tail, pred_base)
        mae = mean_absolute_error(y_true_val, y_blend)
        results_thresh.append({"Threshold": t, "MAE": mae})
    print(pd.DataFrame(results_thresh))

    # [EXP 2: Single-Stage Baseline]
    reg_single = RandomForestRegressor(n_estimators=30, max_depth=8, n_jobs=-1, random_state=42).fit(X_tr, y_true_tr)
    mae_single = mean_absolute_error(y_true_val, reg_single.predict(X_val))
    print(f"\nSingle-Stage Regression MAE: {mae_single:.4f}")

    # [EXP 3: Error Contribution Analysis]
    # Where does MAE come from?
    is_tail_val = y_true_val >= q90_val
    err_tail = np.abs(y_true_val[is_tail_val] - pred_tail[is_tail_val])
    err_base = np.abs(y_true_val[~is_tail_val] - pred_base[~is_tail_val])
    print(f"\nPotential Tail MAE (if correctly ID'd): {np.mean(err_tail):.4f}")
    print(f"Potential Base MAE (if correctly ID'd): {np.mean(err_base):.4f}")

if __name__ == "__main__":
    mae_centric_dissection()
