import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def activation_redesign_validation():
    print("--- [MISSION 8: TAIL ACTIVATION LOGIC REDESIGN & VALIDATION] ---")
    
    # Load Data
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    scenario_id = np.load(f"{processed_dir}/scenario_id.npy", allow_pickle=True)
    q90_val = np.percentile(y_true, 90)
    y_binary = (y_true >= q90_val).astype(int)
    
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    X = train_df.select_dtypes(include=[np.number]).fillna(0)
    X = X.drop(columns=['avg_delay_minutes_next_30m', 'ID', 'scenario_id', 'layout_id'], errors='ignore').values
    
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    tr_idx, val_idx = next(kf.split(X))
    
    # Train Models
    clf = RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42).fit(X[tr_idx], y_binary[tr_idx])
    reg_tail = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42).fit(X[tr_idx][y_binary[tr_idx]==1], y_true[tr_idx][y_binary[tr_idx]==1])
    reg_base = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42).fit(X[tr_idx][y_binary[tr_idx]==0], y_true[tr_idx][y_binary[tr_idx]==0])
    
    p = clf.predict_proba(X[val_idx])[:, 1]
    y_tail = reg_tail.predict(X[val_idx])
    y_base = reg_base.predict(X[val_idx])
    y_gt = y_true[val_idx]
    y_bin_gt = y_binary[val_idx]

    # [EXP 1: MAE-Optimal Recall]
    print("\n[TABLE 1: THRESHOLD vs RECALL vs MAE]")
    results_opt = []
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        y_pred_bin = (p >= t).astype(int)
        y_blend = np.where(y_pred_bin == 1, y_tail, y_base)
        mae = mean_absolute_error(y_gt, y_blend)
        prec = precision_score(y_bin_gt, y_pred_bin) if np.sum(y_pred_bin) > 0 else 0
        rec = recall_score(y_bin_gt, y_pred_bin)
        results_opt.append({"Thresh": t, "MAE": mae, "Precision": prec, "Recall": rec})
    print(pd.DataFrame(results_opt))

    # [EXP 2: Expected Gain Correlation]
    # Gain = Base_Error - Tail_Error (Positive means Tail is better)
    gain = np.abs(y_gt - y_base) - np.abs(y_gt - y_tail)
    corr = np.corrcoef(p, gain)[0, 1]
    print(f"\n[RAW NUMBER] Correlation between P and MAE Gain: {corr:.4f}")

    # [EXP 3: Tail Activation Density]
    # How many samples actually improve MAE if activated?
    improves_mae = gain > 0
    actual_tail = y_bin_gt == 1
    print(f"\nSamples where Tail is physically better: {np.sum(improves_mae)} / {len(y_gt)}")
    print(f"Overlap (Tail exists & Tail is better): {np.sum(improves_mae & actual_tail)} / {np.sum(actual_tail)}")

if __name__ == "__main__":
    activation_redesign_validation()
