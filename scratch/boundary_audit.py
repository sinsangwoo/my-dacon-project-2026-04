import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, log_loss
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def boundary_danger_zone_audit():
    print("--- [MISSION 9: BOUNDARY DANGER ZONE RE-AUDIT] ---")
    
    # Load Data
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    scenario_id = np.load(f"{processed_dir}/scenario_id.npy", allow_pickle=True)
    q85 = np.percentile(y_true, 85)
    q90 = np.percentile(y_true, 90)
    q95 = np.percentile(y_true, 95)
    y_binary = (y_true >= q90).astype(int)
    
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

    # [EXP 1: Error Concentration]
    # Zones: Base (0-85), Boundary (85-95), Tail (95-100)
    mask_base = y_gt < q85
    mask_boundary = (y_gt >= q85) & (y_gt < q95)
    mask_tail = y_gt >= q95
    
    # Blended MAE using current structure (hard switch 0.3)
    y_blend = np.where(p >= 0.3, y_tail, y_base)
    errors = np.abs(y_gt - y_blend)
    
    print("\n[TABLE 1: ZONE-WISE ERROR DENSITY]")
    zone_results = []
    for name, m in [("Base (0-85)", mask_base), ("Boundary (85-95)", mask_boundary), ("Tail (95-100)", mask_tail)]:
        zone_err = np.sum(errors[m])
        zone_mae = np.mean(errors[m])
        zone_samples = np.sum(m)
        zone_contribution = (zone_err / np.sum(errors)) * 100
        # Classification error in this zone
        is_wrong = (p[m] >= 0.3) != y_bin_gt[m]
        clf_err_rate = np.mean(is_wrong)
        zone_results.append({"Zone": name, "MAE": zone_mae, "Samples": zone_samples, "MAE_Contrib_%": zone_contribution, "Clf_Err_Rate": clf_err_rate})
    print(pd.DataFrame(zone_results))

    # [EXP 2: Probability Distribution in Boundary]
    print("\n[TABLE 2: BOUNDARY PROBABILITY DISTRIBUTION]")
    p_boundary = p[mask_boundary]
    hist, bins = np.histogram(p_boundary, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    print(pd.DataFrame({"Prob_Bin": [f"{bins[i]}-{bins[i+1]}" for i in range(5)], "Sample_Count": hist, "Ratio": hist / len(p_boundary)}))

if __name__ == "__main__":
    boundary_danger_zone_audit()
