import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, brier_score_loss
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import IsotonicRegression, calibration_curve
from sklearn.model_selection import GroupKFold

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def full_system_rebuild_validation():
    print("--- [MISSION: PROBABILISTIC TAIL CONTROL SYSTEM REBUILD - 100% TRACE] ---")
    
    # 1. Load Data
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    scenario_id = np.load(f"{processed_dir}/scenario_id.npy", allow_pickle=True)
    q90_val = np.percentile(y_true, 90)
    y_binary = (y_true >= q90_val).astype(int)
    
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    X = train_df.select_dtypes(include=[np.number]).fillna(0)
    X = X.drop(columns=['avg_delay_minutes_next_30m', 'ID', 'scenario_id', 'layout_id'], errors='ignore').values
    
    gkf = GroupKFold(n_splits=2)
    tr_idx, val_idx = next(gkf.split(X, y_binary, groups=scenario_id))
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y_binary[tr_idx], y_binary[val_idx]
    y_gt_val = y_true[val_idx]
    
    # Models
    clf = RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42).fit(X_tr, y_tr)
    reg_tail = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42).fit(X_tr[y_tr==1], y_true[tr_idx][y_tr==1])
    reg_base = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42).fit(X_tr[y_tr==0], y_true[tr_idx][y_tr==0])
    
    p_raw = clf.predict_proba(X_val)[:, 1]
    y_tail = reg_tail.predict(X_val)
    y_base = reg_base.predict(X_val)
    gain = np.abs(y_gt_val - y_base) - np.abs(y_gt_val - y_tail)

    # [TASK 4: Probability Validation & Calibration]
    print("\n[TASK 4: PROBABILITY VALIDATION]")
    brier_before = brier_score_loss(y_val, p_raw)
    corr_before = np.corrcoef(p_raw, gain)[0, 1]
    
    iso = IsotonicRegression(out_of_bounds='clip').fit(clf.predict_proba(X_tr)[:, 1], y_tr)
    p_cal = iso.transform(p_raw)
    
    brier_after = brier_score_loss(y_val, p_cal)
    corr_after = np.corrcoef(p_cal, gain)[0, 1]
    
    print(f"Brier Score: Before={brier_before:.4f}, After={brier_after:.4f}")
    print(f"P-Gain Corr: Before={corr_before:.4f}, After={corr_after:.4f}")

    def get_metrics(y_pred, y_true_val, p_val):
        mae = mean_absolute_error(y_true_val, y_pred)
        fp_mask = (y_true_val < q90_val) & (y_pred > y_base)
        fn_mask = (y_true_val >= q90_val) & (y_pred <= y_base)
        fp_damage = np.sum(np.abs(y_pred[fp_mask] - y_true_val[fp_mask])) / len(y_true_val)
        b_mask = (p_val >= 0.2) & (p_val <= 0.4)
        b_contrib = np.sum(np.abs(y_pred[b_mask] - y_true_val[b_mask])) / len(y_true_val)
        return mae, np.mean(fp_mask), np.mean(fn_mask), fp_damage, b_contrib

    # [TASK 1-3 Summary Table]
    print("\n[TASK 1-3: STRUCTURAL REDESIGN RESULTS]")
    # Task 3 Residual Clipping
    max_delta = np.percentile(np.abs(y_true[tr_idx][y_tr==0] - reg_base.predict(X_tr[y_tr==0])), 95)
    delta_clipped = np.clip(y_tail - y_base, -max_delta, max_delta)
    
    results_t13 = []
    # Power n=2 + Clipping
    y_p2_clip = y_base + (p_cal**2) * delta_clipped
    results_t13.append({"Method": "Power n=2 + Clip", "Metrics": get_metrics(y_p2_clip, y_gt_val, p_cal)})
    # B-Suppression + Clip
    y_supp_clip = y_p2_clip.copy()
    y_supp_clip[(p_cal >= 0.2) & (p_cal <= 0.4)] = y_base[(p_cal >= 0.2) & (p_cal <= 0.4)]
    results_t13.append({"Method": "B-Supp + Clip", "Metrics": get_metrics(y_supp_clip, y_gt_val, p_cal)})
    
    for r in results_t13:
        m = r['Metrics']
        print(f"{r['Method']}: MAE={m[0]:.4f}, FP_R={m[1]:.4f}, FN_R={m[2]:.4f}, FP_Cost={m[3]:.4f}, B_Contrib={m[4]:.4f}")

    # [TASK 5: MISSION 10 FULL VALIDATION]
    print("\n[TASK 5: FINAL SYSTEM OPTIMIZATION SWEEP]")
    # Re-run Power Damping under: Calibrated P + Residual Clipping
    results_t5 = []
    for n in [1.0, 1.5, 2.0, 2.5, 3.0]:
        y_final = y_base + (p_cal**n) * delta_clipped
        m = get_metrics(y_final, y_gt_val, p_cal)
        results_t5.append({"n": n, "MAE": m[0], "FP_R": m[1], "FP_Cost": m[3]})
    
    print(pd.DataFrame(results_t5))

if __name__ == "__main__":
    full_system_rebuild_validation()
