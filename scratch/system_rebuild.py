import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import IsotonicRegression
from sklearn.model_selection import GroupKFold

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def rebuild_probabilistic_system():
    print("--- [MISSION: PROBABILISTIC TAIL CONTROL SYSTEM REBUILD] ---")
    
    # 1. Load Data
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    scenario_id = np.load(f"{processed_dir}/scenario_id.npy", allow_pickle=True)
    q90_val = np.percentile(y_true, 90)
    y_binary = (y_true >= q90_val).astype(int)
    
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    X = train_df.select_dtypes(include=[np.number]).fillna(0)
    X = X.drop(columns=['avg_delay_minutes_next_30m', 'ID', 'scenario_id', 'layout_id'], errors='ignore').values
    
    gkf = GroupKFold(n_splits=2) # 2-fold for faster structural testing
    tr_idx, val_idx = next(gkf.split(X, y_binary, groups=scenario_id))
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y_binary[tr_idx], y_binary[val_idx]
    y_gt_val = y_true[val_idx]
    
    # Pre-train Models
    clf = RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42).fit(X_tr, y_tr)
    reg_tail = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42).fit(X_tr[y_tr==1], y_true[tr_idx][y_tr==1])
    reg_base = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42).fit(X_tr[y_tr==0], y_true[tr_idx][y_tr==0])
    
    p_raw = clf.predict_proba(X_val)[:, 1]
    y_tail = reg_tail.predict(X_val)
    y_base = reg_base.predict(X_val)
    
    # [TASK 4: Calibration]
    print("\n[TASK 4] Probability Calibration...")
    iso = IsotonicRegression(out_of_bounds='clip').fit(clf.predict_proba(X_tr)[:, 1], y_tr)
    p = iso.transform(p_raw)
    
    def get_metrics(y_pred, y_true_val, p_val, y_bin_val):
        mae = mean_absolute_error(y_true_val, y_pred)
        # FP defined as samples where GT < Q90 but prediction is affected by tail
        fp_mask = (y_true_val < q90_val) & (y_pred > y_base)
        fp_rate = np.mean(fp_mask)
        # FN defined as samples where GT >= Q90 but prediction is near base
        fn_mask = (y_true_val >= q90_val) & (y_pred <= y_base)
        fn_rate = np.mean(fn_mask)
        # FP Damage
        fp_damage = np.sum(np.abs(y_pred[fp_mask] - y_true_val[fp_mask])) / len(y_true_val)
        # Boundary (0.2 <= p <= 0.4)
        b_mask = (p_val >= 0.2) & (p_val <= 0.4)
        b_contrib = np.sum(np.abs(y_pred[b_mask] - y_true_val[b_mask])) / len(y_true_val)
        return mae, fp_rate, fn_rate, fp_damage, b_contrib

    # [TASK 1: Continuous Blending]
    print("\n[TASK 1] Continuous Blending Formulations")
    # A. Power Damping n=2
    y_t1a = y_base + (p**2) * (y_tail - y_base)
    # B. Confidence-normalized
    mu, std = np.mean(p), np.std(p)
    w_b = np.clip((p - mu) / (std + 1e-9), 0, 1)
    y_t1b = y_base + w_b * (y_tail - y_base)
    # C. Rank-based
    w_c = pd.Series(p).rank(pct=True).values
    y_t1c = y_base + w_c * (y_tail - y_base)
    
    for name, yp in [("Power Damping (n=2)", y_t1a), ("Conf-Normalized", y_t1b), ("Rank-based", y_t1c)]:
        m = get_metrics(yp, y_gt_val, p, y_val)
        print(f"{name}: MAE={m[0]:.4f}, FP_R={m[1]:.4f}, FN_R={m[2]:.4f}, FP_Cost={m[3]:.4f}, B_Contrib={m[4]:.4f}")

    # [TASK 2: Boundary Isolation]
    print("\n[TASK 2] Boundary Isolation Strategies (Base: Power Damping n=2)")
    b_mask = (p >= 0.2) & (p <= 0.4)
    # A. Suppression
    y_t2a = y_t1a.copy()
    y_t2a[b_mask] = y_base[b_mask]
    # B. Soft attenuation (alpha=0.3)
    y_t2b = y_base + (p**2) * (y_tail - y_base)
    y_t2b[b_mask] = y_base[b_mask] + 0.3 * (p[b_mask]**2) * (y_tail[b_mask] - y_base[b_mask])
    
    for name, yp in [("B-Suppression", y_t2a), ("B-Attenuation", y_t2b)]:
        m = get_metrics(yp, y_gt_val, p, y_val)
        print(f"{name}: MAE={m[0]:.4f}, FP_R={m[1]:.4f}, FN_R={m[2]:.4f}, FP_Cost={m[3]:.4f}, B_Contrib={m[4]:.4f}")

    # [TASK 3: FP Cost Structural Reduction]
    print("\n[TASK 3] FP Cost Reduction (Base: Power Damping n=2)")
    delta = y_tail - y_base
    # A. Residual Clipping (95th percentile of base errors)
    max_delta = np.percentile(np.abs(y_true[tr_idx][y_tr==0] - reg_base.predict(X_tr[y_tr==0])), 95)
    clipped_delta = np.clip(delta, -max_delta, max_delta)
    y_t3a = y_base + (p**2) * clipped_delta
    
    m = get_metrics(y_t3a, y_gt_val, p, y_val)
    print(f"Residual Clipping: MAE={m[0]:.4f}, FP_R={m[1]:.4f}, FN_R={m[2]:.4f}, FP_Cost={m[3]:.4f}, B_Contrib={m[4]:.4f}")

if __name__ == "__main__":
    rebuild_probabilistic_system()
