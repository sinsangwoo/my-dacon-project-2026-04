import numpy as np
import pandas as pd
import os
import joblib
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, auc, mean_absolute_error, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import calibration_curve
from scipy.stats import entropy

# Paths & Setup
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def run_integrated_causal_mission():
    print(f"--- [MISSION: INTEGRATED CAUSAL VALIDATION & INTERVENTION] ---")
    
    # 1. Load Data
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    scenario_id = np.load(f"{processed_dir}/scenario_id.npy", allow_pickle=True)
    q90_val = np.percentile(y_true, 90)
    y_binary = (y_true >= q90_val).astype(int)
    
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    X_raw = train_df.select_dtypes(include=[np.number]).fillna(0)
    cols_to_drop = ['avg_delay_minutes_next_30m', 'ID', 'scenario_id', 'layout_id']
    X_raw = X_raw.drop(columns=[c for c in cols_to_drop if c in X_raw.columns])
    
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    
    # [PHASE 1: BASELINE SNAPSHOT]
    print("\n[PHASE 1] Baseline Snapshot...")
    oof_p_base = np.zeros(len(y_binary))
    for tr_idx, val_idx in gkf.split(X_raw, y_binary, groups=scenario_id):
        clf = HistGradientBoostingClassifier(max_iter=100, max_depth=10, random_state=42)
        clf.fit(X_raw.values[tr_idx], y_binary[tr_idx])
        oof_p_base[val_idx] = clf.predict_proba(X_raw.values[val_idx])[:, 1]
    
    p_base, r_base, t_base = precision_recall_curve(y_binary, oof_p_base)
    pr_auc_base = auc(r_base, p_base)
    
    f1_scores = 2 * (p_base * r_base) / (p_base + r_base + 1e-9)
    opt_idx = np.argmax(f1_scores)
    opt_thresh = t_base[opt_idx] if opt_idx < len(t_base) else 0.5
    
    print(f"Baseline PR AUC: {pr_auc_base:.4f}")
    print(f"Optimal Threshold (F1): {opt_thresh:.4f}")
    
    def get_prec_at_k(p_scores, y_true, k_pct):
        thresh = np.percentile(p_scores, 100 - k_pct)
        return precision_score(y_true, (p_scores >= thresh).astype(int))

    print(f"Prec@5%:  {get_prec_at_k(oof_p_base, y_binary, 5):.4f}")
    print(f"Prec@10%: {get_prec_at_k(oof_p_base, y_binary, 10):.4f}")
    print(f"Prec@15%: {get_prec_at_k(oof_p_base, y_binary, 15):.4f}")

    # [PHASE 2: H4 BOUNDARY INTERVENTION]
    print("\n[PHASE 2] H4 Boundary Intervention...")
    boundary_results = []
    for q in [80, 85, 90, 95]:
        q_val = np.percentile(y_true, q)
        y_q = (y_true >= q_val).astype(int)
        # Measure separation at this boundary
        # We'll use a single fold proxy for speed but high precision
        tr, val = next(gkf.split(X_raw, y_q, groups=scenario_id))
        clf_q = HistGradientBoostingClassifier(max_iter=50, random_state=42).fit(X_raw.values[tr], y_q[tr])
        p_q = clf_q.predict_proba(X_raw.values[val])[:, 1]
        cur_pr_auc = roc_auc_score(y_q[val], p_q) # Using ROC AUC as proxy for pure separability
        boundary_results.append({"Threshold": f"Q{q}", "Separability(AUC)": cur_pr_auc})
    print(pd.DataFrame(boundary_results))

    # [PHASE 3: H5 NOISE vs INFO GAP]
    print("\n[PHASE 3] H5 Noise vs Info Gap (KNN Consistency)...")
    # Take a sample for KNN
    idx_sample = np.random.choice(len(y_binary), 20000, replace=False)
    knn = KNeighborsClassifier(n_neighbors=10).fit(X_raw.values[idx_sample], y_binary[idx_sample])
    neigh_dist, neigh_idx = knn.kneighbors(X_raw.values[idx_sample])
    # Label variance among neighbors
    neigh_labels = y_binary[idx_sample][neigh_idx]
    label_std = np.std(neigh_labels, axis=1)
    noise_ratio = np.mean(label_std > 0.3) # Significant disagreement
    print(f"Estimated Noise Ratio (Inconsistent Neighbors): {noise_ratio:.4f}")

    # [PHASE 5: H7/H9 DYNAMIC THRESHOLDING]
    print("\n[PHASE 5] H7/H9 Dynamic Thresholding...")
    # Comparison
    base_prec_10 = get_prec_at_k(oof_p_base, y_binary, 10)
    # 1. Optimal Threshold Precision
    y_opt = (oof_p_base >= opt_thresh).astype(int)
    opt_prec = precision_score(y_binary, y_opt)
    print(f"Static Top-10% Prec: {base_prec_10:.4f}")
    print(f"Dynamic Optimal Prec: {opt_prec:.4f}")

    # [PHASE 6: H6 LOSS FUNCTION RE-EVALUATION]
    print("\n[PHASE 6] H6 Loss Function (Weighted BCE Intervention)...")
    oof_p_weighted = np.zeros(len(y_binary))
    for tr_idx, val_idx in gkf.split(X_raw, y_binary, groups=scenario_id):
        # HistGradientBoosting doesn't support class_weight directly easily in all versions, 
        # using sample_weight as proxy for Weighted BCE
        weights = np.where(y_binary[tr_idx] == 1, 10, 1)
        clf_w = HistGradientBoostingClassifier(max_iter=100, random_state=42)
        clf_w.fit(X_raw.values[tr_idx], y_binary[tr_idx], sample_weight=weights)
        oof_p_weighted[val_idx] = clf_w.predict_proba(X_raw.values[val_idx])[:, 1]
    
    prec_w, rec_w, _ = precision_recall_curve(y_binary, oof_p_weighted)
    print(f"Weighted PR AUC: {auc(rec_w, prec_w):.4f}")
    print(f"Weighted Prec@10%: {get_prec_at_k(oof_p_weighted, y_binary, 10):.4f}")

if __name__ == "__main__":
    run_integrated_causal_mission()
