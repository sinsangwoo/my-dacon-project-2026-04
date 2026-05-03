import numpy as np
import pandas as pd
import os
import joblib
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, auc, roc_auc_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from scipy.spatial.distance import cdist

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def run_full_spectrum_causal_mission():
    print("--- [MISSION: FULL-SPECTRUM CAUSAL ENUMERATION] ---")
    
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
    
    def evaluate(X, y, weight=None):
        oof_p = np.zeros(len(y))
        for tr_idx, val_idx in gkf.split(X, y, groups=scenario_id):
            clf = RandomForestClassifier(n_estimators=30, max_depth=8, n_jobs=-1, random_state=42)
            clf.fit(X[tr_idx], y[tr_idx], sample_weight=weight[tr_idx] if weight is not None else None)
            oof_p[val_idx] = clf.predict_proba(X[val_idx])[:, 1]
        
        p, r, t = precision_recall_curve(y, oof_p)
        pr_auc = auc(r, p)
        thresh = np.percentile(oof_p, 90)
        precision = precision_score(y, (oof_p >= thresh).astype(int))
        
        mask_tail = (oof_p >= thresh)
        if np.sum(mask_tail) > 0:
            reg = RandomForestRegressor(n_estimators=20, max_depth=8, n_jobs=-1, random_state=42).fit(X[y==1], y_true[y==1])
            mae = mean_absolute_error(y_true[mask_tail], reg.predict(X[mask_tail]))
        else:
            mae = 0
        return precision, pr_auc, mae, oof_p

    print("Step: Baseline")
    b_prec, b_pr_auc, b_mae, b_p = evaluate(X_raw.values, y_binary)
    
    impacts = []
    
    # C1: PCA 30
    X_pca = PCA(n_components=30).fit_transform(X_raw)
    c1_p, c1_a, c1_m, _ = evaluate(X_pca, y_binary)
    impacts.append({"Cause": "C1 (Distortion)", "D_Prec": c1_p - b_prec, "D_PR_AUC": c1_a - b_pr_auc, "D_MAE": c1_m - b_mae})

    # C2: Top 50
    clf_imp = RandomForestClassifier(n_estimators=30, max_depth=8, n_jobs=-1, random_state=42).fit(X_raw.values[:50000], y_binary[:50000])
    top_50 = np.argsort(clf_imp.feature_importances_)[-50:]
    c2_p, c2_a, c2_m, _ = evaluate(X_raw.values[:, top_50], y_binary)
    impacts.append({"Cause": "C2 (Dilution)", "D_Prec": c2_p - b_prec, "D_PR_AUC": c2_a - b_pr_auc, "D_MAE": c2_m - b_mae})

    # C3: Interactions
    top_5 = top_50[-5:]
    X_h3 = X_raw.values[:, top_5].copy()
    import itertools
    for i, j in itertools.combinations(range(5), 2):
        X_h3 = np.hstack([X_h3, (X_h3[:, i] * X_h3[:, j]).reshape(-1, 1)])
    c3_p, c3_a, c3_m, _ = evaluate(X_h3, y_binary)
    impacts.append({"Cause": "C3 (Interaction)", "D_Prec": c3_p - b_prec, "D_PR_AUC": c3_a - b_pr_auc, "D_MAE": c3_m - b_mae})

    # C4: Temporal
    X_h4 = X_raw.values[:, top_5].copy()
    for i in range(5):
        lag = pd.Series(X_h4[:, i]).shift(1).fillna(0).values.reshape(-1, 1)
        X_h4 = np.hstack([X_h4, lag])
    c4_p, c4_a, c4_m, _ = evaluate(X_h4, y_binary)
    impacts.append({"Cause": "C4 (Temporal)", "D_Prec": c4_p - b_prec, "D_PR_AUC": c4_a - b_pr_auc, "D_MAE": c4_m - b_mae})

    # C5: Boundary Q85
    y_h5 = (y_true >= np.percentile(y_true, 85)).astype(int)
    c5_p, c5_a, c5_m, _ = evaluate(X_raw.values, y_h5)
    impacts.append({"Cause": "C5 (Boundary)", "D_Prec": c5_p - b_prec, "D_PR_AUC": c5_a - b_pr_auc, "D_MAE": c5_m - b_mae})

    # C7: Imbalance 1:10
    w = np.where(y_binary == 1, 10, 1)
    c7_p, c7_a, c7_m, _ = evaluate(X_raw.values, y_binary, weight=w)
    impacts.append({"Cause": "C7 (Imbalance)", "D_Prec": c7_p - b_prec, "D_PR_AUC": c7_a - b_pr_auc, "D_MAE": c7_m - b_mae})

    # C8: Optimal Threshold
    pl, rl, _ = precision_recall_curve(y_binary, b_p)
    opt_p = np.max(pl[rl > 0.01])
    impacts.append({"Cause": "C8 (Threshold)", "D_Prec": opt_p - b_prec, "D_PR_AUC": 0, "D_MAE": 0})

    # C10: Metric (Normalization as proxy for Cosine)
    X_norm = X_raw.values / (np.linalg.norm(X_raw.values, axis=1, keepdims=True) + 1e-9)
    c10_p, c10_a, c10_m, _ = evaluate(X_norm, y_binary)
    impacts.append({"Cause": "C10 (Metric)", "D_Prec": c10_p - b_prec, "D_PR_AUC": c10_a - b_pr_auc, "D_MAE": c10_m - b_mae})

    # Phase 3: Curve
    subset = []
    for count in [50, 100, 150, 214]:
        f_idx = top_50[:count] if count < 214 else range(X_raw.shape[1])
        p, a, _, _ = evaluate(X_raw.values[:, f_idx], y_binary)
        subset.append({"Count": count, "Precision": p, "PR AUC": a})
    
    print("\n--- [RESULTS] ---")
    print(pd.DataFrame(impacts))
    print("\n--- [FEATURE SUBSET CURVE] ---")
    print(pd.DataFrame(subset))

if __name__ == "__main__":
    run_full_spectrum_causal_mission()
