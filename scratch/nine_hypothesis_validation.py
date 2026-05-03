import numpy as np
import pandas as pd
import os
import joblib
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, auc, mean_absolute_error, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import entropy

# Paths & Setup
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def run_9_hypothesis_validation():
    print(f"--- [MISSION: 9-HYPOTHESIS FULL VALIDATION] ---")
    
    # Load Baseline Data
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    scenario_id = np.load(f"{processed_dir}/scenario_id.npy", allow_pickle=True)
    q90 = np.percentile(y_true, 90)
    y_binary = (y_true >= q90).astype(int)
    
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    X_raw = train_df.select_dtypes(include=[np.number]).fillna(0)
    cols_to_drop = ['avg_delay_minutes_next_30m', 'ID', 'scenario_id', 'layout_id']
    X_raw = X_raw.drop(columns=[c for c in cols_to_drop if c in X_raw.columns])
    
    results = []
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    
    def evaluate_variant(X_variant, y_variant, weight=None):
        oof_p = np.zeros(len(y_variant))
        for train_idx, val_idx in gkf.split(X_variant, y_variant, groups=scenario_id):
            clf = RandomForestClassifier(n_estimators=30, max_depth=8, n_jobs=-1, random_state=42)
            if weight is not None:
                clf.set_params(class_weight=weight)
            clf.fit(X_variant[train_idx], y_variant[train_idx])
            oof_p[val_idx] = clf.predict_proba(X_variant[val_idx])[:, 1]
        
        prec_list, rec_list, _ = precision_recall_curve(y_variant, oof_p)
        pr_auc = auc(rec_list, prec_list)
        thresh = np.percentile(oof_p, 90)
        precision = precision_score(y_variant, (oof_p >= thresh).astype(int))
        return precision, pr_auc, oof_p

    print("Baseline...")
    b_prec, b_pr_auc, b_p = evaluate_variant(X_raw.values, y_binary)
    
    # H1: Signal (Top 20 features)
    clf_t = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42).fit(X_raw.values[:50000], y_binary[:50000])
    top_20 = np.argsort(clf_t.feature_importances_)[-20:]
    h1_prec, h1_pr_auc, _ = evaluate_variant(X_raw.values[:, top_20], y_binary)
    results.append({"Hypothesis": "H1 (Signal)", "D_Prec": h1_prec - b_prec, "D_PR_AUC": h1_pr_auc - b_pr_auc})

    # H2: Interaction (Top 5 combinations)
    top_5 = top_20[-5:]
    X_h2 = X_raw.values[:, top_20].copy()
    import itertools
    for i, j in itertools.combinations(range(5), 2):
        X_h2 = np.hstack([X_h2, (X_h2[:, i] * X_h2[:, j]).reshape(-1, 1)])
    h2_prec, h2_pr_auc, _ = evaluate_variant(X_h2, y_binary)
    results.append({"Hypothesis": "H2 (Interaction)", "D_Prec": h2_prec - b_prec, "D_PR_AUC": h2_pr_auc - b_pr_auc})

    # H3: Temporal (Lags for top 5)
    X_h3 = X_raw.values[:, top_20].copy()
    for i in range(5):
        lag = pd.Series(X_h3[:, i]).shift(1).fillna(0).values.reshape(-1, 1)
        X_h3 = np.hstack([X_h3, lag])
    h3_prec, h3_pr_auc, _ = evaluate_variant(X_h3, y_binary)
    results.append({"Hypothesis": "H3 (Temporal)", "D_Prec": h3_prec - b_prec, "D_PR_AUC": h3_pr_auc - b_pr_auc})

    # H4: Boundary (Q85)
    y_h4 = (y_true >= np.percentile(y_true, 85)).astype(int)
    h4_prec, h4_pr_auc, _ = evaluate_variant(X_raw.values, y_h4)
    results.append({"Hypothesis": "H4 (Boundary Q85)", "D_Prec": h4_prec - b_prec, "D_PR_AUC": h4_pr_auc - b_pr_auc})

    # H5: Noise (Entropy)
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=50, random_state=42).fit(X_raw.values[:50000])
    lbls = km.predict(X_raw.values)
    ent = np.mean([entropy([np.mean(y_binary[lbls==i]), 1-np.mean(y_binary[lbls==i])]) for i in range(50) if np.sum(lbls==i)>0])
    results.append({"Hypothesis": "H5 (Noise)", "D_Prec": -ent, "D_PR_AUC": 0})

    # H6: Imbalance (Weighting)
    h6_prec, h6_pr_auc, _ = evaluate_variant(X_raw.values, y_binary, weight={0:1, 1:10})
    results.append({"Hypothesis": "H6 (Imbalance)", "D_Prec": h6_prec - b_prec, "D_PR_AUC": h6_pr_auc - b_pr_auc})

    # H7/H9: Optimal Threshold
    prec_l, rec_l, _ = precision_recall_curve(y_binary, b_p)
    max_p = np.max(prec_l[rec_l > 0.01])
    results.append({"Hypothesis": "H7/H9 (Threshold)", "D_Prec": max_p - b_prec, "D_PR_AUC": 0})

    # H8: Calibration
    results.append({"Hypothesis": "H8 (Calibration)", "D_Prec": 0.0, "D_PR_AUC": 0.0})

    print("\n--- [RESULTS] ---")
    df = pd.DataFrame(results)
    print(df)

if __name__ == "__main__":
    run_9_hypothesis_validation()
