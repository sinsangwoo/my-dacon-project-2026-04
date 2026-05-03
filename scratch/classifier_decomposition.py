import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def classifier_decomposition():
    print("--- [MISSION 2: CLASSIFIER FAILURE DECOMPOSITION] ---")
    
    # Load Data
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    scenario_id = np.load(f"{processed_dir}/scenario_id.npy", allow_pickle=True)
    q90_val = np.percentile(y_true, 90)
    y_binary = (y_true >= q90_val).astype(int)
    
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    X_raw = train_df.select_dtypes(include=[np.number]).fillna(0)
    X_raw = X_raw.drop(columns=['avg_delay_minutes_next_30m', 'ID', 'scenario_id', 'layout_id'], errors='ignore')
    
    gkf = GroupKFold(n_splits=5)
    tr_idx, val_idx = next(gkf.split(X_raw, y_binary, groups=scenario_id))
    X_tr, X_val = X_raw.values[tr_idx], X_raw.values[val_idx]
    y_tr, y_val = y_binary[tr_idx], y_binary[val_idx]

    # [EXPERIMENT 1: Feature-only separability]
    print("\n[EXP 1: MODEL COMPARISON]")
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=100, n_jobs=-1, random_state=42, verbose=-1),
        "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=10)
    }
    
    results_m = []
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        p_val = m.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, p_val)
        prec_list, rec_list, _ = precision_recall_curve(y_val, p_val)
        pr_auc = auc(rec_list, prec_list)
        thresh = np.percentile(p_val, 90)
        p10 = precision_score(y_val, (p_val >= thresh).astype(int))
        results_res = {"Model": name, "ROC_AUC": roc_auc, "PR_AUC": pr_auc, "P@10": p10}
        results_m.append(results_res)
    print(pd.DataFrame(results_m))

    # [EXPERIMENT 2: Feature subset sensitivity]
    print("\n[EXP 2: FEATURE SUBSET]")
    clf_imp = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42).fit(X_tr, y_tr)
    importances = clf_imp.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    results_f = []
    for k in [20, 50, 100, 150, len(importances)]:
        X_tr_k = X_tr[:, sorted_idx[:k]]
        X_val_k = X_val[:, sorted_idx[:k]]
        clf_k = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42).fit(X_tr_k, y_tr)
        p_k = clf_k.predict_proba(X_val_k)[:, 1]
        prec_list, rec_list, _ = precision_recall_curve(y_val, p_k)
        pr_auc = auc(rec_list, prec_list)
        thresh = np.percentile(p_k, 90)
        p10 = precision_score(y_val, (p_k >= thresh).astype(int))
        results_f.append({"K": k, "PR_AUC": pr_auc, "P@10": p10})
    print(pd.DataFrame(results_f))

    # [EXPERIMENT 3: Label ambiguity probe]
    print("\n[EXP 3: BOUNDARY REGION (Q85-Q95)]")
    q85 = np.percentile(y_true, 85)
    q95 = np.percentile(y_true, 95)
    mask_boundary = (y_true[val_idx] >= q85) & (y_true[val_idx] <= q95)
    # Re-use best model (RF) for consistency
    clf_b = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42).fit(X_tr, y_tr)
    p_b = clf_b.predict_proba(X_val)[:, 1]
    p10_overall = precision_score(y_val, (p_b >= np.percentile(p_b, 90)).astype(int))
    p10_boundary = precision_score(y_val[mask_boundary], (p_b[mask_boundary] >= np.percentile(p_b, 90)).astype(int))
    print(f"Overall P@10: {p10_overall:.4f}")
    print(f"Boundary P@10: {p10_boundary:.4f}")

    # [EXPERIMENT 4: Scenario-wise separability]
    print("\n[EXP 4: SCENARIO VARIANCE]")
    results_s = []
    unique_scen = np.unique(scenario_id[val_idx])
    for s in unique_scen:
        mask_s = scenario_id[val_idx] == s
        if np.sum(y_val[mask_s]) > 0:
            scen_p = p_b[mask_s]
            scen_y = y_val[mask_s]
            roc_auc_s = roc_auc_score(scen_y, scen_p) if len(np.unique(scen_y)) > 1 else 0.5
            results_s.append({"Scenario": s, "ROC_AUC": roc_auc_s})
    
    df_s = pd.DataFrame(results_s)
    print(f"Best Scenario AUC: {df_s['ROC_AUC'].max():.4f}")
    print(f"Worst Scenario AUC: {df_s['ROC_AUC'].min():.4f}")
    print(f"Scenario Variance: {df_s['ROC_AUC'].var():.4f}")

if __name__ == "__main__":
    classifier_decomposition()
