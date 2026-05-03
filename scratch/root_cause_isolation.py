import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def root_cause_isolation():
    print("--- [MISSION 6: ROOT CAUSE ISOLATION] ---")
    
    # 1. Load Data
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    scenario_id = np.load(f"{processed_dir}/scenario_id.npy", allow_pickle=True)
    
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    X_raw = train_df.select_dtypes(include=[np.number]).fillna(0)
    X_raw = X_raw.drop(columns=['avg_delay_minutes_next_30m', 'ID', 'scenario_id', 'layout_id'], errors='ignore')
    
    # [EXP 1: Label Definition vs Separability]
    print("\n[TABLE 1: LABEL THRESHOLD vs AUC]")
    results_label = []
    # Sample for speed
    idx_s = np.random.choice(len(y_true), 50000, replace=False)
    X_s = X_raw.values[idx_s]
    y_s_true = y_true[idx_s]
    
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr_all, y_val_all = train_test_split(X_s, y_s_true, test_size=0.3, random_state=42)
    
    for q in [50, 70, 90, 95, 99]:
        q_val = np.percentile(y_tr_all, q)
        y_tr_q = (y_tr_all >= q_val).astype(int)
        y_val_q = (y_val_all >= q_val).astype(int)
        
        if len(np.unique(y_tr_q)) > 1:
            clf = RandomForestClassifier(n_estimators=30, max_depth=8, n_jobs=-1, random_state=42).fit(X_tr, y_tr_q)
            auc_q = roc_auc_score(y_val_q, clf.predict_proba(X_val)[:, 1])
            results_label.append({"Threshold": f"Q{q}", "ROC_AUC": auc_q})
    print(pd.DataFrame(results_label))

    # [EXP 2: Mutual Information Analysis]
    print("\n[TABLE 2: TOP FEATURE MUTUAL INFORMATION]")
    # Use Q90 as current target
    y_q90 = (y_tr_all >= np.percentile(y_tr_all, 90)).astype(int)
    # MI on top 20 features to see raw information gain
    clf_temp = RandomForestClassifier(n_estimators=30, random_state=42).fit(X_tr, y_q90)
    top_20_idx = np.argsort(clf_temp.feature_importances_)[-20:]
    
    mi_scores = mutual_info_classif(X_tr[:, top_20_idx], y_q90, discrete_features=False)
    print(f"Mean MI (Top 20): {np.mean(mi_scores):.6f}")
    print(f"Max MI:           {np.max(mi_scores):.6f}")

    # [EXP 3: Capacity Test]
    print("\n[TABLE 3: MODEL CAPACITY vs PRECISION]")
    from sklearn.linear_model import LogisticRegression
    clf_lin = LogisticRegression(max_iter=1000).fit(X_tr, y_q90)
    clf_deep = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1).fit(X_tr, y_q90)
    
    def get_p10(clf, X, y):
        p = clf.predict_proba(X)[:, 1]
        return roc_auc_score(y, p)

    print(f"Linear Model AUC: {get_p10(clf_lin, X_val, (y_val_all >= np.percentile(y_val_all, 90)).astype(int)):.4f}")
    print(f"Deep Model AUC:   {get_p10(clf_deep, X_val, (y_val_all >= np.percentile(y_val_all, 90)).astype(int)):.4f}")

if __name__ == "__main__":
    root_cause_isolation()
