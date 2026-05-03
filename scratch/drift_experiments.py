import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

RUN_ID = "run_20260429_115949"
PROCESSED_PATH = f"./outputs/{RUN_ID}/processed"

def get_auc(X_tr, X_te, name):
    X = np.vstack([X_tr, X_te])
    y = np.hstack([np.zeros(len(X_tr)), np.ones(len(X_te))])
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, val_idx in skf.split(X, y):
        clf = LGBMClassifier(n_estimators=100, max_depth=3, random_state=42, verbose=-1)
        clf.fit(X[tr_idx], y[tr_idx])
        aucs.append(roc_auc_score(y[val_idx], clf.predict_proba(X[val_idx])[:, 1]))
    auc = np.mean(aucs)
    print(f"[{name}] ADV AUC: {auc:.4f}")
    return auc

def run_experiments():
    train_base = pd.read_pickle(f"{PROCESSED_PATH}/train_base.pkl")
    test_base = pd.read_pickle(f"{PROCESSED_PATH}/test_base.pkl")
    s_tr = train_base.sample(min(10000, len(train_base)), random_state=42)
    s_te = test_base.sample(min(10000, len(test_base)), random_state=42)
    
    common_cols = [c for c in s_tr.columns if c in s_te.columns and c not in ['ID', 'scenario_id', 'layout_id', 'target']]
    common_cols = [c for c in common_cols if pd.api.types.is_numeric_dtype(s_tr[c])]
    
    # Test PCA embeddings
    pca_cols = [c for c in common_cols if 'embed_' in c]
    if pca_cols:
        X_tr_pca = s_tr[pca_cols].fillna(-999).values
        X_te_pca = s_te[pca_cols].fillna(-999).values
        get_auc(X_tr_pca, X_te_pca, "H4: PCA EMBEDDINGS")
        
    # Test Clusters
    cluster_cols = [c for c in common_cols if 'regime_proxy' in c]
    if cluster_cols:
        X_tr_cluster = s_tr[cluster_cols].fillna(-999).values
        X_te_cluster = s_te[cluster_cols].fillna(-999).values
        get_auc(X_tr_cluster, X_te_cluster, "H4: CLUSTERS (REGIME)")

    # Test Time-Series features ONLY
    ts_suffixes = ['_rolling_mean_', '_rolling_std_', '_slope_', '_rate_', '_diff_']
    ts_cols = [c for c in common_cols if any(s in c for s in ts_suffixes)]
    if ts_cols:
        X_tr_ts = s_tr[ts_cols].fillna(-999).values
        X_te_ts = s_te[ts_cols].fillna(-999).values
        get_auc(X_tr_ts, X_te_ts, "TIME-SERIES DERIVATIVES ONLY")

if __name__ == "__main__":
    run_experiments()
