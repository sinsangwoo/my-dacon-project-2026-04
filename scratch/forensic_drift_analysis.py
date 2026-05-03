import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

RUN_ID = "run_20260429_115949"
PROCESSED_PATH = f"./outputs/{RUN_ID}/processed"
DRIFT_AUDIT_PATH = f"./logs/{RUN_ID}/summary/distribution/drift_audit_raw.csv"

def run_forensic():
    drift_df = pd.read_csv(DRIFT_AUDIT_PATH)
    train_base = pd.read_pickle(f"{PROCESSED_PATH}/train_base.pkl")
    test_base = pd.read_pickle(f"{PROCESSED_PATH}/test_base.pkl")
    s_tr = train_base.sample(min(20000, len(train_base)), random_state=42)
    s_te = test_base.sample(min(20000, len(test_base)), random_state=42)
    common_cols = [c for c in s_tr.columns if c in s_te.columns and c not in ['ID', 'scenario_id', 'layout_id', 'target']]
    common_cols = [c for c in common_cols if pd.api.types.is_numeric_dtype(s_tr[c])]
    X_tr = s_tr[common_cols].fillna(-999).values
    X_te = s_te[common_cols].fillna(-999).values
    X = np.vstack([X_tr, X_te])
    y = np.hstack([np.zeros(len(X_tr)), np.ones(len(X_te))])
    adv_clf = LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, verbose=-1)
    adv_clf.fit(X, y)
    imp_df = pd.DataFrame({'feature': common_cols, 'importance': adv_clf.feature_importances_})
    merged = pd.merge(drift_df, imp_df, on='feature', how='inner')
    from src.data_loader import get_protected_candidates
    protected = get_protected_candidates(merged['feature'].tolist())
    merged['is_protected'] = merged['feature'].isin(protected)
    
    with open("scratch/forensic_drift_report.txt", "w") as f:
        f.write(f"ADV AUC on sample: {roc_auc_score(y, adv_clf.predict_proba(X)[:, 1]):.4f}\n")
        f.write("\nTOP 30 ADVERSARIAL IMPORTANCE\n")
        f.write(merged.sort_values(by='importance', ascending=False).head(30).to_string(index=False))
        f.write("\n\nTOP 30 KS STAT\n")
        f.write(merged.sort_values(by='ks_stat', ascending=False).head(30).to_string(index=False))
        f.write("\n\nTOP 30 PROTECTED DRIFT\n")
        f.write(merged[merged['is_protected']].sort_values(by='ks_stat', ascending=False).head(30).to_string(index=False))
        
if __name__ == "__main__":
    run_forensic()
