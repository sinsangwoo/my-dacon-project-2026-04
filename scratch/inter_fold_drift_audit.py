import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def inter_fold_drift_audit():
    print(f"--- [INTER-FOLD ADVERSARIAL AUDIT] ---")
    
    scenario_id = np.load(f"{processed_dir}/scenario_id.npy", allow_pickle=True)
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    numeric_df = train_df.select_dtypes(include=[np.number])
    
    unique_scenarios = np.unique(scenario_id)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    auc_matrix = np.zeros((5, 5))
    
    # We will check AUC between Fold 0 and others as a proxy
    folds = list(kf.split(unique_scenarios))
    
    for i in range(5):
        for j in range(i + 1, 5):
            scen_i = unique_scenarios[folds[i][1]]
            scen_j = unique_scenarios[folds[j][1]]
            
            mask_i = np.isin(scenario_id, scen_i)
            mask_j = np.isin(scenario_id, scen_j)
            
            X_i = numeric_df[mask_i].values
            X_j = numeric_df[mask_j].values
            
            # Binary classification: Fold i vs Fold j
            X_combined = np.vstack([X_i, X_j])
            y_combined = np.array([0] * len(X_i) + [1] * len(X_j))
            
            # Fast RFC for adversarial check
            clf = RandomForestClassifier(n_estimators=50, max_depth=5, n_jobs=-1, random_state=42)
            # Sample for speed
            idx = np.random.permutation(len(X_combined))[:20000]
            clf.fit(X_combined[idx], y_combined[idx])
            
            auc = roc_auc_score(y_combined, clf.predict_proba(X_combined)[:, 1])
            print(f"Adversarial AUC [Fold {i} vs Fold {j}]: {auc:.4f}")
            auc_matrix[i, j] = auc

    print(f"\nMean Inter-Fold Adversarial AUC: {np.mean(auc_matrix[auc_matrix > 0]):.4f}")

if __name__ == "__main__":
    inter_fold_drift_audit()
