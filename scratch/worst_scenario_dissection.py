import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def worst_scenario_dissection():
    print("--- [MISSION 4: WORST SCENARIO FAILURE DISSECTION] ---")
    
    # Load Data
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    scenario_id = np.load(f"{processed_dir}/scenario_id.npy", allow_pickle=True)
    q90_val = np.percentile(y_true, 90)
    y_binary = (y_true >= q90_val).astype(int)
    
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    X_raw = train_df.select_dtypes(include=[np.number]).fillna(0)
    X_raw = X_raw.drop(columns=['avg_delay_minutes_next_30m', 'ID', 'scenario_id', 'layout_id'], errors='ignore')
    
    # 1. Identify Worst vs Best Scenarios (Using previous logic proxy)
    # Re-run a simple classifier to get AUC per scenario
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    tr_idx, val_idx = next(gkf.split(X_raw, y_binary, groups=scenario_id))
    X_tr, X_val = X_raw.values[tr_idx], X_raw.values[val_idx]
    y_tr, y_val = y_binary[tr_idx], y_binary[val_idx]
    sid_val = scenario_id[val_idx]
    
    clf = RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42).fit(X_tr, y_tr)
    p_val = clf.predict_proba(X_val)[:, 1]
    
    scen_results = []
    for sid in np.unique(sid_val):
        mask = sid_val == sid
        if np.sum(y_val[mask]) > 0 and len(np.unique(y_val[mask])) > 1:
            auc = roc_auc_score(y_val[mask], p_val[mask])
            scen_results.append({"sid": sid, "auc": auc, "X": X_val[mask], "y": y_val[mask]})
    
    df_scen = pd.DataFrame(scen_results).sort_values("auc")
    worst_sid = df_scen.iloc[0]['sid']
    best_sid = df_scen.iloc[-1]['sid']
    
    X_worst = df_scen.iloc[0]['X']
    y_worst = df_scen.iloc[0]['y']
    X_best = df_scen.iloc[-1]['X']
    y_best = df_scen.iloc[-1]['y']
    
    # [EXP 1: Cross-Scenario Feature Distance]
    dist_matrix = cdist(X_worst, X_best, metric='euclidean')
    min_dist = np.mean(np.min(dist_matrix, axis=1))
    # Reference: Internal distance within Best
    internal_dist_best = np.mean(cdist(X_best, X_best, metric='euclidean'))
    
    print("\n[TABLE 1: FEATURE SPACE PROXIMITY]")
    print(f"Worst Scenario ID: {worst_sid} (AUC: {df_scen.iloc[0]['auc']:.4f})")
    print(f"Best Scenario ID:  {best_sid} (AUC: {df_scen.iloc[-1]['auc']:.4f})")
    print(f"Mean Distance (Worst to Best): {min_dist:.4f}")
    print(f"Internal Distance (Best):       {internal_dist_best:.4f}")

    # [EXP 2: Internal Consistency]
    def get_consistency(X, y):
        knn = NearestNeighbors(n_neighbors=5).fit(X)
        _, indices = knn.kneighbors(X)
        neigh_y = y[indices]
        return np.mean(np.std(neigh_y, axis=1) < 0.1)

    print("\n[TABLE 2: INTERNAL CONSISTENCY]")
    print(f"Worst Scenario Consistency: {get_consistency(X_worst, y_worst):.4f}")
    print(f"Best Scenario Consistency:  {get_consistency(X_best, y_best):.4f}")

    # [EXP 3: Target Variance]
    print("\n[TABLE 3: TARGET VARIANCE]")
    print(f"Worst Scenario Target Var: {np.var(y_worst):.4f}")
    print(f"Best Scenario Target Var:  {np.var(y_best):.4f}")

if __name__ == "__main__":
    worst_scenario_dissection()
