import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def regime_conflict_analysis_optimized():
    print("--- [MISSION 5: REGIME SHIFT & CONFLICT ANALYSIS (Optimized)] ---")
    
    # 1. Load Data
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    scenario_id = np.load(f"{processed_dir}/scenario_id.npy", allow_pickle=True)
    q90_val = np.percentile(y_true, 90)
    y_binary = (y_true >= q90_val).astype(int)
    
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    X_raw = train_df.select_dtypes(include=[np.number]).fillna(0)
    X_raw = X_raw.drop(columns=['avg_delay_minutes_next_30m', 'ID', 'scenario_id', 'layout_id'], errors='ignore')
    
    # Sample 100 Scenarios
    unique_sids = np.unique(scenario_id)
    sampled_sids = np.random.choice(unique_sids, 100, replace=False)
    
    # [EXP 1: Local vs Global AUC Gap]
    print("\n[TABLE 1: LOCAL vs GLOBAL AUC GAP (N=10 Sample)]")
    results_gap = []
    
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    tr_idx, val_idx = next(gkf.split(X_raw, y_binary, groups=scenario_id))
    global_clf = RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42).fit(X_raw.values[tr_idx], y_binary[tr_idx])
    
    for sid in sampled_sids[:10]: 
        mask_s = scenario_id == sid
        if np.sum(y_binary[mask_s]) > 2 and len(np.unique(y_binary[mask_s])) > 1:
            X_s, y_s = X_raw.values[mask_s], y_binary[mask_s]
            p_global = global_clf.predict_proba(X_s)[:, 1]
            auc_global = roc_auc_score(y_s, p_global)
            
            # Local (Internal Split)
            from sklearn.model_selection import train_test_split
            try:
                X_s_tr, X_s_val, y_s_tr, y_s_val = train_test_split(X_s, y_s, test_size=0.3, random_state=42)
                if len(np.unique(y_s_tr)) > 1 and len(np.unique(y_s_val)) > 1:
                    local_clf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42).fit(X_s_tr, y_s_tr)
                    auc_local = roc_auc_score(y_s_val, local_clf.predict_proba(X_s_val)[:, 1])
                    auc_global_small = roc_auc_score(y_s_val, global_clf.predict_proba(X_s_val)[:, 1])
                    results_gap.append({"sid": sid, "Local": auc_local, "Global": auc_global_small, "Gap": auc_local - auc_global_small})
            except:
                continue
    
    print(pd.DataFrame(results_gap))

    # [EXP 2: Directional Conflict]
    print("\n[TABLE 2: DIRECTIONAL CONFLICT (N=100 Scenarios)]")
    top_10_feats = np.argsort(global_clf.feature_importances_)[-10:]
    conflict_counts = []
    
    for f_idx in top_10_feats:
        rhos = []
        for sid in sampled_sids:
            mask = scenario_id == sid
            if np.std(X_raw.values[mask, f_idx]) > 0 and np.std(y_binary[mask]) > 0:
                rho, _ = spearmanr(X_raw.values[mask, f_idx], y_binary[mask])
                if not np.isnan(rho): rhos.append(rho)
        
        rhos = np.array(rhos)
        pos_ratio = np.mean(rhos > 0.2)
        neg_ratio = np.mean(rhos < -0.2)
        neutral = 1.0 - pos_ratio - neg_ratio
        conflict_counts.append({"F_Idx": f_idx, "Pos_Ratio": pos_ratio, "Neg_Ratio": neg_ratio, "Conflict_Index": min(pos_ratio, neg_ratio) * 2})
    
    print(pd.DataFrame(conflict_counts))

if __name__ == "__main__":
    regime_conflict_analysis_optimized()
