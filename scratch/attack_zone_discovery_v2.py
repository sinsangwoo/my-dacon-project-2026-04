import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def attack_zone_transformation_v2():
    print("--- [MISSION: CLUSTER 1 -> TRUE ATTACK ZONE (V2)] ---")
    
    # 1. Load Data
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    q90_val = np.percentile(y_true, 90)
    y_binary = (y_true >= q90_val).astype(int)
    
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    X = train_df.select_dtypes(include=[np.number]).fillna(0)
    X = X.drop(columns=['avg_delay_minutes_next_30m', 'ID', 'scenario_id', 'layout_id'], errors='ignore').values
    
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    tr_idx, val_idx = next(kf.split(X))
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y_binary[tr_idx], y_binary[val_idx]
    y_gt_tr, y_gt_val = y_true[tr_idx], y_true[val_idx]
    
    # Models
    clf = RandomForestClassifier(n_estimators=30, max_depth=8, n_jobs=-1, random_state=42).fit(X_tr, y_tr)
    reg_base = RandomForestRegressor(n_estimators=30, max_depth=8, n_jobs=-1).fit(X_tr[y_tr==0], y_gt_tr[y_tr==0])
    reg_tail1 = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=1).fit(X_tr[y_tr==1], y_gt_tr[y_tr==1])
    reg_tail2 = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=2).fit(X_tr[y_tr==1], y_gt_tr[y_tr==1])
    
    p_raw = clf.predict_proba(X_val)[:, 1]
    y_base = reg_base.predict(X_val)
    y_t1 = reg_tail1.predict(X_val)
    y_t2 = reg_tail2.predict(X_val)
    y_tail = (y_t1 + y_t2) / 2
    
    # Identify Cluster 1 Corrected
    low_p_mask = p_raw <= 0.4
    X_low = X_val[low_p_mask]
    kmeans_initial = KMeans(n_clusters=3, random_state=42).fit(X_low)
    cluster_initial = kmeans_initial.labels_
    
    # Create a full-sized mask for Cluster 1
    c1_full_mask = np.zeros_like(p_raw, dtype=bool)
    c1_full_mask[low_p_mask] = cluster_initial == 1
    c1_indices = np.where(c1_full_mask)[0]
    
    X_c1 = X_val[c1_indices]
    y_gt_c1 = y_gt_val[c1_indices]
    
    # [TASK 1: Sub-decomposition]
    sub_kmeans = KMeans(n_clusters=2, random_state=42).fit(X_c1)
    sub_clusters = sub_kmeans.labels_
    sub_results = []
    for sc in range(2):
        sc_mask = sub_clusters == sc
        sc_idx = c1_indices[sc_mask]
        gain = np.abs(y_gt_val[sc_idx] - y_base[sc_idx]) - np.abs(y_gt_val[sc_idx] - y_tail[sc_idx])
        sub_results.append({"Sub-Cluster": sc, "Size": np.sum(sc_mask), "MAE": mean_absolute_error(y_gt_val[sc_idx], y_tail[sc_idx]), "Avg_Gain": np.mean(gain), "Noise_Ratio": 1 - np.mean(y_val[sc_idx])})
    print("\n[TASK 1: CLUSTER 1 SUB-DECOMPOSITION]")
    print(pd.DataFrame(sub_results))

    # [TASK 2: Noise Kill] & [TASK 3: EV Filter]
    pred_var_c1 = np.var([y_t1[c1_indices], y_t2[c1_indices]], axis=0)
    kill_mask_var = pred_var_c1 > np.percentile(pred_var_c1, 90)
    
    s_gain = np.where(y_val[c1_indices] == 1, np.abs(y_gt_c1 - y_base[c1_indices]) - np.abs(y_gt_c1 - y_tail[c1_indices]), 0)
    s_cost = np.where(y_val[c1_indices] == 0, np.abs(y_tail[c1_indices] - y_gt_c1) - np.abs(y_base[c1_indices] - y_gt_c1), 0)
    ev = p_raw[c1_indices] * s_gain - (1 - p_raw[c1_indices]) * s_cost
    ev_mask = ev > 0
    
    # [TASK 4: Final Attack Zone Definition]
    final_indices = c1_indices[ev_mask & (~kill_mask_var)]
    final_zone_metrics = {
        "Size": len(final_indices),
        "MAE_Base": mean_absolute_error(y_gt_val[final_indices], y_base[final_indices]),
        "MAE_Tail": mean_absolute_error(y_gt_val[final_indices], y_tail[final_indices]),
        "FP_R": np.mean(y_val[final_indices] == 0),
        "MAE_Improvement": mean_absolute_error(y_gt_val[final_indices], y_base[final_indices]) - mean_absolute_error(y_gt_val[final_indices], y_tail[final_indices])
    }
    print("\n[TASK 4: FINAL ATTACK ZONE DEFINITION]")
    print(pd.DataFrame([final_zone_metrics]))

    # [TASK 5: CONTROL EXPERIMENT]
    y_final = y_base.copy()
    y_final[final_indices] = y_tail[final_indices]
    
    y_c1_full = y_base.copy()
    y_c1_full[c1_full_mask] = y_tail[c1_full_mask]
    
    final_comp = [
        {"Method": "No Tail", "MAE": mean_absolute_error(y_gt_val, y_base)},
        {"Method": "Cluster 1 (Full)", "MAE": mean_absolute_error(y_gt_val, y_c1_full)},
        {"Method": "Final Attack Zone", "MAE": mean_absolute_error(y_gt_val, y_final)}
    ]
    print("\n[TASK 5: CONTROL EXPERIMENT]")
    print(pd.DataFrame(final_comp))

if __name__ == "__main__":
    attack_zone_transformation_v2()
