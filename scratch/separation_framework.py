import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def separation_framework_mission():
    print("--- [MISSION: SIGNAL-NOISE SEPARATION FRAMEWORK] ---")
    
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
    # 2 Seeds for stability check
    reg_tail1 = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=1).fit(X_tr[y_tr==1], y_gt_tr[y_tr==1])
    reg_tail2 = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=2).fit(X_tr[y_tr==1], y_gt_tr[y_tr==1])
    
    p_raw = clf.predict_proba(X_val)[:, 1]
    y_base = reg_base.predict(X_val)
    y_t1 = reg_tail1.predict(X_val)
    y_t2 = reg_tail2.predict(X_val)
    y_tail = (y_t1 + y_t2) / 2
    
    # [TASK 1: Signal Score Criteria]
    print("\n[TASK 1: SIGNAL SCORE CALCULATION]")
    gain_score = np.abs(y_gt_val - y_base) - np.abs(y_gt_val - y_tail)
    
    knn = NearestNeighbors(n_neighbors=5).fit(X_tr)
    _, indices = knn.kneighbors(X_val)
    local_consistency = np.mean(y_binary[tr_idx][indices], axis=1)
    
    residual_improvement = np.abs(y_gt_val - y_base) - np.abs(y_gt_val - y_tail)
    prediction_variance = np.var([y_t1, y_t2], axis=0)
    
    # Define SignalScore (Weighted sweep simulation)
    # Norm components to [0,1] for score consistency
    def norm(x): return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-9)
    signal_score = norm(local_consistency) + norm(residual_improvement) - norm(prediction_variance)
    
    results_t1 = []
    for q in [0.9, 0.8, 0.7]:
        threshold = np.percentile(signal_score, q*100)
        mask = signal_score >= threshold
        mae = mean_absolute_error(y_gt_val[mask], y_tail[mask])
        fpr = np.mean((y_gt_val[mask] < q90_val))
        results_t1.append({"Percentile": f"Top {(1-q)*100:.0f}%", "MAE": mae, "FP_R": fpr, "Mean_Score": np.mean(signal_score[mask])})
    print(pd.DataFrame(results_t1))

    # [TASK 2: Low-Confidence Sub-regime (p <= 0.4)]
    print("\n[TASK 2: SUB-REGIME CLUSTERING (p <= 0.4)]")
    low_p_mask = p_raw <= 0.4
    X_low = X_val[low_p_mask]
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X_low)
    clusters = kmeans.labels_
    
    cluster_results = []
    for c in range(3):
        c_mask = clusters == c
        c_gain = np.mean(gain_score[low_p_mask][c_mask])
        c_signal_ratio = np.mean(y_val[low_p_mask][c_mask])
        cluster_results.append({"Cluster": c, "Size": np.sum(c_mask), "Avg_Gain": c_gain, "Signal_Ratio": c_signal_ratio})
    print(pd.DataFrame(cluster_results))

    # [TASK 5: FINAL COMPARISON]
    print("\n[TASK 5: BASELINE vs SEPARATION]")
    # Conditional Recovery: Only recover if Cluster Gain is high or SignalScore is high
    best_cluster = pd.DataFrame(cluster_results).sort_values("Avg_Gain", ascending=False).iloc[0]["Cluster"]
    best_c_mask = np.zeros_like(p_raw, dtype=bool)
    best_c_mask[low_p_mask] = clusters == best_cluster
    
    # Separation + Conditional Recovery (Only Top 10% SignalScore)
    safe_mask = signal_score >= np.percentile(signal_score, 90)
    y_cond = y_base.copy()
    y_cond[safe_mask] = y_tail[safe_mask]
    
    final_res = [
        {"Method": "No-tail", "MAE": mean_absolute_error(y_gt_val, y_base)},
        {"Method": "Baseline (Full Tail)", "MAE": mean_absolute_error(y_gt_val, y_tail)},
        {"Method": "Sep + Cond Recovery", "MAE": mean_absolute_error(y_gt_val, y_cond)}
    ]
    print(pd.DataFrame(final_res))

if __name__ == "__main__":
    separation_framework_mission()
