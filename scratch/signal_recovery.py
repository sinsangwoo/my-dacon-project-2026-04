import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def signal_recovery_mission():
    print("--- [MISSION: SIGNAL RECOVERY & AMPLIFICATION] ---")
    
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
    reg_tail = RandomForestRegressor(n_estimators=30, max_depth=8, n_jobs=-1).fit(X_tr[y_tr==1], y_gt_tr[y_tr==1])
    
    p_raw = clf.predict_proba(X_val)[:, 1]
    y_base = reg_base.predict(X_val)
    y_tail = reg_tail.predict(X_val)
    
    def eval_struct(yp, name):
        mae = mean_absolute_error(y_gt_val, yp)
        fpr = np.mean((y_gt_val < q90_val) & (yp > y_base))
        fpc = np.sum(np.abs(yp[(y_gt_val < q90_val) & (yp > y_base)] - y_gt_val[(y_gt_val < q90_val) & (yp > y_base)])) / len(y_gt_val)
        recall = np.sum((y_gt_val >= q90_val) & (yp > y_base)) / np.sum(y_gt_val >= q90_val)
        return {"Method": name, "MAE": mae, "FP_R": fpr, "FP_C": fpc, "Tail_Recall": recall}

    # [TASK 3: Low-Confidence Decomposition (p <= 0.4)]
    print("\n[TASK 3: LOW-CONFIDENCE DECOMPOSITION (p <= 0.4)]")
    low_p_mask = p_raw <= 0.4
    signal_mask = low_p_mask & (y_val == 1)
    noise_mask = low_p_mask & (y_val == 0)
    
    gain_low_p = (np.abs(y_gt_val - y_base) - np.abs(y_gt_val - y_tail))[low_p_mask]
    
    print(f"Signal Ratio in Low-P: {np.mean(y_val[low_p_mask]):.4f}")
    print(f"Noise Ratio in Low-P:  {1 - np.mean(y_val[low_p_mask]):.4f}")
    print(f"Avg Gain in Low-P:     {np.mean(gain_low_p):.4f}")

    # [TASK 1: Low-Confidence Recovery]
    print("\n[TASK 1: RECOVERY STRUCTURES]")
    results_t1 = []
    # A. Gain-based (ex-post proxy)
    y_t1a = np.where(gain_low_p > 0, y_tail[low_p_mask], y_base[low_p_mask]) # This is an upper bound simulation
    # B. KNN-based Recovery
    knn = NearestNeighbors(n_neighbors=5).fit(X_tr)
    _, indices = knn.kneighbors(X_val)
    neigh_y = y_binary[tr_idx][indices]
    consistency = np.mean(neigh_y, axis=1)
    # Rescue if consistency is high even if p is low
    rescue_gate = (p_raw <= 0.4) & (consistency >= 0.4)
    y_t1b = y_base + (p_raw**2) * (y_tail - y_base)
    y_t1b[rescue_gate] = y_tail[rescue_gate]
    results_t1.append(eval_struct(y_t1b, "T1B: KNN-Rescue"))
    print(pd.DataFrame(results_t1))

    # [TASK 2: Signal Amplification]
    print("\n[TASK 2: AMPLIFICATION STRATEGIES]")
    results_t2 = []
    # 1. Sharpening (Non-linear boost)
    p_boost = 1 / (1 + np.exp(-15 * (p_raw - 0.25)))
    y_t2_1 = y_base + p_boost * (y_tail - y_base)
    results_t2.append(eval_struct(y_t2_1, "T2-1: Sharp-Boost"))
    
    # 2. Interaction Expansion (Simplified via top features)
    # Using p^0.5 as a proxy for recovered non-linear signal
    p_amp = p_raw ** 0.5
    y_t2_2 = y_base + p_amp * (y_tail - y_base)
    results_t2.append(eval_struct(y_t2_2, "T2-2: Amp-Signal"))
    print(pd.DataFrame(results_t2))

    # [TASK 5: Final Comparison]
    print("\n[TASK 5: FINAL INTEGRATION]")
    # Recovery (KNN) + Amplification (Sharp)
    p_final = 1 / (1 + np.exp(-15 * (p_raw - 0.25)))
    y_final = y_base + p_final * (y_tail - y_base)
    y_final[rescue_gate] = y_tail[rescue_gate]
    
    final_res = [eval_struct(y_base + (p_raw**2) * (y_tail - y_base), "Baseline"),
                 eval_struct(y_final, "Recovery+Amp (FULL)")]
    print(pd.DataFrame(final_res))

if __name__ == "__main__":
    signal_recovery_mission()
