import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, brier_score_loss
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.calibration import IsotonicRegression
from scipy.stats import skew, entropy

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def phase_3_controlled_sharpness():
    print("--- [MISSION: FP-ROBUST PHASE 3 - COMPREHENSIVE AUDIT] ---")
    
    # 1. Load Data
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    scenario_id = np.load(f"{processed_dir}/scenario_id.npy", allow_pickle=True)
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
    
    # Base Models
    clf = RandomForestClassifier(n_estimators=30, max_depth=8, n_jobs=-1, random_state=42).fit(X_tr, y_tr)
    reg_base = RandomForestRegressor(n_estimators=30, max_depth=8, n_jobs=-1).fit(X_tr[y_tr==0], y_gt_tr[y_tr==0])
    reg_tail = RandomForestRegressor(n_estimators=30, max_depth=8, n_jobs=-1).fit(X_tr[y_tr==1], y_gt_tr[y_tr==1])
    
    p_raw = clf.predict_proba(X_val)[:, 1]
    # Calibration for Filter
    iso = IsotonicRegression(out_of_bounds='clip').fit(clf.predict_proba(X_tr)[:, 1], y_tr)
    p_cal = iso.transform(p_raw)
    
    y_base = reg_base.predict(X_val)
    y_tail = reg_tail.predict(X_val)
    delta = y_tail - y_base
    gain_raw = np.abs(y_gt_val - y_base) - np.abs(y_gt_val - y_tail)

    # [TASK 1: Filter Only Structure]
    print("\n[TASK 1: FILTER ONLY STRUCTURE]")
    # 3. Filtered Structure: use raw p for blend, but reject if p_cal < 0.2
    mask_rej = p_cal < 0.2
    y_filt = y_base + (p_raw**2) * delta
    y_filt[mask_rej] = y_base[mask_rej]
    
    def m(yp, p_in, name):
        mae = mean_absolute_error(y_gt_val, yp)
        fpr = np.mean((y_gt_val < q90_val) & (yp > y_base))
        fpc = np.sum(np.abs(yp[(y_gt_val < q90_val) & (yp > y_base)] - y_gt_val[(y_gt_val < q90_val) & (yp > y_base)])) / len(y_gt_val)
        corr = np.corrcoef(p_in, gain_raw)[0, 1]
        return {"Method": name, "MAE": mae, "FP_R": fpr, "FP_C": fpc, "Corr": corr}

    results_t1 = [m(y_base + (p_raw**2) * delta, p_raw, "Raw-p"), 
                  m(y_base + (p_cal**2) * delta, p_cal, "Cal-p"),
                  m(y_filt, p_raw, "Filtered-Raw")]
    print(pd.DataFrame(results_t1))

    # [TASK 2: Confidence Collapse Dissection]
    print("\n[TASK 2: CONFIDENCE COLLAPSE ANALYSIS]")
    # Logit distribution (proxy using internal prob if logit not directly accessible from RF)
    logits = np.log(p_raw / (1 - p_raw + 1e-9))
    results_t2 = {
        "Metric": ["Logit Dist", "Prob Dist", "Entropy", "Scenario Var"],
        "Mean": [np.mean(logits), np.mean(p_raw), np.mean([entropy([pi, 1-pi]) for pi in p_raw]), np.var(p_raw)],
        "Std": [np.std(logits), np.std(p_raw), np.std([entropy([pi, 1-pi]) for pi in p_raw]), 0],
        "Skew": [skew(logits), skew(p_raw), 0, 0]
    }
    print(pd.DataFrame(results_t2))

    # [TASK 4: Expected Gain vs Cost]
    print("\n[TASK 4: EXPECTED GAIN vs COST]")
    # Cost = Penalty when FP (Ground truth < Q90 but we use tail)
    cost = np.where(y_gt_val < q90_val, np.abs(y_tail - y_gt_val) - np.abs(y_base - y_gt_val), 0)
    # Gain = Improvement when True Tail (Ground truth >= Q90)
    gain = np.where(y_gt_val >= q90_val, np.abs(y_base - y_gt_val) - np.abs(y_tail - y_gt_val), 0)
    
    print(f"Mean Expected Gain: {np.mean(gain[gain>0]):.4f}")
    print(f"Mean Expected Cost: {np.mean(cost[cost>0]):.4f}")
    print(f"Gain/Cost Ratio:    {np.mean(gain[gain>0]) / np.mean(cost[cost>0]):.4f}")

    # [TASK 3, 5, 7: Integrated Strategy Benchmark]
    print("\n[TASK 3,5,7: INTEGRATED STRATEGY BENCHMARK]")
    benchmarks = []
    # T3B: Local Consistency (KNN)
    knn = NearestNeighbors(n_neighbors=5).fit(X_tr)
    _, indices = knn.kneighbors(X_val)
    neigh_y = y_binary[tr_idx][indices]
    consistency = np.mean(neigh_y, axis=1)
    y_t3b = np.where(consistency > 0.4, y_filt, y_base)
    benchmarks.append(m(y_t3b, p_raw, "T3B: KNN-Gate"))
    
    # T5[1]: Asymmetric Residual Scaling (Reduce negative delta)
    y_t5_1 = y_base + (p_raw**2) * np.where(delta < 0, delta * 0.3, delta)
    benchmarks.append(m(y_t5_1, p_raw, "T5-1: Asym-Scale"))
    
    # T7-3: Hybrid Gate (Confidence + Consistency)
    y_t7_3 = np.where((p_cal > 0.25) & (consistency > 0.5), y_base + (p_raw**1.5) * delta, y_base)
    benchmarks.append(m(y_t7_3, p_raw, "T7-3: Hybrid-Gate"))

    print(pd.DataFrame(benchmarks))

if __name__ == "__main__":
    phase_3_controlled_sharpness()
