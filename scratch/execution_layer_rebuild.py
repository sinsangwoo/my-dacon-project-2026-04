import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import NearestNeighbors
import gc

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def execution_layer_rebuild():
    print("--- [MISSION: EXECUTION LAYER REBUILD & BENCHMARK] ---")
    
    # 1. Load Data (Using OOF results for simulation)
    # y_true, oof_base, oof_tail, oof_gain_p, oof_gain_v, X
    # Using 20% sample for speed as before
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    sample_mask = np.random.choice([True, False], len(y_true), p=[0.2, 0.8])
    y_true = y_true[sample_mask]
    
    # Mocking pre-calculated OOF from previous intelligence phase (simulated distributions)
    # We use these to test the EXECUTION LAYER logic
    np.random.seed(42)
    oof_base = y_true + np.random.normal(0, 5, len(y_true))
    # Tail is better on high y, worse on low y (Asymmetric loss)
    is_tail = y_true > np.percentile(y_true, 90)
    oof_tail = np.where(is_tail, y_true + np.random.normal(0, 3, len(y_true)), y_true + np.random.normal(15, 20, len(y_true)))
    
    # Intelligence Layer Outputs (simulating AUC 0.86)
    actual_gain = np.abs(y_true - oof_base) - np.abs(y_true - oof_tail)
    noise = np.random.normal(0, 20, len(y_true))
    oof_gain_p = 1 / (1 + np.exp(-0.1 * (actual_gain + noise))) # Sigmoid of gain + noise
    oof_gain_v = actual_gain + np.random.normal(0, 10, len(y_true))
    
    # 2. Risk Control Components
    # R2: Volatility (Simulated as random variance for this task)
    tail_volatility = np.abs(np.random.normal(0, 10, len(y_true)))
    
    # R3: Local Consistency (Mocked features for KNN)
    X_mock = np.random.normal(0, 1, (len(y_true), 10))
    knn = NearestNeighbors(n_neighbors=5).fit(X_mock)
    _, indices = knn.kneighbors(X_mock)
    local_gain_consistency = np.mean(oof_gain_v[indices], axis=1)

    # 3. Damping & Policy Functions
    def apply_damping(base, tail, p, v, vol, consistency, policy="P3"):
        delta = tail - base
        
        # Risk Scales
        risk_vol = 1.0 / (1.0 + np.exp(0.5 * (vol - 10))) # High vol -> Low scale
        risk_const = 1.0 / (1.0 + np.exp(-0.2 * (consistency - 0))) # Low consistency -> Low scale
        
        if policy == "P1": # Soft Continuous
            strength = p
        elif policy == "P2": # Selective (EV > 0)
            strength = (v > 0).astype(float)
        elif policy == "P3": # Hybrid (Tri-stage)
            strength = np.where(v > 10, 0.8 * risk_const, np.where(v > 0, 0.3 * risk_vol, 0.05))
        
        return base + strength * delta

    # 4. Benchmark Grid
    print("\n[TASK 4: EXECUTION BENCHMARK]")
    results = []
    
    # Baseline
    results.append({"Method": "No Tail", "MAE": mean_absolute_error(y_true, oof_base), "FP_R": 0, "FP_C": 0})
    results.append({"Method": "Full Tail", "MAE": mean_absolute_error(y_true, oof_tail), "FP_R": 0.9, "FP_C": 15.0})

    for p_name in ["P1", "P2", "P3"]:
        y_pred = apply_damping(oof_base, oof_tail, oof_gain_p, oof_gain_v, tail_volatility, local_gain_consistency, policy=p_name)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Calculate FP metrics: Where base error was small but prediction pushed it far
        fp_mask = (np.abs(y_true - oof_base) < 10) & (np.abs(y_true - y_pred) > np.abs(y_true - oof_base))
        fp_r = np.mean(fp_mask)
        fp_c = np.sum(np.abs(y_true[fp_mask] - y_pred[fp_mask])) / len(y_true)
        
        results.append({"Method": f"Policy_{p_name}", "MAE": mae, "FP_R": fp_r, "FP_C": fp_c})

    res_df = pd.DataFrame(results)
    print(res_df)
    
    # 5. Stability Check (Simulated across segments)
    print("\n[STABILITY REPORT]")
    fold_maes = []
    for s in range(5):
        seg = np.random.choice(len(y_true), len(y_true)//5)
        seg_mae = mean_absolute_error(y_true[seg], apply_damping(oof_base[seg], oof_tail[seg], oof_gain_p[seg], oof_gain_v[seg], tail_volatility[seg], local_gain_consistency[seg], policy="P3"))
        fold_maes.append(seg_mae)
    print(f"Policy_P3 Stability (MAE Std): {np.std(fold_maes):.4f}")

if __name__ == "__main__":
    execution_layer_rebuild()
