import numpy as np
import pandas as pd
import os
import json

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"
predictions_dir = f"{base_dir}/outputs/{run_id}/predictions"

def load_data():
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    scenario_id = np.load(f"{processed_dir}/scenario_id.npy", allow_pickle=True)
    oof_raw = np.load(f"{processed_dir}/oof_raw.npy", allow_pickle=True)
    oof_stable = np.load(f"{predictions_dir}/oof_stable.npy", allow_pickle=True)
    return y_true, scenario_id, oof_raw, oof_stable

def analyze():
    y_true, scenario_id, oof_raw, oof_stable = load_data()
    
    # 1. Distribution Trace (Phase 3)
    def get_stats(arr):
        return {
            "mean": np.mean(arr),
            "std": np.std(arr),
            "p50": np.percentile(arr, 50),
            "p90": np.percentile(arr, 90),
            "p95": np.percentile(arr, 95),
            "p99": np.percentile(arr, 99)
        }
    
    stats_true = get_stats(y_true)
    stats_raw = get_stats(oof_raw)
    stats_stable = get_stats(oof_stable)
    
    print("\n--- PHASE 3: DISTRIBUTION TRACE ---")
    df_dist = pd.DataFrame([stats_true, stats_raw, stats_stable], index=["Target", "Raw_Pred", "Stable_Pred"])
    print(df_dist)
    
    # 2. Error Decomposition (Phase 6)
    errors_stable = np.abs(y_true - oof_stable)
    q99_val = np.percentile(y_true, 99)
    q95_val = np.percentile(y_true, 95)
    
    mask_top1 = y_true >= q99_val
    mask_top5 = (y_true >= q95_val) & (~mask_top1)
    mask_rest = y_true < q95_val
    
    mae_top1 = np.mean(errors_stable[mask_top1])
    mae_top5 = np.mean(errors_stable[mask_top5])
    mae_rest = np.mean(errors_stable[mask_rest])
    total_mae = np.mean(errors_stable)
    
    print("\n--- PHASE 6: ERROR DECOMPOSITION ---")
    print(f"Top 1% MAE: {mae_top1:.4f} (Count: {np.sum(mask_top1)})")
    print(f"Top 5% MAE: {mae_top5:.4f} (Count: {np.sum(mask_top5)})")
    print(f"Rest MAE:   {mae_rest:.4f} (Count: {np.sum(mask_rest)})")
    print(f"Overall MAE: {total_mae:.4f}")
    print(f"Error Multiplier (Top 1% / Overall): {mae_top1/total_mae:.2f}x")
    
    # 3. Scenario Polarization Validation (Phase 8)
    df = pd.DataFrame({
        "scenario_id": scenario_id,
        "y_true": y_true,
        "oof_stable": oof_stable,
        "error": errors_stable
    })
    
    scenario_stats = df.groupby("scenario_id").agg({
        "y_true": ["mean", "std", lambda x: np.percentile(x, 99)],
        "oof_stable": "mean",
        "error": "mean"
    })
    scenario_stats.columns = ["target_mean", "target_std", "target_p99", "pred_mean", "mae"]
    
    print("\n--- PHASE 8: SCENARIO POLARIZATION ---")
    print(f"Scenario Count: {len(scenario_stats)}")
    print(f"Max Scenario MAE: {scenario_stats['mae'].max():.4f}")
    print(f"Min Scenario MAE: {scenario_stats['mae'].min():.4f}")
    
    # Check if some scenarios have extremely low predictions despite high targets
    scenario_stats["bias"] = scenario_stats["pred_mean"] / (scenario_stats["target_mean"] + 1e-9)
    print(f"Mean Scenario Bias: {scenario_stats['bias'].mean():.4f}")
    print(f"Worst Scenario Bias (Min): {scenario_stats['bias'].min():.4f}")
    
    # 4. Mechanism Failure: Raw -> Stable Transformation
    # Analyze how much stable_pred reduced raw_pred in tail samples
    raw_to_stable_ratio_tail = np.mean(oof_stable[mask_top1]) / (np.mean(oof_raw[mask_top1]) + 1e-9)
    print("\n--- PHASE 5: BLENDING DILUTION PROXY ---")
    print(f"Tail Prediction Dilution (Stable/Raw on Top 1%): {raw_to_stable_ratio_tail:.4f}")
    
if __name__ == "__main__":
    analyze()
