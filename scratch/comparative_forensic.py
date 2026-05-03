import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"
predictions_dir = f"{base_dir}/outputs/{run_id}/predictions"

def compare_models():
    print(f"--- [COMPARATIVE FORENSIC: RAW vs STABLE] ---")
    
    # Load Labels
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    q99 = np.percentile(y_true, 99)
    mask_tail = y_true >= q99
    
    # Load OOFs
    oof_raw = np.load(f"{processed_dir}/oof_raw.npy", allow_pickle=True)
    oof_stable = np.load(f"{predictions_dir}/oof_stable.npy", allow_pickle=True)
    
    def get_metrics(name, oof):
        mae = mean_absolute_error(y_true, oof)
        tail_mae = mean_absolute_error(y_true[mask_tail], oof[mask_tail])
        mean_ratio = np.mean(oof) / np.mean(y_true)
        std_ratio = np.std(oof) / np.std(y_true)
        p99_pred = np.percentile(oof, 99)
        p99_ratio = p99_pred / q99
        return {
            "MAE": mae,
            "Tail_MAE": tail_mae,
            "Mean_Ratio": mean_ratio,
            "Std_Ratio": std_ratio,
            "P99_Ratio": p99_ratio
        }
    
    metrics_raw = get_metrics("Raw", oof_raw)
    metrics_stable = get_metrics("Stable", oof_stable)
    
    df_comp = pd.DataFrame([metrics_raw, metrics_stable], index=["Raw (1-Stage)", "Stable (2-Stage)"])
    print(df_comp.T)
    
    print("\n--- [INTERPRETATION] ---")
    diff_mae = (metrics_stable["MAE"] - metrics_raw["MAE"]) / metrics_raw["MAE"] * 100
    print(f"MAE Change: {diff_mae:+.2f}%")
    
    if metrics_stable["P99_Ratio"] < metrics_raw["P99_Ratio"] * 0.95:
        print("CRITICAL: Stable model significantly WORSENED tail coverage.")
    elif metrics_stable["P99_Ratio"] > metrics_raw["P99_Ratio"] * 1.05:
        print("INFO: Stable model improved tail coverage, but check MAE.")
    else:
        print("INFO: Tail coverage remained similar between stages.")

if __name__ == "__main__":
    compare_models()
