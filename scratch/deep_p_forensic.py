import numpy as np
import pandas as pd
import os
import joblib
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"
models_dir = f"{base_dir}/outputs/{run_id}/models"

def run_p_forensic():
    print(f"--- [PHASE 2: PROBABILITY FORENSIC] ---")
    
    # 1. Load Data
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    q90 = np.percentile(y_true, 90)
    y_binary = (y_true >= q90).astype(int)
    
    # Load processed training data (contains features)
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    # We need to drop non-feature columns if any, but since it's processed, it should be ready
    # Note: We need to match the 685 features used in the run.
    # The manifest might have the feature list.
    
    # For now, let's look for oof_p in the run directory again or reconstruct from models
    # Since I cannot easily run full inference here without the exact feature list, 
    # I will look for any saved OOF p files.
    
    print(f"Data Loaded. y_true shape: {y_true.shape}, Tail Threshold: {q90:.4f}")
    
    # RECONSTRUCTION: Since I have the oof_stable.npy and oof_raw.npy, 
    # and I know the formula: stable = (p^k * tail + (1-p^k) * non_tail) * scalar
    # I can actually INVERT this to get a proxy p if k=1 (or if I have raw/stable)
    # But oof_raw might not be simple 'non_tail'. 
    
    # WAIT! I found residuals_raw.npy.
    # Let's check if we can find any classifier-specific artifacts.
    
    print("\n--- [SEARCHING FOR HIDDEN P-DATA] ---")
    p_files = [f for f in os.listdir(f"{base_dir}/outputs/{run_id}/predictions") if "p" in f]
    print(f"P-related files: {p_files}")

if __name__ == "__main__":
    run_p_forensic()
