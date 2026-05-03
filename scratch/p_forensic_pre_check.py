import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
import matplotlib.pyplot as plt

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"
predictions_dir = f"{base_dir}/outputs/{run_id}/predictions"

def load_data():
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    # We don't have the raw p, but we can reconstruct a 'proxy p' from oof_stable if we assume the blending logic
    # Actually, it's better to find if oof_p.npy exists. If not, we have to look at the logs for AUC.
    # WAIT! The prompt asks for p distribution. I must find the actual p.
    # Let me check the directory again for any p-related files.
    return y_true

def analyze_p_from_logs():
    # Since we don't have the oof_p.npy, we will use the logged metrics from 5_train_leakage_free.log
    # and 7_inference.log if possible.
    # But for a deep forensic, I'll check if I can find a way to get the probabilities.
    pass

if __name__ == "__main__":
    y_true = load_data()
    q90 = np.percentile(y_true, 90)
    y_binary = (y_true >= q90).astype(int)
    
    print(f"Tail Threshold (Q90): {q90:.4f}")
    print(f"Tail Positive Count: {np.sum(y_binary)}")
    
    # Check if we have oof_p or something similar in the processed folder that was missed
    print("\nFiles in processed dir:")
    for f in os.listdir(processed_dir):
        print(f" - {f}")
