import pandas as pd
import numpy as np
import os
import json
import logging
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from scipy.stats import ks_2samp

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StructuralDiagnosis")

def run_diagnosis():
    RUN_ID = "run_20260427_105332"
    PROCESSED_PATH = f"outputs/{RUN_ID}/processed"
    
    # Load Data
    train_df = pd.read_pickle(f"{PROCESSED_PATH}/train_base.pkl")
    y_train = pd.read_csv("c:/Github_public/my_dacon_project/my-dacon-project-2026-04/data/train.csv")['avg_delay_minutes_next_30m']
    
    # Task 2: Variance Failure Root Cause - Prediction Compression Test
    # Load OOF if exists
    oof_path = f"outputs/{RUN_ID}/predictions/oof_stable.npy"
    if os.path.exists(oof_path):
        oof = np.load(oof_path)
        std_y = np.std(y_train)
        std_oof = np.std(oof)
        logger.info(f"[TASK 2] Target Std: {std_y:.4f} | Prediction Std: {std_oof:.4f} | Ratio: {std_oof/std_y:.4f}")
    
    # Task 2: Feature Variance Audit
    variances = train_df.var()
    low_var_count = (variances < 0.01).sum()
    logger.info(f"[TASK 2] Total Features: {len(variances)} | Low Var (<0.01): {low_var_count} ({low_var_count/len(variances):.1%})")

    # Task 3: P99 / Tail Failure Analysis - Isolation Test
    p99_threshold = np.percentile(y_train, 99)
    tail_mask = y_train >= p99_threshold
    logger.info(f"[TASK 3] P99 Threshold: {p99_threshold:.4f} | Tail Samples: {tail_mask.sum()}")
    
    # Task 4: Interaction Redundancy Check
    inter_cols = [c for c in train_df.columns if c.startswith('inter_')]
    if inter_logs := inter_cols:
        logger.info(f"[TASK 4] Found {len(inter_logs)} interactions. Checking max correlation with base features.")
        base_cols = [c for c in train_df.columns if not c.startswith('inter_') and not c.startswith('id')]
        corrs = []
        for i_col in inter_logs:
            # Check correlation with the two components if possible (parsing name)
            # inter_A_x_B
            parts = i_col.replace('inter_', '').split('_x_')
            max_corr = 0
            for p in parts:
                if p in train_df.columns:
                    c = train_df[i_col].corr(train_df[p])
                    max_corr = max(max_corr, abs(c))
            corrs.append(max_corr)
        if corrs:
            logger.info(f"[TASK 4] Interaction Redundancy (Max Corr with Components): Mean={np.mean(corrs):.4f} | Max={np.max(corrs):.4f}")

if __name__ == "__main__":
    run_diagnosis()
