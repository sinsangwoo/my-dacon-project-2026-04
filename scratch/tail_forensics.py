import pandas as pd
import numpy as np
import os
import json
import logging
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TailForensics")

def run_tail_forensics():
    RUN_ID = "run_20260427_112548"
    PROCESSED_PATH = f"outputs/{RUN_ID}/processed"
    
    # Load Data
    train_df = pd.read_pickle(f"{PROCESSED_PATH}/train_base.pkl")
    y_train = pd.read_csv("c:/Github_public/my_dacon_project/my-dacon-project-2026-04/data/train.csv")['avg_delay_minutes_next_30m']
    
    # 500 rows in smoke test, let's use them as a proxy
    y_smoke = y_train.iloc[:500]
    
    # 1. Prediction Range Audit
    oof_path = f"outputs/{RUN_ID}/predictions/oof_stable.npy"
    if os.path.exists(oof_path):
        oof = np.load(oof_path)
        logger.info(f"[TAIL_AUDIT] Target Max: {y_smoke.max():.4f} | OOF Max: {oof.max():.4f}")
        logger.info(f"[TAIL_AUDIT] Target P99: {np.percentile(y_smoke, 99):.4f} | OOF P99: {np.percentile(oof, 99):.4f}")
        logger.info(f"[TAIL_AUDIT] P99 Ratio: {np.percentile(oof, 99) / np.percentile(y_smoke, 99):.4f}")

    # 2. Feature vs Target in Tail
    p99_thresh = np.percentile(y_smoke, 99)
    tail_mask = y_smoke >= p99_thresh
    
    # Compare feature correlations with target in tail vs global
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    corrs_global = train_df[numeric_cols].corrwith(y_smoke).abs()
    corrs_tail = train_df[tail_mask][numeric_cols].corrwith(y_smoke[tail_mask]).abs()
    
    # Features that lose signal in the tail
    signal_loss = (corrs_global - corrs_tail).sort_values(ascending=False)
    logger.info(f"[TAIL_AUDIT] Top Signal Loss features in tail:\n{signal_loss.head(10)}")
    
    # Features that gain signal in the tail
    signal_gain = (corrs_tail - corrs_global).sort_values(ascending=False)
    logger.info(f"[TAIL_AUDIT] Top Signal Gain features in tail:\n{signal_gain.head(10)}")

    # 3. Tail-only Training (Simplified)
    # Train on global, evaluate on tail
    # (Already did this with Q99_MAE=26.18)
    
    # H2: Model Output Compression Test
    # If we train with objective='regression' (MSE), it targets the mean.
    # If we use objective='mape' or 'quantile', it might stretch the range.
    
if __name__ == "__main__":
    run_tail_forensics()
