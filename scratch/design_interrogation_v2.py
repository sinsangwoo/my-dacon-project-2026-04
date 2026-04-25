
import os
import pandas as pd
import numpy as np
import logging
from src.config import Config
from src.data_loader import load_data, infer_feature_types, add_time_series_features, add_extreme_detection_features
from src.schema import BASE_COLS, FEATURE_SCHEMA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DesignInterrogation")

def run_interrogation():
    train, _ = load_data()
    df_ts = add_time_series_features(train.iloc[:10000].copy())
    df_ext = add_extreme_detection_features(df_ts)
    
    ts_cols = [c for c in df_ext.columns if c not in train.columns]
    
    categories = ['rolling_mean', 'rolling_std', 'diff_1', 'rate_1', 'slope_5', 'expanding_mean', 'expanding_std', 'accel', 'rel_to_mean', 'rel_rank', 'regime']
    
    high_corr_pairs = []
    base_col = 'order_inflow_15m' # Use a high-signal column
    
    logger.info(f"Checking redundancy for {base_col}...")
    relevant_feats = [c for c in df_ext.columns if base_col in c and c != base_col]
    
    corr_matrix = df_ext[relevant_feats].corr().abs()
    
    for i in range(len(relevant_feats)):
        for j in range(i + 1, len(relevant_feats)):
            c = corr_matrix.iloc[i, j]
            if c > 0.85:
                high_corr_pairs.append((relevant_feats[i], relevant_feats[j], c))
    
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    for f1, f2, c in high_corr_pairs:
        logger.info(f"REDUNDANT: {f1:<40} vs {f2:<40} | Corr: {c:.4f}")

if __name__ == "__main__":
    run_interrogation()
