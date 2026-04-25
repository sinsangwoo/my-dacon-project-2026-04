
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
    logger.info("Loading data for first-principles validation...")
    train, test = load_data()
    
    # ---------------------------------------------------------
    # TASK 1: Hardcoded Value Justification
    # ---------------------------------------------------------
    logger.info("--- TASK 1: Hardcoded Value Justification ---")
    
    # 1.1 NaN Jump Logic
    nan_ratios = train.isna().mean().sort_values()
    diffs = nan_ratios.diff().dropna()
    mean_jump = diffs.mean()
    std_jump = diffs.std()
    significant_jump = mean_jump + 2 * std_jump
    
    logger.info(f"NaN Ratios: Mean={nan_ratios.mean():.4f}, Std={nan_ratios.std():.4f}")
    logger.info(f"NaN Jumps: Mean={mean_jump:.4f}, Std={std_jump:.4f}, 2-Sigma={significant_jump:.4f}")
    
    max_jump = diffs.max()
    max_jump_idx = diffs.idxmax()
    logger.info(f"Max NaN Jump: {max_jump:.4f} at {max_jump_idx} (Threshold: {nan_ratios[max_jump_idx]:.4f})")
    
    # 1.2 Correlation Distribution
    # (Using a sample for speed)
    sample_df = train[BASE_COLS].sample(n=min(10000, len(train)), random_state=42).fillna(0)
    corr_matrix = sample_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).values.flatten()
    upper_tri = upper_tri[~np.isnan(upper_tri)]
    
    p95_corr = np.percentile(upper_tri, 95)
    p99_corr = np.percentile(upper_tri, 99)
    p995_corr = np.percentile(upper_tri, 99.5)
    
    logger.info(f"Correlation Percentiles: P95={p95_corr:.4f}, P99={p99_corr:.4f}, P99.5={p995_corr:.4f}")
    
    # 1.3 Variance Distribution
    variances = train[BASE_COLS].var()
    p1_var = variances.quantile(0.01)
    p5_var = variances.quantile(0.05)
    logger.info(f"Variance Percentiles: P1={p1_var:.2e}, P5={p5_var:.2e}")
    
    # ---------------------------------------------------------
    # TASK 2 & 3: Redundancy & TS Justification
    # ---------------------------------------------------------
    logger.info("--- TASK 2 & 3: Redundancy & TS Justification ---")
    
    df_ts = add_time_series_features(train.iloc[:5000].copy())
    df_ext = add_extreme_detection_features(df_ts)
    
    ts_cols = [c for c in df_ext.columns if c not in train.columns]
    logger.info(f"Generated {len(ts_cols)} features from {len(BASE_COLS)} base columns.")
    
    # Check correlation between categories
    # Categories: rolling_mean, rolling_std, diff, rate, slope, expanding, accel, rel_to_mean, rel_rank, regime
    categories = ['rolling_mean', 'rolling_std', 'diff', 'rate', 'slope', 'expanding_mean', 'expanding_std', 'accel', 'rel_to_mean', 'rel_rank', 'regime']
    
    redundancy_map = {}
    for cat in categories:
        cat_cols = [c for c in ts_cols if cat in c]
        if not cat_cols: continue
        
        # Check internal redundancy
        cat_corr = df_ext[cat_cols].corr().abs()
        mean_corr = (cat_corr.values.sum() - len(cat_cols)) / (len(cat_cols) * (len(cat_cols) - 1)) if len(cat_cols) > 1 else 0
        redundancy_map[cat] = mean_corr
        logger.info(f"Category {cat:<20}: {len(cat_cols):>3} features | Mean Internal Corr: {mean_corr:.4f}")

    # Check cross-category redundancy
    logger.info("Checking cross-category redundancy...")
    for i, cat1 in enumerate(categories):
        for cat2 in categories[i+1:]:
            cols1 = [c for c in ts_cols if cat1 in c]
            cols2 = [c for c in ts_cols if cat2 in c]
            if not cols1 or not cols2: continue
            
            # Sample one base column to check overlap
            base_col = BASE_COLS[0]
            feat1 = f"{base_col}_{cat1}" if f"{base_col}_{cat1}" in df_ext.columns else None
            feat2 = f"{base_col}_{cat2}" if f"{base_col}_{cat2}" in df_ext.columns else None
            
            if feat1 and feat2:
                c = df_ext[[feat1, feat2]].corr().iloc[0, 1]
                if abs(c) > 0.9:
                    logger.warning(f"HIGH REDUNDANCY: {feat1} vs {feat2} | Corr: {c:.4f}")

    # ---------------------------------------------------------
    # TASK 4: Overfitting Risk
    # ---------------------------------------------------------
    logger.info("--- TASK 4: Overfitting Risk ---")
    n_samples = len(train)
    n_features = len(FEATURE_SCHEMA['all_features'])
    logger.info(f"Samples: {n_samples} | Features: {n_features}")
    logger.info(f"Ratio: {n_samples / n_features:.2f} samples per feature")

if __name__ == "__main__":
    run_interrogation()
