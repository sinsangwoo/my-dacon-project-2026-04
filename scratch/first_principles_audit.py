
import os
import pandas as pd
import numpy as np
import logging
from src.config import Config
from src.data_loader import load_data, infer_feature_types, add_time_series_features, add_extreme_detection_features
from src.schema import BASE_COLS, FEATURE_SCHEMA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FirstPrinciplesAudit")

def run_audit():
    logger.info("Loading data for First-Principles Audit...")
    train, _ = load_data()
    
    # Generate all current features
    df_ts = add_time_series_features(train.iloc[:20000].copy())
    df_ext = add_extreme_detection_features(df_ts)
    
    # Identify TS features
    ts_cols = [c for c in df_ext.columns if any(suffix in c for suffix in ['_rolling_', '_diff_', '_rate_', '_slope_', '_expanding_', '_rel_', '_accel', '_regime', '_consecutive'])]
    
    # 1. Cluster Analysis: Group features by signal type across all base columns
    suffixes = [
        '_rolling_mean_5', '_rolling_std_5', '_diff_1', '_rate_1', '_slope_5', 
        '_expanding_mean', '_rel_to_mean_5', '_rel_rank_5', '_accel', 
        '_volatility_expansion_std', '_regime_id', '_consecutive_above_q75'
    ]
    
    signal_overlap = {}
    for s in suffixes:
        cols = [c for c in ts_cols if c.endswith(s)]
        if not cols: continue
        
        # Mean correlation of this signal across all base columns vs others
        # (This is hard to summarize globally, let's pick top 5 base columns)
        top_base = ['order_inflow_15m', 'robot_utilization', 'congestion_score', 'avg_trip_distance', 'unique_sku_15m']
        
        for base in top_base:
            base_feats = [c for c in ts_cols if c.startswith(base)]
            if not base_feats: continue
            
            corr = df_ext[base_feats].corr().abs()
            
            # Find what this suffix overlaps with
            target_feat = f"{base}{s}"
            if target_feat not in corr.columns: continue
            
            high_corr = corr[target_feat][(corr[target_feat] > 0.85) & (corr.index != target_feat)]
            if not high_corr.empty:
                for idx, val in high_corr.items():
                    pair = tuple(sorted([target_feat, idx]))
                    signal_overlap[pair] = val

    logger.info("--- HIGH REDUNDANCY CLUSTERS (>0.85) ---")
    sorted_overlap = sorted(signal_overlap.items(), key=lambda x: x[1], reverse=True)
    for pair, val in sorted_overlap[:30]:
        logger.info(f"{pair[0]:<40} vs {pair[1]:<40} | Corr: {val:.4f}")

    # 2. Heuristic Sensitivity Analysis
    logger.info("--- HEURISTIC SENSITIVITY ---")
    # Clipping ranges
    target = train[Config.TARGET]
    p1, p99 = target.quantile(0.01), target.quantile(0.99)
    p05, p995 = target.quantile(0.005), target.quantile(0.995)
    logger.info(f"Target Clipping: [P1, P99] = [{p1:.2f}, {p99:.2f}] | [P0.5, P99.5] = [{p05:.2f}, {p995:.2f}]")
    
    # Correlation distribution
    sample_df = df_ext[ts_cols].sample(n=5000, random_state=42).fillna(0)
    corr_matrix = sample_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).values.flatten()
    upper_tri = upper_tri[~np.isnan(upper_tri)]
    
    logger.info(f"Global Correlation Percentiles: P90={np.percentile(upper_tri, 90):.4f}, P95={np.percentile(upper_tri, 95):.4f}, P99={np.percentile(upper_tri, 99):.4f}")

if __name__ == "__main__":
    run_audit()
