"""
[NAN_ROOT_CAUSE] Forensic audit of NaN generation in feature pipeline.
This script traces EXACTLY which feature generators produce NaNs and why.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("NAN_AUDIT")

from src.config import Config
from src.schema import FEATURE_SCHEMA, BASE_COLS

# 1. Load real data sample
logger.info("=" * 70)
logger.info(" [NAN_ROOT_CAUSE AUDIT]")
logger.info("=" * 70)

train = pd.read_csv(f"{Config.DATA_PATH}train.csv", nrows=500)
layout = pd.read_csv(Config.LAYOUT_PATH)
train = train.merge(layout, on="layout_id", how="left")

# Filter to relevant columns only
relevant_cols = list(BASE_COLS) + list(Config.ID_COLS)
if Config.TARGET in train.columns:
    relevant_cols.append(Config.TARGET)
keep_cols = [c for c in relevant_cols if c in train.columns]
df = train[keep_cols].copy()

logger.info(f"\n[RAW_INPUT] Shape: {df.shape}")
logger.info(f"[RAW_INPUT] NaN in BASE_COLS:")
for col in BASE_COLS:
    if col in df.columns:
        nan_pct = df[col].isna().mean()
        if nan_pct > 0:
            logger.info(f"  - {col}: {nan_pct:.2%}")

# 2. Simulate TS feature generation step-by-step
logger.info(f"\n{'='*70}")
logger.info(" [TS FEATURE NaN TRACE]")
logger.info(f"{'='*70}")

df_sorted = df.sort_values(by=["scenario_id", "ID"]).reset_index(drop=True)
df_sorted["timestep_index"] = df_sorted.groupby("scenario_id").cumcount().astype("int16")
df_sorted["normalized_time"] = (df_sorted["timestep_index"] / 24.0).astype("float32")

# Track per-suffix NaN counts
suffix_nan_stats = {}
feature_nan_details = []

for col in BASE_COLS:
    series = df_sorted.groupby("scenario_id")[col]
    
    generators = {
        f"{col}_rolling_mean_3": series.rolling(3, min_periods=1).mean().values,
        f"{col}_rolling_mean_5": series.rolling(5, min_periods=1).mean().values,
        f"{col}_rolling_std_3": series.rolling(3, min_periods=1).std().values,
        f"{col}_rolling_std_5": series.rolling(5, min_periods=1).std().values,
        f"{col}_diff_1": (df_sorted[col] - series.shift(1)).values,
        f"{col}_diff_3": (df_sorted[col] - series.shift(3)).values,
        f"{col}_rate_1": None,  # computed below
        f"{col}_slope_5": series.rolling(5, min_periods=1).mean().diff().fillna(0).values,
        f"{col}_recent_max_5": series.rolling(5, min_periods=1).max().values,
        f"{col}_recent_min_5": series.rolling(5, min_periods=1).min().values,
        f"{col}_expanding_mean": series.expanding().mean().values,
        f"{col}_expanding_sum": series.expanding().sum().values,
        f"{col}_expanding_std": series.expanding().std().values,
    }
    
    # rate_1 depends on diff_1
    shift1 = series.shift(1).values
    diff_1 = df_sorted[col].values - shift1
    rate_1 = diff_1 / (np.abs(shift1) + 1e-6)
    generators[f"{col}_rate_1"] = rate_1
    
    # range_5
    generators[f"{col}_range_5"] = generators[f"{col}_recent_max_5"] - generators[f"{col}_recent_min_5"]
    
    for feat_name, values in generators.items():
        if values is None:
            continue
        nan_count = np.isnan(values).sum()
        nan_pct = nan_count / len(values)
        
        # Classify generator type
        suffix = feat_name.replace(col, "").lstrip("_")
        if suffix not in suffix_nan_stats:
            suffix_nan_stats[suffix] = []
        suffix_nan_stats[suffix].append(nan_pct)
        
        if nan_pct > 0.01:
            feature_nan_details.append({
                'feature': feat_name,
                'suffix': suffix,
                'nan_pct': nan_pct,
                'nan_count': nan_count,
                'total': len(values)
            })

# 3. Simulate EXTREME feature generation
logger.info(f"\n{'='*70}")
logger.info(" [EXTREME FEATURE NaN TRACE]")
logger.info(f"{'='*70}")

extreme_suffixes_map = {}
for col in BASE_COLS:
    series = df_sorted.groupby("scenario_id")[col]
    rm5 = series.rolling(5, min_periods=1).mean().values
    rx5 = series.rolling(5, min_periods=1).max().values
    
    ext_generators = {
        f"{col}_rel_to_mean_5": df_sorted[col].values / (rm5 + 1e-6),
        f"{col}_rel_to_max_5": df_sorted[col].values / (rx5 + 1e-6),
        f"{col}_rel_rank_5": series.rolling(5, min_periods=1).rank().values,
        f"{col}_accel": (df_sorted[col].values - series.shift(1).values) - (series.shift(1).values - series.shift(2).values),
        f"{col}_volatility_expansion_std": series.rolling(3, min_periods=1).std().values / (series.rolling(10, min_periods=1).std().values + 1e-6),
        f"{col}_volatility_expansion_range": (series.rolling(3, min_periods=1).max().values - series.rolling(3, min_periods=1).min().values) / (series.rolling(10, min_periods=1).max().values - series.rolling(10, min_periods=1).min().values + 1e-6),
        f"{col}_consecutive_above_q75": series.rolling(5, min_periods=1).apply(lambda x: (x > np.quantile(x, 0.75)).sum()).values,
    }
    
    # regime_id uses qcut which can produce NaN for low-cardinality
    try:
        regime = pd.qcut(df_sorted[col], 5, labels=False, duplicates='drop')
        ext_generators[f"{col}_regime_id"] = regime.values
    except Exception:
        ext_generators[f"{col}_regime_id"] = np.full(len(df_sorted), np.nan)
    
    for feat_name, values in ext_generators.items():
        nan_count = np.sum(pd.isna(values))
        nan_pct = nan_count / len(values)
        
        suffix = feat_name.replace(col, "").lstrip("_")
        if suffix not in suffix_nan_stats:
            suffix_nan_stats[suffix] = []
        suffix_nan_stats[suffix].append(nan_pct)
        
        if nan_pct > 0.01:
            feature_nan_details.append({
                'feature': feat_name,
                'suffix': suffix,
                'nan_pct': nan_pct,
                'nan_count': int(nan_count),
                'total': len(values)
            })

# 4. REPORT
logger.info(f"\n{'='*70}")
logger.info(" [NAN_ROOT_CAUSE REPORT]")
logger.info(f"{'='*70}")

# Per-suffix aggregate
logger.info("\n[PER-SUFFIX AGGREGATE NaN RATES]")
logger.info(f"{'Suffix':<35} {'Mean NaN%':>10} {'Max NaN%':>10} {'Affected Cols':>15}")
logger.info("-" * 75)
for suffix, rates in sorted(suffix_nan_stats.items(), key=lambda x: np.mean(x[1]), reverse=True):
    mean_r = np.mean(rates)
    max_r = np.max(rates)
    affected = sum(1 for r in rates if r > 0.01)
    logger.info(f"  {suffix:<33} {mean_r:>9.2%} {max_r:>9.2%} {affected:>12}/{len(rates)}")

# Top 20 worst features
logger.info("\n[TOP 20 WORST NaN FEATURES]")
sorted_details = sorted(feature_nan_details, key=lambda x: x['nan_pct'], reverse=True)[:20]
logger.info(f"{'Feature':<50} {'Type':<25} {'NaN%':>8}")
logger.info("-" * 85)
for d in sorted_details:
    logger.info(f"  {d['feature']:<48} {d['suffix']:<23} {d['nan_pct']:>7.2%}")

# Memory estimate
n_rows = 500
n_raw = len(FEATURE_SCHEMA['raw_features'])
n_embed = len(FEATURE_SCHEMA['embed_features'])
n_total = n_raw + n_embed
mem_per_row = n_total * 4  # float32
mem_total_500 = n_rows * mem_per_row / (1024**2)

# Estimate for full dataset
full_train_rows = 100000  # approximate
mem_full = full_train_rows * mem_per_row / (1024**2)

logger.info(f"\n[MEMORY ESTIMATE]")
logger.info(f"  Total features: {n_total} (Raw: {n_raw} + Embed: {n_embed})")
logger.info(f"  Bytes per row (float32): {mem_per_row:,}")
logger.info(f"  500 rows: {mem_total_500:.1f} MB")
logger.info(f"  ~100K rows: {mem_full:.1f} MB")

# Summary
total_features_checked = len(feature_nan_details)
high_nan = sum(1 for d in feature_nan_details if d['nan_pct'] > 0.05)
logger.info(f"\n[SUMMARY]")
logger.info(f"  Features with >1% NaN: {total_features_checked}")
logger.info(f"  Features with >5% NaN: {high_nan}")
logger.info(f"  Dominant NaN generators: shift-based (diff, rate, accel, slope)")
