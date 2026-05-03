import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

run_dir = 'outputs/run_20260429_103656'

print("=== RESEARCH FOR IMPLEMENTATION PLAN ===")

# 1. Target Distribution Analysis (for Mission 2 & 4)
y_train = np.load(f'{run_dir}/processed/y_train.npy')
q50 = np.median(y_train)
q90 = np.quantile(y_train, 0.90)
q95 = np.quantile(y_train, 0.95)
q99 = np.quantile(y_train, 0.99)
std = np.std(y_train)
iqr = np.quantile(y_train, 0.75) - np.quantile(y_train, 0.25)

print("\n--- Target Distribution (y_train) ---")
print(f"Mean: {np.mean(y_train):.4f}")
print(f"Std: {std:.4f}")
print(f"IQR: {iqr:.4f}")
print(f"Q50: {q50:.4f}, Q90: {q90:.4f}, Q95: {q95:.4f}, Q99: {q99:.4f}")
print(f"Max: {np.max(y_train):.4f}")

# 2. KS Distribution Analysis (for Mission 1)
# Let's sample train and test to get the real KS distribution
print("\n--- Loading data for KS Analysis ---")
train_base = pd.read_pickle(f'{run_dir}/processed/train_base.pkl')
test_base = pd.read_pickle(f'{run_dir}/processed/test_base.pkl')

# Use only numeric columns to match distribution.py
numeric_cols = [c for c in train_base.columns if pd.api.types.is_numeric_dtype(train_base[c]) and c not in ['ID', 'scenario_id', 'layout_id', 'avg_delay_minutes_next_30m']]
numeric_cols = [c for c in numeric_cols if c in test_base.columns]

print(f"Computing KS for {len(numeric_cols)} numeric columns (subsampled for speed)...")
s_tr = train_base.sample(min(20000, len(train_base)), random_state=42)
s_te = test_base.sample(min(20000, len(test_base)), random_state=42)

ks_stats = []
for col in numeric_cols:
    ks, _ = ks_2samp(s_tr[col].dropna(), s_te[col].dropna())
    ks_stats.append(ks)

ks_stats = np.array(ks_stats)
print(f"\n--- KS Statistic Distribution ---")
print(f"Min: {np.min(ks_stats):.4f}, Max: {np.max(ks_stats):.4f}")
print(f"Mean: {np.mean(ks_stats):.4f}, Median: {np.median(ks_stats):.4f}")
print(f"Q10: {np.quantile(ks_stats, 0.10):.4f}")
print(f"Q25: {np.quantile(ks_stats, 0.25):.4f}")
print(f"Q50: {np.quantile(ks_stats, 0.50):.4f}")
print(f"Q75: {np.quantile(ks_stats, 0.75):.4f}")
print(f"Q85: {np.quantile(ks_stats, 0.85):.4f}")
print(f"Q90: {np.quantile(ks_stats, 0.90):.4f}")
print(f"Q95: {np.quantile(ks_stats, 0.95):.4f}")
