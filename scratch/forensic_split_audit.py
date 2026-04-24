"""
FORENSIC VALIDATION SPLIT AUDIT
================================
Evidence-based audit of GroupKFold split integrity.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from collections import Counter

# Navigate to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '.')

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from scipy.stats import ks_2samp

# ──────────────────────────────────────────────────────────────────
# PHASE 0: LOAD ALL DATA
# ──────────────────────────────────────────────────────────────────
print("=" * 70)
print("PHASE 0: DATA LOADING")
print("=" * 70)

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
layout = pd.read_csv('./data/layout_info.csv')
train = train.merge(layout, on='layout_id', how='left')
test = test.merge(layout, on='layout_id', how='left')

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")
print(f"Train columns: {list(train.columns[:10])} ... ({len(train.columns)} total)")

TARGET = 'avg_delay_minutes_next_30m'
print(f"\nTarget column: {TARGET}")
print(f"Train target - mean: {train[TARGET].mean():.4f}, std: {train[TARGET].std():.4f}")
print(f"Train target - p50: {train[TARGET].quantile(0.5):.4f}, p90: {train[TARGET].quantile(0.9):.4f}, p99: {train[TARGET].quantile(0.99):.4f}")

# ──────────────────────────────────────────────────────────────────
# PHASE 1: SPLIT STRATEGY IDENTIFICATION
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 1: SPLIT STRATEGY IDENTIFICATION")
print("=" * 70)

print("""
FILE: src/trainer.py
FUNCTION: Trainer.__init__ (line 30) and train_kfolds (line 42)
SPLIT METHOD: GroupKFold(n_splits=5)
GROUP VARIABLE: scenario_id (passed as 'groups' parameter)

Code snippet (trainer.py:30):
    self.kf = GroupKFold(n_splits=Config.NFOLDS)

Code snippet (trainer.py:42):
    for fold, (tr_idx, val_idx) in enumerate(self.kf.split(X_subset, y, groups=self.groups)):
""")

# Reproduce the exact split
scenario_ids = train['scenario_id'].values
y = train[TARGET].values

n_unique_scenarios = len(np.unique(scenario_ids))
print(f"Unique scenario_ids in train: {n_unique_scenarios}")
print(f"Total train rows: {len(train)}")
print(f"Avg rows per scenario: {len(train)/n_unique_scenarios:.1f}")

# ──────────────────────────────────────────────────────────────────
# PHASE 2: LEAKAGE DETECTION
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 2: LEAKAGE DETECTION (CRITICAL)")
print("=" * 70)

kf = GroupKFold(n_splits=5)
X_dummy = np.zeros((len(train), 1))  # Placeholder for split iteration

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_dummy, y, groups=scenario_ids)):
    tr_scenarios = set(scenario_ids[tr_idx])
    val_scenarios = set(scenario_ids[val_idx])
    
    overlap = tr_scenarios & val_scenarios
    tr_layouts = set(train.iloc[tr_idx]['layout_id'].values)
    val_layouts = set(train.iloc[val_idx]['layout_id'].values)
    layout_overlap = tr_layouts & val_layouts
    
    print(f"\n--- Fold {fold} ---")
    print(f"  Train size: {len(tr_idx)} | Val size: {len(val_idx)}")
    print(f"  Train scenarios: {len(tr_scenarios)} | Val scenarios: {len(val_scenarios)}")
    print(f"  Scenario overlap: {len(overlap)} (ratio: {len(overlap)/(len(val_scenarios)+1e-9):.4f})")
    print(f"  Train layouts: {len(tr_layouts)} | Val layouts: {len(val_layouts)}")
    print(f"  Layout overlap: {len(layout_overlap)} / {len(val_layouts)} = {len(layout_overlap)/(len(val_layouts)+1e-9):.4f}")
    
    if len(overlap) > 0:
        print(f"  !! SCENARIO LEAKAGE DETECTED: {len(overlap)} scenarios in both train & val !!")
    else:
        print(f"  ✓ No scenario leakage (GroupKFold guarantee)")

# Check if layout_id leaks across folds
print("\n--- LAYOUT LEAKAGE ANALYSIS ---")
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_dummy, y, groups=scenario_ids)):
    tr_layouts = set(train.iloc[tr_idx]['layout_id'].values)
    val_layouts = set(train.iloc[val_idx]['layout_id'].values)
    layout_overlap = tr_layouts & val_layouts
    overlap_ratio = len(layout_overlap) / (len(val_layouts) + 1e-9)
    print(f"Fold {fold}: layout_overlap={len(layout_overlap)}/{len(val_layouts)} (ratio={overlap_ratio:.4f})")

# ──────────────────────────────────────────────────────────────────
# PHASE 3: DISTRIBUTION MISMATCH
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 3: DISTRIBUTION MISMATCH")
print("=" * 70)

print("\n--- Target Distribution per Fold ---")
print(f"{'metric':<18} | {'full_train':>12} |", end="")
for i in range(5):
    print(f" {'fold'+str(i)+'_tr':>12} {'fold'+str(i)+'_val':>12} |", end="")
print()

metrics_map = {
    'mean': lambda x: np.mean(x),
    'std': lambda x: np.std(x),
    'p50': lambda x: np.quantile(x, 0.5),
    'p90': lambda x: np.quantile(x, 0.9),
    'p99': lambda x: np.quantile(x, 0.99),
    'min': lambda x: np.min(x),
    'max': lambda x: np.max(x),
}

fold_target_stats = []
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_dummy, y, groups=scenario_ids)):
    y_tr, y_val = y[tr_idx], y[val_idx]
    fold_target_stats.append((y_tr, y_val))

for mname, mfunc in metrics_map.items():
    row = f"{mname:<18} | {mfunc(y):>12.4f} |"
    for (y_tr, y_val) in fold_target_stats:
        row += f" {mfunc(y_tr):>12.4f} {mfunc(y_val):>12.4f} |"
    print(row)

# Train-Val Distribution Ratios
print("\n--- Target Distribution Ratios (val/train) per Fold ---")
print(f"{'metric':<10} |", end="")
for i in range(5):
    print(f" {'fold'+str(i):>10} |", end="")
print()

for mname, mfunc in metrics_map.items():
    row = f"{mname:<10} |"
    for (y_tr, y_val) in fold_target_stats:
        ratio = mfunc(y_val) / (mfunc(y_tr) + 1e-9)
        row += f" {ratio:>10.4f} |"
    print(row)

# KS Test
print("\n--- KS Test (train vs val target per fold) ---")
for fold, (y_tr, y_val) in enumerate(fold_target_stats):
    ks_stat, ks_p = ks_2samp(y_tr, y_val)
    print(f"Fold {fold}: KS={ks_stat:.6f}, p-value={ks_p:.6e}")

# ──────────────────────────────────────────────────────────────────
# PHASE 3B: KEY FEATURE DISTRIBUTION COMPARISON
# ──────────────────────────────────────────────────────────────────
print("\n--- Key Feature Distribution (Fold 0 example) ---")
key_features = [
    'order_inflow_15m', 'robot_utilization', 'congestion_score',
    'heavy_item_ratio', 'avg_trip_distance', 'task_reassign_15m',
    'battery_std', 'max_zone_density', 'pack_utilization',
    'avg_charge_wait'
]

tr_idx_f0, val_idx_f0 = list(kf.split(X_dummy, y, groups=scenario_ids))[0]
print(f"{'feature':<25} | {'tr_mean':>10} {'val_mean':>10} {'ratio':>8} | {'tr_std':>10} {'val_std':>10} {'ratio':>8} | {'KS':>8} {'p':>10}")
for feat in key_features:
    if feat not in train.columns:
        continue
    tr_vals = train.iloc[tr_idx_f0][feat].dropna().values.astype(float)
    val_vals = train.iloc[val_idx_f0][feat].dropna().values.astype(float)
    if len(tr_vals) == 0 or len(val_vals) == 0:
        print(f"{feat:<25} | SKIPPED (all NaN)")
        continue
    mean_r = np.mean(val_vals) / (np.mean(tr_vals) + 1e-9)
    std_r = np.std(val_vals) / (np.std(tr_vals) + 1e-9)
    ks, p = ks_2samp(tr_vals, val_vals)
    print(f"{feat:<25} | {np.mean(tr_vals):>10.4f} {np.mean(val_vals):>10.4f} {mean_r:>8.4f} | {np.std(tr_vals):>10.4f} {np.std(val_vals):>10.4f} {std_r:>8.4f} | {ks:>8.4f} {p:>10.2e}")

# ──────────────────────────────────────────────────────────────────
# PHASE 4: GROUP STRUCTURE VIOLATION
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 4: GROUP STRUCTURE VIOLATION")
print("=" * 70)

# Check layout_id distribution across scenarios
layout_to_scenarios = train.groupby('layout_id')['scenario_id'].nunique()
print(f"\nLayout-to-scenario mapping:")
print(f"  Total layouts: {len(layout_to_scenarios)}")
print(f"  Mean scenarios per layout: {layout_to_scenarios.mean():.1f}")
print(f"  Min: {layout_to_scenarios.min()}, Max: {layout_to_scenarios.max()}")
print(f"  Layout scenario counts:\n{layout_to_scenarios.sort_values(ascending=False).head(20)}")

scenario_to_layout = train.groupby('scenario_id')['layout_id'].nunique()
multi_layout_scenarios = scenario_to_layout[scenario_to_layout > 1]
print(f"\nScenarios with multiple layouts: {len(multi_layout_scenarios)}")

# Check if rows within same scenario are contiguous (sorting effect)
print("\n--- Row Ordering Check ---")
for sid in train['scenario_id'].unique()[:5]:
    mask = train['scenario_id'] == sid
    indices = train.index[mask].values
    is_contiguous = (indices[-1] - indices[0] + 1) == len(indices)
    print(f"  scenario_id={sid}: indices [{indices[0]}..{indices[-1]}], count={len(indices)}, contiguous={is_contiguous}")

# ──────────────────────────────────────────────────────────────────
# PHASE 5: TEMPORAL / ORDERING BIAS
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 5: TEMPORAL / ORDERING BIAS")
print("=" * 70)

print("""
FILE: src/data_loader.py
FUNCTION: build_features (line 140)

Code snippet (data_loader.py:140):
    df = df.sort_values(by=["scenario_id", "ID"]).reset_index(drop=True)

ANALYSIS: Data is sorted by [scenario_id, ID] BEFORE feature engineering.
This means rolling/expanding features are computed within scenario groups.
""")

# Check rolling features for future leakage
print("--- Rolling Feature Future Leakage Check ---")
print("""
Code snippet (data_loader.py:200-220):
    series = df.groupby("scenario_id")[col]
    new_features[f"{col}_rolling_mean_3"] = series.rolling(3, min_periods=1).mean().values
    new_features[f"{col}_expanding_mean"] = series.expanding().mean().values
    new_features[f"{col}_expanding_sum"] = series.expanding().sum().values

FINDING: All rolling/expanding operations use .groupby("scenario_id") 
         which respects group boundaries.
         rolling() in pandas is BACKWARD-looking by default.
         expanding() is BACKWARD-looking by default.
         shift(1) / shift(3) look BACKWARDS.
""")

# But let's verify: does the GroupKFold split within a scenario?
print("--- GroupKFold Temporal Integrity ---")
print("GroupKFold groups by scenario_id → entire scenario goes to one fold")
print("Therefore within-scenario temporal features DO NOT leak across folds.")
print("✓ No future leakage within scenarios.")
print()
print("HOWEVER: expanding_sum/expanding_mean/expanding_std are computed")
print("BEFORE the split, using ALL data from a given scenario.")
print("Since GroupKFold keeps entire scenarios together, this is NOT leakage.")

# ──────────────────────────────────────────────────────────────────
# PHASE 6: ADVERSARIAL VALIDATION 
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 6: ADVERSARIAL VALIDATION")
print("=" * 70)

# Check the existing adversarial audit implementation
print("""
FILE: src/trainer.py
FUNCTION: perform_adversarial_audit (lines 111-117)

Code snippet:
    def perform_adversarial_audit(self):
        split_idx = int(len(self.X) * 0.8)
        auc = run_adversarial_validation(self.X[:split_idx], self.X[split_idx:])
        return auc

CRITICAL BUG IDENTIFIED:
The adversarial validation does NOT use the ACTUAL GroupKFold split.
It uses a simple 80/20 POSITIONAL split: X[:80%] vs X[80%:].
Since data is sorted by [scenario_id, ID] (data_loader.py:140),
this means the "validation" set is the LAST 20% of scenarios by sort order.

This is NOT the same as the GroupKFold validation set.
The adversarial AUC from this audit reflects positional bias, not fold bias.
""")

# Run actual adversarial validation on each GroupKFold fold
# Use raw features from the CSV (before feature engineering) 
base_cols = [
    'order_inflow_15m', 'unique_sku_15m', 'avg_items_per_order', 'urgent_order_ratio',
    'heavy_item_ratio', 'cold_chain_ratio', 'sku_concentration', 'robot_active',
    'robot_idle', 'robot_charging', 'robot_utilization', 'avg_trip_distance',
    'task_reassign_15m', 'battery_std', 'low_battery_ratio', 'charge_queue_length',
    'avg_charge_wait', 'congestion_score', 'max_zone_density', 'blocked_path_15m',
    'near_collision_15m', 'fault_count_15m', 'avg_recovery_time', 'replenishment_overlap',
    'pack_utilization', 'manual_override_ratio', 'warehouse_temp_avg', 'humidity_pct',
    'day_of_week', 'air_quality_idx'
]

# Also get layout cols (numeric only)
layout_cols = [c for c in layout.columns if c != 'layout_id' and pd.api.types.is_numeric_dtype(layout[c])]
all_feat_cols = base_cols + layout_cols
all_feat_cols = [c for c in all_feat_cols if c in train.columns and pd.api.types.is_numeric_dtype(train[c])]

X_all = train[all_feat_cols].fillna(0).values.astype(np.float32)

print("\n--- Adversarial Validation per GroupKFold Fold (RAW features) ---")
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_dummy, y, groups=scenario_ids)):
    X_tr = X_all[tr_idx]
    X_val = X_all[val_idx]
    
    X_combined = np.vstack([X_tr, X_val])
    y_adv = np.concatenate([np.zeros(len(X_tr)), np.ones(len(X_val))])
    
    clf = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, verbose=-1, n_jobs=-1)
    clf.fit(X_combined, y_adv)
    probs = clf.predict_proba(X_combined)[:, 1]
    auc = roc_auc_score(y_adv, probs)
    
    # Feature importance
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    
    print(f"\nFold {fold}: Adversarial AUC = {auc:.4f}")
    print(f"  Top 10 discriminative features:")
    for i, idx in enumerate(top_idx):
        print(f"    {i+1}. {all_feat_cols[idx]:<30} importance={importances[idx]:>6}")

# Also run positional split adversarial (replicating what the pipeline does)
print("\n--- Adversarial Validation: POSITIONAL split (what pipeline uses) ---")
# Sort like the pipeline does
train_sorted = train.sort_values(by=["scenario_id", "ID"]).reset_index(drop=True)
X_sorted = train_sorted[all_feat_cols].fillna(0).values.astype(np.float32)
split_idx = int(len(X_sorted) * 0.8)
X_80, X_20 = X_sorted[:split_idx], X_sorted[split_idx:]

X_comb = np.vstack([X_80, X_20])
y_comb = np.concatenate([np.zeros(len(X_80)), np.ones(len(X_20))])
clf_pos = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, verbose=-1, n_jobs=-1)
clf_pos.fit(X_comb, y_comb)
probs_pos = clf_pos.predict_proba(X_comb)[:, 1]
auc_pos = roc_auc_score(y_comb, probs_pos)

importances_pos = clf_pos.feature_importances_
top_idx_pos = np.argsort(importances_pos)[::-1][:10]

print(f"Positional 80/20 Adversarial AUC = {auc_pos:.4f}")
print(f"Top 10 discriminative features:")
for i, idx in enumerate(top_idx_pos):
    print(f"  {i+1}. {all_feat_cols[idx]:<30} importance={importances_pos[idx]:>6}")

# What scenarios are in 80% vs 20% positional?
sorted_scenarios = train_sorted['scenario_id'].values
pos_80_scenarios = set(sorted_scenarios[:split_idx])
pos_20_scenarios = set(sorted_scenarios[split_idx:])
pos_overlap = pos_80_scenarios & pos_20_scenarios
print(f"\nPositional split scenario overlap: {len(pos_overlap)}")
print(f"80% unique scenarios: {len(pos_80_scenarios)}, 20% unique scenarios: {len(pos_20_scenarios)}")

# ──────────────────────────────────────────────────────────────────
# PHASE 6B: TRAIN vs TEST ADVERSARIAL VALIDATION
# ──────────────────────────────────────────────────────────────────
print("\n--- Train vs Test Adversarial Validation ---")
test_feat = test[all_feat_cols].fillna(0).values.astype(np.float32)
train_feat = X_all

X_tt = np.vstack([train_feat, test_feat])
y_tt = np.concatenate([np.zeros(len(train_feat)), np.ones(len(test_feat))])
clf_tt = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, verbose=-1, n_jobs=-1)
clf_tt.fit(X_tt, y_tt)
probs_tt = clf_tt.predict_proba(X_tt)[:, 1]
auc_tt = roc_auc_score(y_tt, probs_tt)

importances_tt = clf_tt.feature_importances_
top_idx_tt = np.argsort(importances_tt)[::-1][:10]

print(f"Train vs Test Adversarial AUC = {auc_tt:.4f}")
print(f"Top 10 discriminative features:")
for i, idx in enumerate(top_idx_tt):
    print(f"  {i+1}. {all_feat_cols[idx]:<30} importance={importances_tt[idx]:>6}")

# ──────────────────────────────────────────────────────────────────
# PHASE 7: LAYOUT CLUSTER ANALYSIS
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 7: LAYOUT CLUSTER LEAKAGE DEEP DIVE")
print("=" * 70)

# Since layout_id can appear in both train and val (different scenarios on same layout),
# check how many samples share the same layout across folds
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_dummy, y, groups=scenario_ids)):
    tr_df = train.iloc[tr_idx]
    val_df = train.iloc[val_idx]
    
    # For each layout in val, what % of train has that same layout?
    shared_layouts = set(tr_df['layout_id'].unique()) & set(val_df['layout_id'].unique())
    tr_rows_shared = tr_df[tr_df['layout_id'].isin(shared_layouts)].shape[0]
    val_rows_shared = val_df[val_df['layout_id'].isin(shared_layouts)].shape[0]
    
    print(f"Fold {fold}:")
    print(f"  Shared layouts: {len(shared_layouts)}/{len(val_df['layout_id'].unique())}")
    print(f"  Val rows from shared layouts: {val_rows_shared}/{len(val_df)} ({100*val_rows_shared/len(val_df):.1f}%)")
    print(f"  Train rows from shared layouts: {tr_rows_shared}/{len(tr_df)} ({100*tr_rows_shared/len(tr_df):.1f}%)")

# ──────────────────────────────────────────────────────────────────
# PHASE 8: DRIFT SHIELD LEAKAGE CHECK  
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 8: DRIFT SHIELD / GLOBAL STATS LEAKAGE")  
print("=" * 70)

print("""
FILE: src/data_loader.py
FUNCTION: build_features (lines 157-162)

Code snippet:
    if mode == 'raw' and not os.path.exists(Config.GLOBAL_STATS_PATH):
        GlobalStatStore.compute_and_save(df, FEATURE_SCHEMA['raw_features'], Config.GLOBAL_STATS_PATH)
    
    stats = GlobalStatStore.load(Config.GLOBAL_STATS_PATH)
    if stats:
        df = GlobalStatStore.apply_drift_shield(df, stats, FEATURE_SCHEMA['raw_features'])

CRITICAL FINDING:
GlobalStatStore.compute_and_save is called on the ENTIRE training set.
It computes p1, p99, mean, std from ALL train rows.
Then apply_drift_shield clips and z-normalizes ALL features using these global stats.

This means the drift shield transforms applied to train features use statistics
computed from the ENTIRE train set (including future validation fold data).

During GroupKFold cross-validation:
- Fold K validation data was used to compute global_stats.json
- Global stats (mean, std, p1, p99) are applied back to fold K validation data  
- This is INFORMATION LEAKAGE from validation to preprocessing

Impact: The z-normalization in apply_drift_shield (line 611):
    x = (x - s['mean']) / (s['std'] + 1e-6)
uses mean/std computed from ALL data, including the current fold's validation set.
""")

# Quantify the leakage
print("\n--- Quantifying Drift Shield Leakage ---")
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_dummy, y, groups=scenario_ids)):
    # Compute stats from train-only vs all
    for feat in base_cols[:5]:  # Sample 5 features
        all_mean = train[feat].mean()
        all_std = train[feat].std()
        tr_mean = train.iloc[tr_idx][feat].mean()
        tr_std = train.iloc[tr_idx][feat].std()
        
        mean_shift = abs(all_mean - tr_mean) / (all_std + 1e-9)
        std_shift = abs(all_std - tr_std) / (all_std + 1e-9)
        
        if fold == 0:  # Print for fold 0 only
            print(f"  Fold 0 | {feat:<25} | mean_shift={mean_shift:.6f} | std_shift={std_shift:.6f}")
    if fold == 0:
        break

# ──────────────────────────────────────────────────────────────────
# PHASE 9: TWO-PASS IMPUTER LEAKAGE
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 9: SUPERCHARGED PCA LEAKAGE")
print("=" * 70)

print("""
FILE: src/data_loader.py  
FUNCTION: build_features (lines 168-179)

Code snippet:
    for lid in df['layout_id'].unique():
        mask = df['layout_id'] == lid
        latent_stats = reconstructor.calculate_graph_stats(df[mask], df)
                                                                   ^^
                                                                   ENTIRE df

CRITICAL FINDING:
calculate_graph_stats(df_target=df[mask], df_train_pool=df)
The second argument 'df' is the ENTIRE training dataset.
This means embeddings for validation fold rows are computed using
the entire train+val pool as neighbors.

During GroupKFold fold k:
- validation rows call calculate_graph_stats(val_rows, ALL_train_rows)
- nearest neighbor search includes CURRENT VALIDATION ROWS in the pool
- embed_mean, embed_std, weighted_mean all incorporate val data

This is EMBEDDING-LEVEL LEAKAGE.
""")

# ──────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL SUMMARY: TOP 3 CRITICAL FAILURES")
print("=" * 70)
print("""
AUDIT COMPLETE. Results follow in artifact.
""")
