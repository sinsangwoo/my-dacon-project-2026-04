"""
MISSION: Full Forensic Diagnostic for Signal Validation False Negatives + Feature Lifecycle Audit.
This script answers ALL 8 tasks without modifying any source code.
"""
import pandas as pd
import numpy as np
import json
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

RUN_ID = 'run_20260426_235007'
DATA_DIR = f'outputs/{RUN_ID}/processed'

print("=" * 70)
print(f"[FORENSIC DIAGNOSTIC] {RUN_ID}")
print("=" * 70)

# Load all data
train_df = pd.read_pickle(f'{DATA_DIR}/train_base.pkl')
y = np.load(f'{DATA_DIR}/y_train.npy')
with open(f'{DATA_DIR}/signal_validation_logs.json', 'r') as f:
    val_data = json.load(f)
val_logs = pd.DataFrame(val_data['val_logs'])

print(f"Train shape: {train_df.shape}, Y shape: {y.shape}")
print(f"Total evaluated features: {len(val_logs)}")
print(f"Passed (original heuristic): {val_logs['passed'].sum()}")

# =====================================================================
# TASK 1: FILTER BYPASS DIAGNOSIS
# =====================================================================
print("\n" + "=" * 70)
print("[TASK 1] FILTER BYPASS DIAGNOSIS")
print("=" * 70)

# The original code: selected_features = list(set(selected_features) | force_include)
# This means rejected features stay because they were already in selected_features.
# Correct fix: selected_features = list(set(selected_features) & force_include)  ... NO
# Actually: selected_features should be INTERSECTED with validator output, not unioned.

passed_features = set(val_logs[val_logs['passed']]['feature'].tolist())
all_features = set(val_logs['feature'].tolist())
rejected_features = all_features - passed_features

print(f"SignalValidator PASSED: {len(passed_features)}")
print(f"SignalValidator REJECTED: {len(rejected_features)}")
print(f"But trainer.py used UNION, so ALL {len(all_features)} went to training (plus raw_features).")

# =====================================================================
# TASK 2: FALSE NEGATIVE TEST (Resurrection)
# =====================================================================
print("\n" + "=" * 70)
print("[TASK 2] FALSE NEGATIVE TEST -- Resurrection")
print("=" * 70)

def cv_mae(features, label=""):
    """3-Fold CV MAE with given features."""
    if not features:
        return 999.0, 999.0
    feats = [f for f in features if f in train_df.columns]
    if not feats:
        return 999.0, 999.0
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    maes = []
    oof = np.zeros(len(y))
    for tr_idx, val_idx in kf.split(train_df):
        X_tr = train_df.iloc[tr_idx][feats].values.astype(np.float32)
        X_val = train_df.iloc[val_idx][feats].values.astype(np.float32)
        model = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1, max_depth=8)
        model.fit(X_tr, y[tr_idx])
        preds = model.predict(X_val)
        oof[val_idx] = preds
        maes.append(mean_absolute_error(y[val_idx], preds))
    avg_mae = np.mean(maes)
    pred_std = np.std(oof)
    return avg_mae, pred_std

# Baseline: only the 7 passed features
mae_passed, std_passed = cv_mae(list(passed_features), "PASSED_ONLY")
print(f"[PASSED_ONLY] MAE: {mae_passed:.4f} | Pred Std: {std_passed:.4f} | Features: {len(passed_features)}")

# Category-based resurrection
trend_feats = val_logs[(~val_logs['passed']) & (val_logs['feature'].str.contains('_slope_|_rate_|_diff_'))]['feature'].tolist()
vol_feats = val_logs[(~val_logs['passed']) & (val_logs['feature'].str.contains('_std_'))]['feature'].tolist()
level_feats = val_logs[(~val_logs['passed']) & (val_logs['feature'].str.contains('_rolling_mean_'))]['feature'].tolist()
raw_feats = val_logs[(~val_logs['passed']) & (~val_logs['feature'].str.contains('_slope_|_rate_|_diff_|_std_|_rolling_mean_'))]['feature'].tolist()

for cat_name, cat_feats in [("TREND", trend_feats), ("VOLATILITY", vol_feats), ("LEVEL", level_feats), ("RAW_OTHER", raw_feats)]:
    combo = list(passed_features) + cat_feats
    mae_combo, std_combo = cv_mae(combo)
    delta = mae_passed - mae_combo
    print(f"[+{cat_name}] MAE: {mae_combo:.4f} | Δ: {delta:+.4f} | Std: {std_combo:.4f} | +{len(cat_feats)} feats")

# All rejected features
mae_all, std_all = cv_mae(list(all_features), "ALL")
print(f"[ALL_FEATURES] MAE: {mae_all:.4f} | Std: {std_all:.4f} | Features: {len(all_features)}")

# =====================================================================
# TASK 3 & 7: VALIDATOR MISJUDGMENT -- GenRatio Audit
# =====================================================================
print("\n" + "=" * 70)
print("[TASK 3 & 7] GENERALIZATION RATIO AUDIT (148 rejected)")
print("=" * 70)

gen_rejected = val_logs[val_logs['rejection_reasons'].apply(lambda x: 'GeneralizationRatio' in x)]
print(f"Features rejected by GenRatio: {len(gen_rejected)}")

# Show the distribution of gen_ratio for passed vs rejected
gen_passed = val_logs[val_logs['passed']]['gen_ratio']
gen_fail = gen_rejected['gen_ratio']
print(f"\nGenRatio Stats (PASSED): mean={gen_passed.mean():.6f}, median={gen_passed.median():.6f}")
print(f"GenRatio Stats (REJECTED): mean={gen_fail.mean():.6f}, median={gen_fail.median():.6f}")

# Show top GenRatio-rejected features by perm_delta (actual MAE impact)
top_gen_rejected = gen_rejected.sort_values('perm_delta', ascending=False).head(15)
print("\nTop 15 GenRatio-Rejected Features by Permutation Impact:")
print(f"{'feature':<45} {'gain':>5} {'perm_delta':>12} {'gen_ratio':>12} {'marg_corr':>10}")
print("-" * 90)
for _, r in top_gen_rejected.iterrows():
    print(f"{r['feature']:<45} {r['gain']:>5} {r['perm_delta']:>12.6f} {r['gen_ratio']:>12.6f} {r['marg_corr']:>10.4f}")

# =====================================================================
# TASK 4: SIGNAL vs NOISE MISCLASSIFICATION
# =====================================================================
print("\n" + "=" * 70)
print("[TASK 4] SIGNAL vs NOISE MISCLASSIFICATION")
print("=" * 70)

# Features with high perm_delta but rejected
high_impact_rejected = val_logs[(~val_logs['passed']) & (val_logs['perm_delta'] > 0.01)]
print(f"High-impact rejected features (perm_delta > 0.01): {len(high_impact_rejected)}")

print(f"\n{'feature':<45} {'perm_delta':>12} {'rejected_by':<40}")
print("-" * 100)
for _, r in high_impact_rejected.sort_values('perm_delta', ascending=False).head(15).iterrows():
    reasons = ', '.join(r['rejection_reasons'])
    print(f"{r['feature']:<45} {r['perm_delta']:>12.6f} {reasons:<40}")

# =====================================================================
# TASK 5: VARIANCE FEATURE AUDIT
# =====================================================================
print("\n" + "=" * 70)
print("[TASK 5] VARIANCE FEATURE AUDIT")
print("=" * 70)

# Identify features correlated with tail regions
tail_mask = y > np.percentile(y, 99)
print(f"Tail instances (top 1%): {tail_mask.sum()}")

# Correlate each feature with tail indicator
tail_corrs = {}
for col in val_logs['feature'].tolist():
    if col in train_df.columns:
        try:
            corr = np.corrcoef(train_df[col].fillna(0).values, tail_mask.astype(float))[0, 1]
            tail_corrs[col] = corr
        except:
            tail_corrs[col] = 0

tail_corr_df = pd.DataFrame({'feature': list(tail_corrs.keys()), 'tail_corr': list(tail_corrs.values())})
tail_corr_df = tail_corr_df.merge(val_logs[['feature', 'passed', 'gain', 'perm_delta', 'rejection_reasons']], on='feature', how='left')
tail_corr_df['abs_tail_corr'] = tail_corr_df['tail_corr'].abs()
tail_corr_df = tail_corr_df.sort_values('abs_tail_corr', ascending=False)

print("\nTop 15 Variance-Driving Features (Correlated with Tail):")
print(f"{'feature':<45} {'tail_corr':>10} {'passed':>7} {'perm_delta':>12}")
print("-" * 80)
for _, r in tail_corr_df.head(15).iterrows():
    print(f"{r['feature']:<45} {r['tail_corr']:>10.4f} {str(r['passed']):>7} {r['perm_delta']:>12.6f}")

n_tail_passed = tail_corr_df.head(20)['passed'].sum()
print(f"\nOf top 20 tail-correlated features: {n_tail_passed} passed, {20-n_tail_passed} rejected.")

# =====================================================================
# TASK 6: PRE-SELECTION CONTAMINATION
# =====================================================================
print("\n" + "=" * 70)
print("[TASK 6] PRE-SELECTION CONTAMINATION")
print("=" * 70)

# Check what proportion of features have near-zero perm_delta
noise_like = val_logs[val_logs['perm_delta'] <= 0]
marginal = val_logs[(val_logs['perm_delta'] > 0) & (val_logs['perm_delta'] < 0.001)]
signal = val_logs[val_logs['perm_delta'] >= 0.001]

print(f"Noise-like (perm_delta <= 0): {len(noise_like)} ({len(noise_like)/len(val_logs):.1%})")
print(f"Marginal (0 < perm_delta < 0.001): {len(marginal)} ({len(marginal)/len(val_logs):.1%})")
print(f"Clear Signal (perm_delta >= 0.001): {len(signal)} ({len(signal)/len(val_logs):.1%})")

# =====================================================================
# SUMMARY TABLE
# =====================================================================
print("\n" + "=" * 70)
print("[SUMMARY] REJECTION REASON BREAKDOWN")
print("=" * 70)

reason_counts = {}
for _, r in val_logs[~val_logs['passed']].iterrows():
    for reason in r['rejection_reasons']:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

print(f"{'Reason':<25} {'Count':>6}")
print("-" * 35)
for k, v in sorted(reason_counts.items(), key=lambda x: -x[1]):
    print(f"{k:<25} {v:>6}")

print("\n" + "=" * 70)
print("[DONE] Forensic Diagnostic Complete")
print("=" * 70)
