"""
[MISSION] TARGETING FAILURE FORENSIC ANALYSIS
Pure analysis — no model changes.
"""
import pandas as pd
import numpy as np
import json, os, pickle
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

RUN_ID = "run_20260503_172033"
PROC = f"outputs/{RUN_ID}/processed"
RECON = f"outputs/{RUN_ID}/models/reconstructors"

# ─── Load Data ───
train_base = pd.read_pickle(f"{PROC}/train_base.pkl")
y_train = np.load(f"{PROC}/y_train.npy")
oof_raw = np.load(f"outputs/{RUN_ID}/predictions/oof_raw.npy")
oof_base = np.load(f"outputs/{RUN_ID}/predictions/oof_base_raw.npy")

# ─── Load forensic fold data to get p-values ───
all_p = np.full(len(y_train), np.nan)
all_weight = np.full(len(y_train), np.nan)
all_gap = np.full(len(y_train), np.nan)
all_preds_t = np.full(len(y_train), np.nan)
all_preds_nt = np.full(len(y_train), np.nan)

# We need fold indices - reconstruct from the splits
# Since forensic data has y_val that maps to OOF indices, use y_val matching
for fold in range(5):
    path = f"{RECON}/forensic_fold_{fold}.json"
    if not os.path.exists(path): continue
    with open(path) as f: d = json.load(f)
    p = np.array(d['p_val'])
    y_val = np.array(d['y_val'])
    weight = np.array(d['final_weight'])
    gap = np.array(d['gap'])
    preds_t = np.array(d['preds_t'])
    preds_nt = np.array(d['preds_nt'])
    
    # Match by finding indices where oof is not NaN AND matches y_val pattern
    # Use the non-NaN mask from oof to find which samples belong to this fold
    valid_mask = ~np.isnan(oof_raw)
    # Actually, let's just use sequential accumulation
    # The fold data maps to val indices - we need to find them
    # Match y_val values to y_train to find indices
    # This is approximate but sufficient for forensic analysis

# Alternative: just use oof-level analysis directly
valid = ~np.isnan(oof_raw)
y_v = y_train[valid]
oof_v = oof_raw[valid]
base_v = oof_base[valid]
train_v = train_base[valid].reset_index(drop=True)

q90 = np.percentile(y_v, 90)
q95 = np.percentile(y_v, 95)

# Define groups based on OOF predictions vs actuals
is_true_tail = y_v >= q90
# "Predicted tail" = model shifted prediction significantly toward tail model
# Use weight proxy: where oof deviates significantly from base
pred_shift = np.abs(oof_v - base_v) / (np.abs(base_v) + 1e-6)
is_predicted_tail = pred_shift > 0.1  # significant shift from baseline

true_positive = is_true_tail & is_predicted_tail  # correctly attacked
false_positive = ~is_true_tail & is_predicted_tail  # wrongly attacked
false_negative = is_true_tail & ~is_predicted_tail  # missed tail

print("=" * 75)
print(" [1] TRUE vs FALSE TAIL FORENSIC ANALYSIS")
print("=" * 75)
print(f"Q90 threshold: {q90:.2f}")
print(f"Total valid samples: {len(y_v)}")
print(f"True Tail (y >= Q90): {is_true_tail.sum()} ({is_true_tail.mean()*100:.1f}%)")
print(f"Predicted Tail (shifted): {is_predicted_tail.sum()} ({is_predicted_tail.mean()*100:.1f}%)")
print(f"TRUE POSITIVE (correct attack): {true_positive.sum()}")
print(f"FALSE POSITIVE (wrong attack): {false_positive.sum()}")
print(f"FALSE NEGATIVE (missed): {false_negative.sum()}")
print(f"Precision: {true_positive.sum() / (true_positive.sum() + false_positive.sum() + 1e-9):.4f}")
print(f"Recall: {true_positive.sum() / (true_positive.sum() + false_negative.sum() + 1e-9):.4f}")

# Key features to compare
key_features = [
    'robot_idle', 'congestion_score', 'order_inflow_15m', 'robot_utilization',
    'robot_charging', 'task_reassign_15m', 'max_zone_density', 'battery_std',
    'low_battery_ratio', 'near_collision_15m', 'avg_trip_distance',
    'pack_utilization', 'timestep_index', 'day_of_week',
    'robot_idle_rolling_mean_5', 'congestion_score_rolling_mean_5',
    'order_inflow_15m_rolling_mean_5', 'robot_utilization_rolling_mean_5',
    'robot_idle_diff_1', 'congestion_score_diff_1',
    'robot_idle_rel_to_mean_5', 'congestion_score_rel_to_mean_5',
]
key_features = [f for f in key_features if f in train_v.columns]

print("\n--- Feature Distribution Comparison (Mean ± Std) ---")
print(f"{'Feature':<45} | {'TRUE_POS':>10} | {'FALSE_POS':>10} | {'FALSE_NEG':>10} | {'BULK':>10}")
print("-" * 100)

bulk = ~is_true_tail & ~is_predicted_tail

for feat in key_features:
    vals_tp = train_v.loc[true_positive, feat].dropna()
    vals_fp = train_v.loc[false_positive, feat].dropna()
    vals_fn = train_v.loc[false_negative, feat].dropna()
    vals_bulk = train_v.loc[bulk, feat].dropna()
    
    m_tp = vals_tp.mean() if len(vals_tp) > 0 else 0
    m_fp = vals_fp.mean() if len(vals_fp) > 0 else 0
    m_fn = vals_fn.mean() if len(vals_fn) > 0 else 0
    m_bulk = vals_bulk.mean() if len(vals_bulk) > 0 else 0
    
    # Highlight if FP looks different from TP
    diff_ratio = abs(m_tp - m_fp) / (abs(m_tp) + 1e-9)
    marker = " <<<" if diff_ratio > 0.3 else ""
    print(f"{feat:<45} | {m_tp:>10.3f} | {m_fp:>10.3f} | {m_fn:>10.3f} | {m_bulk:>10.3f}{marker}")

# ─── SECTION 2: DEMAND vs CAPACITY INTERACTIONS ───
print("\n" + "=" * 75)
print(" [2] DEMAND vs CAPACITY INTERACTION VERIFICATION")
print("=" * 75)

interactions = {}
if 'congestion_score' in train_v.columns and 'robot_idle' in train_v.columns:
    interactions['congestion_per_idle'] = train_v['congestion_score'] / (train_v['robot_idle'] + 1e-6)
if 'order_inflow_15m' in train_v.columns and 'robot_idle' in train_v.columns:
    interactions['inflow_per_idle'] = train_v['order_inflow_15m'] / (train_v['robot_idle'] + 1e-6)
if 'charge_queue_length' in train_v.columns and 'robot_active' in train_v.columns:
    interactions['queue_per_active'] = train_v['charge_queue_length'] / (train_v['robot_active'] + 1e-6)
if 'task_reassign_15m' in train_v.columns and 'robot_idle' in train_v.columns:
    interactions['reassign_per_idle'] = train_v['task_reassign_15m'] / (train_v['robot_idle'] + 1e-6)
if 'max_zone_density' in train_v.columns and 'robot_idle' in train_v.columns:
    interactions['density_per_idle'] = train_v['max_zone_density'] / (train_v['robot_idle'] + 1e-6)
if 'order_inflow_15m' in train_v.columns and 'robot_active' in train_v.columns:
    interactions['inflow_per_active'] = train_v['order_inflow_15m'] / (train_v['robot_active'] + 1e-6)
if 'congestion_score' in train_v.columns and 'robot_active' in train_v.columns:
    interactions['congestion_per_active'] = train_v['congestion_score'] / (train_v['robot_active'] + 1e-6)
if 'near_collision_15m' in train_v.columns and 'robot_idle' in train_v.columns:
    interactions['collision_per_idle'] = train_v['near_collision_15m'] / (train_v['robot_idle'] + 1e-6)
if 'battery_std' in train_v.columns and 'robot_idle' in train_v.columns:
    interactions['battery_stress_idle'] = train_v['battery_std'] / (train_v['robot_idle'] + 1e-6)
if 'pack_utilization' in train_v.columns and 'robot_idle' in train_v.columns:
    interactions['pack_per_idle'] = train_v['pack_utilization'] / (train_v['robot_idle'] + 1e-6)

print(f"\n{'Interaction':<35} | {'AUC(tail)':>10} | {'KS(TP-FP)':>10} | {'KS p-val':>10} | {'Mean TP':>10} | {'Mean FP':>10}")
print("-" * 105)

for name, vals in interactions.items():
    vals_clean = vals.fillna(0).replace([np.inf, -np.inf], 0)
    # AUC for tail detection
    try:
        auc = roc_auc_score(is_true_tail.astype(int), vals_clean)
    except:
        auc = 0.5
    
    # KS between TP and FP
    tp_vals = vals_clean[true_positive].values
    fp_vals = vals_clean[false_positive].values
    if len(tp_vals) > 5 and len(fp_vals) > 5:
        ks_stat, ks_p = ks_2samp(tp_vals, fp_vals)
    else:
        ks_stat, ks_p = 0, 1
    
    m_tp = np.mean(tp_vals) if len(tp_vals) > 0 else 0
    m_fp = np.mean(fp_vals) if len(fp_vals) > 0 else 0
    
    marker = " <<<" if ks_stat > 0.2 and ks_p < 0.01 else ""
    print(f"{name:<35} | {auc:>10.4f} | {ks_stat:>10.4f} | {ks_p:>10.4g} | {m_tp:>10.3f} | {m_fp:>10.3f}{marker}")

# ─── SECTION 3: FEATURE DISCRIMINATION SCORE ───
print("\n" + "=" * 75)
print(" [3] FEATURE DISCRIMINATION SCORE (FP Killer Features)")
print("=" * 75)

numeric_cols = train_v.select_dtypes(include=[np.number]).columns.tolist()
# Remove ID-like columns
numeric_cols = [c for c in numeric_cols if c not in ['ID', 'scenario_id']]

scores = []
for col in numeric_cols:
    vals = train_v[col].fillna(0).replace([np.inf, -np.inf], 0)
    
    # 1. Tail separation (AUC)
    try:
        tail_auc = roc_auc_score(is_true_tail.astype(int), vals)
    except:
        tail_auc = 0.5
    
    # 2. TP vs FP separation (KS) - THE KEY METRIC
    tp_vals = vals[true_positive].values
    fp_vals = vals[false_positive].values
    if len(tp_vals) > 5 and len(fp_vals) > 5:
        ks_tp_fp, ks_p = ks_2samp(tp_vals, fp_vals)
    else:
        ks_tp_fp, ks_p = 0, 1
    
    scores.append({
        'feature': col,
        'tail_auc': tail_auc,
        'tp_fp_ks': ks_tp_fp,
        'tp_fp_p': ks_p,
        'mean_tp': np.mean(tp_vals) if len(tp_vals) > 0 else 0,
        'mean_fp': np.mean(fp_vals) if len(fp_vals) > 0 else 0,
    })

scores_df = pd.DataFrame(scores)

# TOP FP KILLERS: features that separate TP from FP
print("\n[TOP 15 FP-KILLER FEATURES] (Highest KS between True Positive and False Positive)")
top_fp_killers = scores_df.nlargest(15, 'tp_fp_ks')
print(f"{'Feature':<45} | {'KS(TP-FP)':>10} | {'p-value':>10} | {'Tail AUC':>10} | {'Mean TP':>10} | {'Mean FP':>10}")
print("-" * 110)
for _, row in top_fp_killers.iterrows():
    print(f"{row['feature']:<45} | {row['tp_fp_ks']:>10.4f} | {row['tp_fp_p']:>10.4g} | {row['tail_auc']:>10.4f} | {row['mean_tp']:>10.3f} | {row['mean_fp']:>10.3f}")

# MISLEADING FEATURES: high tail AUC but LOW TP-FP separation (cannot tell real from fake)
print("\n[TOP 10 MISLEADING FEATURES] (High Tail AUC but CANNOT separate TP from FP)")
misleading = scores_df[scores_df['tail_auc'] > 0.55].nsmallest(10, 'tp_fp_ks')
print(f"{'Feature':<45} | {'Tail AUC':>10} | {'KS(TP-FP)':>10} | {'VERDICT':>15}")
print("-" * 95)
for _, row in misleading.iterrows():
    print(f"{row['feature']:<45} | {row['tail_auc']:>10.4f} | {row['tp_fp_ks']:>10.4f} | {'MISLEADING':>15}")

# ─── SECTION 4: CLASSIFIER FAILURE ROOT CAUSE ───
print("\n" + "=" * 75)
print(" [4] CLASSIFIER FAILURE ROOT CAUSE")
print("=" * 75)

# Analyze FP characteristics
print(f"\n[FP Profile] ({false_positive.sum()} samples)")
print(f"  Mean y_true:  {y_v[false_positive].mean():.2f} (should be < {q90:.2f})")
print(f"  Max y_true:   {y_v[false_positive].max():.2f}")
print(f"  Q75 y_true:   {np.percentile(y_v[false_positive], 75):.2f}")
print(f"  Q90 y_true:   {np.percentile(y_v[false_positive], 90):.2f}")

# How close are FPs to the boundary?
fp_y = y_v[false_positive]
boundary_fp = (fp_y >= q90 * 0.7).sum()
print(f"  Near-boundary FPs (y >= 0.7*Q90): {boundary_fp} ({boundary_fp/len(fp_y)*100:.1f}%)")
deep_fp = (fp_y < q90 * 0.3).sum()
print(f"  Deep FPs (y < 0.3*Q90): {deep_fp} ({deep_fp/len(fp_y)*100:.1f}%)")

print(f"\n[FN Profile] ({false_negative.sum()} samples)")
print(f"  Mean y_true:  {y_v[false_negative].mean():.2f}")
print(f"  Min y_true:   {y_v[false_negative].min():.2f}")
print(f"  Q25 y_true:   {np.percentile(y_v[false_negative], 25):.2f}")

# Label noise check: distribution of y in Q85-Q95 region
q85 = np.percentile(y_v, 85)
boundary_zone = (y_v >= q85) & (y_v <= q95)
print(f"\n[BOUNDARY ZONE ANALYSIS] (Q85-Q95: {boundary_zone.sum()} samples)")
print(f"  These are 'ambiguous' samples near the tail threshold.")
bz_pred_tail = is_predicted_tail[boundary_zone].sum()
print(f"  Predicted as tail: {bz_pred_tail} ({bz_pred_tail/boundary_zone.sum()*100:.1f}%)")

# ─── SECTION 5: GATING FAILURE ANALYSIS ───
print("\n" + "=" * 75)
print(" [5] GATING FAILURE ANALYSIS")
print("=" * 75)

# Load per-fold forensic data for gating analysis
all_cases = []
for fold in range(5):
    path = f"{RECON}/forensic_fold_{fold}.json"
    if not os.path.exists(path): continue
    with open(path) as f: d = json.load(f)
    for i in range(len(d['p_val'])):
        all_cases.append({
            'p': d['p_val'][i],
            'weight': d['final_weight'][i],
            'gap': d['gap'][i],
            'preds_t': d['preds_t'][i],
            'preds_nt': d['preds_nt'][i],
            'y': d['y_val'][i],
        })

cases_df = pd.DataFrame(all_cases)
cases_df['is_tail'] = cases_df['y'] >= np.percentile(cases_df['y'], 90)
cases_df['is_fp'] = (~cases_df['is_tail']) & (cases_df['weight'] > 0.3)
cases_df['is_tp'] = cases_df['is_tail'] & (cases_df['weight'] > 0.3)
cases_df['regressor_ratio'] = cases_df['preds_t'] / (cases_df['preds_nt'] + 1e-6)

# Case 1: High P but FP
high_p_fp = cases_df[(cases_df['p'] > 0.65) & cases_df['is_fp']]
high_p_tp = cases_df[(cases_df['p'] > 0.65) & cases_df['is_tp']]
print(f"\n[CASE A] High Confidence FPs (p > 0.65)")
print(f"  Count: {len(high_p_fp)} FPs vs {len(high_p_tp)} TPs")
if len(high_p_fp) > 0:
    print(f"  FP Avg Weight: {high_p_fp['weight'].mean():.4f}")
    print(f"  FP Avg Gap: {high_p_fp['gap'].mean():.4f}")
    print(f"  FP Avg Regressor Ratio: {high_p_fp['regressor_ratio'].mean():.4f}")
    print(f"  TP Avg Regressor Ratio: {high_p_tp['regressor_ratio'].mean():.4f}")
    
    # Does the regressor agree with the classifier on FPs?
    fp_regressor_agrees = (high_p_fp['regressor_ratio'] > 1.5).sum()
    print(f"  FPs where regressor ALSO agrees (ratio > 1.5): {fp_regressor_agrees} ({fp_regressor_agrees/len(high_p_fp)*100:.1f}%)")
    fp_regressor_disagrees = (high_p_fp['regressor_ratio'] < 1.2).sum()
    print(f"  FPs where regressor DISAGREES (ratio < 1.2): {fp_regressor_disagrees} ({fp_regressor_disagrees/len(high_p_fp)*100:.1f}%)")

# Case 2: OR-gate opened by condition B (regressor divergence)
cond_b_fp = cases_df[(cases_df['p'] <= 0.65) & (cases_df['p'] > 0.4) & (cases_df['regressor_ratio'] > 2.0) & cases_df['is_fp']]
cond_b_tp = cases_df[(cases_df['p'] <= 0.65) & (cases_df['p'] > 0.4) & (cases_df['regressor_ratio'] > 2.0) & cases_df['is_tp']]
print(f"\n[CASE B] OR-Gate Bypass FPs (low p, high regressor divergence)")
print(f"  Count: {len(cond_b_fp)} FPs vs {len(cond_b_tp)} TPs")
if len(cond_b_fp) > 0:
    print(f"  FP Avg P: {cond_b_fp['p'].mean():.4f}")
    print(f"  FP Avg Weight: {cond_b_fp['weight'].mean():.4f}")
    print(f"  FP Avg y_true: {cond_b_fp['y'].mean():.2f}")

# Case 3: Risk controller intervention analysis
risk_active = cases_df[cases_df['gap'] > 3.0]  # GAP_THRESHOLD
risk_fp = risk_active[risk_active['is_fp']]
risk_tp = risk_active[risk_active['is_tp']]
print(f"\n[CASE C] Risk Controller Active Zone (gap > 3.0)")
print(f"  Total: {len(risk_active)}, FPs: {len(risk_fp)}, TPs: {len(risk_tp)}")
if len(risk_fp) > 0 and len(risk_tp) > 0:
    print(f"  Risk Zone FP Avg Weight: {risk_fp['weight'].mean():.4f}")
    print(f"  Risk Zone TP Avg Weight: {risk_tp['weight'].mean():.4f}")
    print(f"  Risk Zone FP Avg P: {risk_fp['p'].mean():.4f}")
    print(f"  Risk Zone TP Avg P: {risk_tp['p'].mean():.4f}")

# FP damage quantification
print(f"\n[FP DAMAGE QUANTIFICATION]")
fp_cases = cases_df[cases_df['is_fp']]
if len(fp_cases) > 0:
    fp_pred = fp_cases['weight'] * fp_cases['preds_t'] + (1 - fp_cases['weight']) * fp_cases['preds_nt']
    fp_base = fp_cases['preds_nt']
    fp_damage = np.abs(fp_pred - fp_cases['y']) - np.abs(fp_base - fp_cases['y'])
    print(f"  Total FP Count: {len(fp_cases)}")
    print(f"  FPs that INCREASED error: {(fp_damage > 0).sum()} ({(fp_damage > 0).mean()*100:.1f}%)")
    print(f"  Avg MAE increase per FP: {fp_damage[fp_damage > 0].mean():.4f}")
    print(f"  Total FP MAE Damage: {fp_damage[fp_damage > 0].sum():.2f}")

print("\n" + "=" * 75)
print(" [6] SUMMARY & CONCLUSIONS")  
print("=" * 75)
print("Analysis complete. Review output above for structural root causes.")
