import pandas as pd
import numpy as np
import json, os, pickle
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

RUN_ID = "run_20260503_172033"
PROC = f"outputs/{RUN_ID}/processed"
RECON = f"outputs/{RUN_ID}/models/reconstructors"

print("="*80)
print(f" [RANK-BASED TARGETING FORENSIC] RUN: {RUN_ID}")
print("="*80)

# Load base data
train_base = pd.read_pickle(f"{PROC}/train_base.pkl")
y_train = np.load(f"{PROC}/y_train.npy")
oof_raw = np.load(f"outputs/{RUN_ID}/predictions/oof_raw.npy")
oof_base = np.load(f"outputs/{RUN_ID}/predictions/oof_base_raw.npy")

valid = ~np.isnan(oof_raw)
train_v = train_base[valid].reset_index(drop=True)

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

df_f = pd.DataFrame(all_cases)
q90 = np.percentile(df_f['y'], 90)
df_f['is_true_tail'] = (df_f['y'] >= q90).astype(int)

# Rank calculation
df_f['rank_pct'] = df_f['p'].rank(pct=True, ascending=False)
df_f['rank_idx'] = df_f['p'].rank(ascending=False)

# --- TASK 1: RANK-BASED CLASSIFIER RECONSTRUCTION ---
print("\n" + "="*80)
print("1. Rank-Based Classifier Reconstruction")
print("="*80)

n_tails = df_f['is_true_tail'].sum()
print(f"Total True Tails: {n_tails}")

for k_pct in [0.01, 0.03, 0.05, 0.10, 0.20]:
    mask = df_f['rank_pct'] <= k_pct
    recall = df_f.loc[mask, 'is_true_tail'].sum() / n_tails
    prec = df_f.loc[mask, 'is_true_tail'].mean()
    print(f"  [Top {k_pct*100:2.0f}%] Recall: {recall:.4f} | Precision: {prec:.4f}")

tail_ranks = df_f.loc[df_f['is_true_tail'] == 1, 'rank_pct']
print(f"\n[Tail Rank Distribution]")
print(f"  Median Rank: Top {tail_ranks.median()*100:.1f}%")
print(f"  Q25 Rank:    Top {tail_ranks.quantile(0.25)*100:.1f}%")
print(f"  Q75 Rank:    Top {tail_ranks.quantile(0.75)*100:.1f}%")


# --- TASK 2: FEATURE PURIFICATION ---
print("\n" + "="*80)
print("2. Feature Purification Taxonomy")
print("="*80)

pred_shift = np.abs(df_f['preds_t'] * df_f['weight'] + df_f['preds_nt'] * (1 - df_f['weight']) - df_f['preds_nt']) / (df_f['preds_nt'] + 1e-6)
is_pred_tail = df_f['weight'] > 0.3
tp_mask = is_pred_tail & (df_f['is_true_tail'] == 1)
fp_mask = is_pred_tail & (df_f['is_true_tail'] == 0)

num_cols = train_v.select_dtypes(include=np.number).columns
num_cols = [c for c in num_cols if c not in ['ID', 'scenario_id', 'fold']]

tax_results = []
for col in num_cols:
    v = train_v[col].fillna(0).values
    try: auc = roc_auc_score(df_f['is_true_tail'], v)
    except: auc = 0.5
    
    v_tp = v[tp_mask]
    v_fp = v[fp_mask]
    ks = ks_2samp(v_tp, v_fp).statistic if len(v_tp)>5 and len(v_fp)>5 else 0
    
    if auc > 0.6 and ks < 0.05: cat = "POISON (Misleading)"
    elif auc > 0.6 and ks >= 0.1: cat = "TRUE_SIGNAL"
    else: cat = "NOISE/WEAK"
    tax_results.append((col, auc, ks, cat))

tax_df = pd.DataFrame(tax_results, columns=['Feature', 'Tail_AUC', 'KS_TP_FP', 'Category'])
print("\n[POISON FEATURES TO QUARANTINE] (AUC > 0.6, KS < 0.05)")
poison = tax_df[tax_df['Category'] == 'POISON (Misleading)'].sort_values('Tail_AUC', ascending=False).head(10)
for _, r in poison.iterrows():
    print(f"  {r['Feature']:<40} | AUC: {r['Tail_AUC']:.3f} | KS: {r['KS_TP_FP']:.3f}")

print("\n[TRUE SIGNALS] (AUC > 0.6, KS >= 0.1)")
true_sig = tax_df[tax_df['Category'] == 'TRUE_SIGNAL'].sort_values('KS_TP_FP', ascending=False).head(10)
for _, r in true_sig.iterrows():
    print(f"  {r['Feature']:<40} | AUC: {r['Tail_AUC']:.3f} | KS: {r['KS_TP_FP']:.3f}")


# --- TASK 3: CAPACITY-AWARE FEATURE INJECTION ---
print("\n" + "="*80)
print("3. Capacity-Aware Feature Injection")
print("="*80)

cap_feats = {}
if 'congestion_score' in train_v and 'robot_active' in train_v: cap_feats['congestion_per_active'] = train_v['congestion_score'] / (train_v['robot_active'] + 1e-6)
if 'max_zone_density' in train_v and 'robot_idle' in train_v: cap_feats['density_per_idle'] = train_v['max_zone_density'] / (train_v['robot_idle'] + 1e-6)
if 'battery_std' in train_v and 'robot_idle' in train_v: cap_feats['battery_stress_idle'] = train_v['battery_std'] / (train_v['robot_idle'] + 1e-6)
if 'order_inflow_15m' in train_v and 'robot_active' in train_v: cap_feats['inflow_per_active'] = train_v['order_inflow_15m'] / (train_v['robot_active'] + 1e-6)

print(f"{'Feature':<30} | {'AUC':>6} | {'KS(TP-FP)':>9} | {'P-Val':>8}")
print("-" * 60)
for name, v in cap_feats.items():
    v = v.fillna(0).replace([np.inf, -np.inf], 0).values
    try: auc = roc_auc_score(df_f['is_true_tail'], v)
    except: auc = 0.5
    v_tp = v[tp_mask]
    v_fp = v[fp_mask]
    if len(v_tp)>5 and len(v_fp)>5:
        ks, pval = ks_2samp(v_tp, v_fp)
    else: ks, pval = 0, 1
    print(f"{name:<30} | {auc:.4f} | {ks:.4f} | {pval:.4f}")


# --- TASK 4: GATING SYSTEM DESTRUCTION & REBUILD ---
print("\n" + "="*80)
print("4. Gating Rebuild: Rank vs Hybrid")
print("="*80)

# Simulate Pure Rank Gating
k_rank = 0.05  # Top 5%
rank_mask = df_f['rank_pct'] <= k_rank

# Damage calculation
def calc_damage(mask):
    pred = df_f['preds_t'] * mask + df_f['preds_nt'] * (~mask)
    err_base = np.abs(df_f['preds_nt'] - df_f['y'])
    err_pred = np.abs(pred - df_f['y'])
    fp_dmg = np.sum(np.maximum(0, err_pred - err_base))
    gain = np.sum(np.maximum(0, err_base - err_pred))
    tail_mae = np.mean(err_pred[df_f['is_true_tail']==1])
    return fp_dmg, gain, tail_mae

fp_dmg_opt_a, gain_opt_a, tail_mae_opt_a = calc_damage(rank_mask)
print(f"[OPTION A: PURE RANK GATING (Top 5%)]")
print(f"  Passed: {rank_mask.sum()} | Prec: {df_f.loc[rank_mask, 'is_true_tail'].mean():.4f}")
print(f"  FP Damage: {fp_dmg_opt_a:.2f} | Gain Capture: {gain_opt_a:.2f} | Tail MAE: {tail_mae_opt_a:.2f}")

# Simulate Hybrid Gating (Rank Top 10% + Regressor Consensus > 1.5)
# Note: Regressor is bad, but let's test the hybrid logic
hybrid_mask = (df_f['rank_pct'] <= 0.10) & (df_f['preds_t'] > df_f['preds_nt'] * 1.5)
fp_dmg_opt_b, gain_opt_b, tail_mae_opt_b = calc_damage(hybrid_mask)
print(f"\n[OPTION B: HYBRID GATING (Top 10% Rank + Regressor > 1.5)]")
print(f"  Passed: {hybrid_mask.sum()} | Prec: {df_f.loc[hybrid_mask, 'is_true_tail'].mean():.4f}")
print(f"  FP Damage: {fp_dmg_opt_b:.2f} | Gain Capture: {gain_opt_b:.2f} | Tail MAE: {tail_mae_opt_b:.2f}")


# --- TASK 5: DEEP FP ROOT CAUSE TRACE ---
print("\n" + "="*80)
print("5. Deep FP Root Cause Trace")
print("="*80)

deep_fp = fp_mask & (df_f['y'] < 0.3 * q90)
print(f"Deep FP Count: {deep_fp.sum()}")

print("[Deep FP Average Profile vs TRUE POSITIVE]")
compare_cols = ['robot_idle', 'order_inflow_15m', 'congestion_score']
for c in compare_cols:
    if c in train_v.columns:
        fp_v = train_v.loc[deep_fp, c].mean()
        tp_v = train_v.loc[tp_mask, c].mean()
        print(f"  {c:<20}: Deep FP = {fp_v:7.2f} | TP = {tp_v:7.2f}")

print("\n[Deep FP Rank Distribution]")
deep_fp_ranks = df_f.loc[deep_fp, 'rank_pct']
print(f"  Median Rank: Top {deep_fp_ranks.median()*100:.1f}%")
print(f"  Q25 Rank:    Top {deep_fp_ranks.quantile(0.25)*100:.1f}%")
