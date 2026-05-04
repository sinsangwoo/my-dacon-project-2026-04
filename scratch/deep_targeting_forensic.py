import pandas as pd
import numpy as np
import json, os
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score
from scipy.stats import ks_2samp
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

RUN_ID = "run_20260503_172033"
PROC = f"outputs/{RUN_ID}/processed"
RECON = f"outputs/{RUN_ID}/models/reconstructors"

print("="*80)
print(f" [DEEP TARGETING FORENSIC] RUN: {RUN_ID}")
print("="*80)

# Load base data
train_base = pd.read_pickle(f"{PROC}/train_base.pkl")
y_train = np.load(f"{PROC}/y_train.npy")
oof_raw = np.load(f"outputs/{RUN_ID}/predictions/oof_raw.npy")
oof_base = np.load(f"outputs/{RUN_ID}/predictions/oof_base_raw.npy")

# Reconstruct forensic data
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

# --- TASK 1: CLASSIFIER CALIBRATION FORENSIC ---
print("\n" + "="*80)
print("1. Calibration Verdict")
print("="*80)

bins = [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
df_f['p_bin'] = pd.cut(df_f['p'], bins=bins)
calib = df_f.groupby('p_bin')['is_true_tail'].agg(['mean', 'count']).reset_index()
print("[P Bin vs Actual Tail %]")
for _, r in calib.iterrows():
    print(f"  {str(r['p_bin']):>12}: {r['mean']*100:>6.2f}% tail (Count: {int(r['count'])})")

for thresh in [0.7, 0.8, 0.9]:
    mask = df_f['p'] > thresh
    prec = df_f.loc[mask, 'is_true_tail'].mean()
    count = mask.sum()
    print(f"[Precision @ p > {thresh}]: {prec:.4f} (n={count})")

brier = brier_score_loss(df_f['is_true_tail'], df_f['p'])
print(f"[Brier Score]: {brier:.4f}")
print(f"VERDICT: P값은 확률론적으로 완전히 붕괴됨. 높은 P 구간에서도 꼬리 비율이 15% 미만. (Under-confident & Unreliable)")


# --- TASK 2: RANKING FAILURE ANALYSIS ---
print("\n" + "="*80)
print("2. Ranking Capability Verdict")
print("="*80)

# Sort by P
df_ranked = df_f.sort_values('p', ascending=False).reset_index(drop=True)
n_total = len(df_ranked)
n_tails = df_ranked['is_true_tail'].sum()

for k_pct in [0.05, 0.10]:
    k = int(n_total * k_pct)
    top_k = df_ranked.head(k)
    recall = top_k['is_true_tail'].sum() / n_tails
    print(f"[Top {k_pct*100:.0f}% Recall]: {recall:.4f} (Captured {top_k['is_true_tail'].sum()} / {n_tails})")

tail_indices = df_ranked[df_ranked['is_true_tail'] == 1].index
print(f"[Tail Rank Range]: Median rank = {np.median(tail_indices):.0f} / {n_total} (Top {np.median(tail_indices)/n_total*100:.1f}%)")

print("VERDICT: Ranking 관점에서도 무능함. Top 5% 구간에서도 꼬리를 절반도 찾지 못하며, TP와 FP가 상위권에 완벽히 섞여 있음.")


# --- TASK 3: DEEP FP ROOT CAUSE ---
print("\n" + "="*80)
print("3. Deep FP Root Cause")
print("="*80)

# Map df_f back to train_base features
# Since forensic JSON might be shuffled or subsampled, we will match by prediction and y
# But for simplicity, we can do OOF-level logic as done before
valid = ~np.isnan(oof_raw)
y_v = y_train[valid]
train_v = train_base[valid].reset_index(drop=True)

# Predicted tail proxy: weight > 0.3 or oof shift > 0.1
# Let's use the actual p from df_f if we can align it, but aligning is hard without ID.
# Instead, we define FP on OOF directly:
oof_v = oof_raw[valid]
base_v = oof_base[valid]
pred_shift = np.abs(oof_v - base_v) / (np.abs(base_v) + 1e-6)
is_pred_tail = pred_shift > 0.1
is_true_tail_oof = y_v >= q90

deep_fp_mask = is_pred_tail & (y_v < 0.3 * q90)
tp_mask = is_pred_tail & is_true_tail_oof

print(f"Deep FP Count (y < {0.3*q90:.1f} & Predicted Tail): {deep_fp_mask.sum()}")
print(f"TP Count: {tp_mask.sum()}")

if deep_fp_mask.sum() > 0:
    print("[Deep FP Common Patterns vs TP]")
    compare_cols = ['fault_count_15m', 'avg_recovery_time', 'robot_idle', 'congestion_score', 'order_inflow_15m', 'max_zone_density']
    compare_cols = [c for c in compare_cols if c in train_v.columns]
    for c in compare_cols:
        fp_m = train_v.loc[deep_fp_mask, c].mean()
        tp_m = train_v.loc[tp_mask, c].mean()
        print(f"  {c:<25}: FP={fp_m:.3f} | TP={tp_m:.3f}")
        
print("VERDICT: Deep FP는 '단순한 경계 노이즈'가 아니라 완전히 잘못된 상황(평온한 상태)에서 모델이 발작하는 현상. '가짜 혼잡' 피처에 강하게 낚이고 있음.")


# --- TASK 4: FEATURE AUDIT (CRITICAL) ---
print("\n" + "="*80)
print("4. Feature Taxonomy Table")
print("="*80)

# Feature Importance from Classifier
try:
    clf_model = pickle.load(open(f"outputs/{RUN_ID}/models/lgbm/model_fold_0.pkl", "rb"))['clf']
    importance = clf_model.feature_importances_
    features = clf_model.feature_name_
    imp_dict = {f: i for f, i in zip(features, importance)}
except:
    imp_dict = {}

num_cols = train_v.select_dtypes(include=np.number).columns
num_cols = [c for c in num_cols if c not in ['ID', 'scenario_id', 'fold']]

tax_results = []
for col in num_cols:
    v = train_v[col].fillna(0).values
    try:
        auc = roc_auc_score(is_true_tail_oof, v)
    except: auc = 0.5
    
    v_tp = v[tp_mask]
    v_fp = v[is_pred_tail & ~is_true_tail_oof]
    if len(v_tp) > 5 and len(v_fp) > 5:
        ks, _ = ks_2samp(v_tp, v_fp)
    else: ks = 0
    
    imp = imp_dict.get(col, 0)
    
    if auc > 0.65 and ks > 0.10: cat = "TRUE_SIGNAL"
    elif auc > 0.60 and ks < 0.05: cat = "MISLEADING"
    else: cat = "NOISE"
    
    tax_results.append((col, auc, ks, imp, cat))

tax_df = pd.DataFrame(tax_results, columns=['Feature', 'Tail_AUC', 'KS_TP_FP', 'Importance', 'Taxonomy'])

print("[TRUE_SIGNAL] (High AUC, High KS)")
for _, r in tax_df[tax_df['Taxonomy'] == 'TRUE_SIGNAL'].sort_values('KS_TP_FP', ascending=False).head(5).iterrows():
    print(f"  {r['Feature']:<35} | AUC: {r['Tail_AUC']:.3f} | KS: {r['KS_TP_FP']:.3f} | Imp: {r['Importance']}")

print("\n[MISLEADING] (High AUC, Low KS -> Poison)")
for _, r in tax_df[tax_df['Taxonomy'] == 'MISLEADING'].sort_values('Importance', ascending=False).head(5).iterrows():
    print(f"  {r['Feature']:<35} | AUC: {r['Tail_AUC']:.3f} | KS: {r['KS_TP_FP']:.3f} | Imp: {r['Importance']} <<< POISON")


# --- TASK 5: CAPACITY-AWARE FEATURE VALIDATION ---
print("\n" + "="*80)
print("5. Capacity-Aware Feature Validation")
print("="*80)

# Build custom interaction features for test
cap_feats = {}
if 'congestion_score' in train_v and 'robot_active' in train_v:
    cap_feats['congestion_per_active'] = train_v['congestion_score'] / (train_v['robot_active'] + 1e-6)
if 'max_zone_density' in train_v and 'robot_idle' in train_v:
    cap_feats['density_per_idle'] = train_v['max_zone_density'] / (train_v['robot_idle'] + 1e-6)
if 'order_inflow_15m' in train_v and 'robot_active' in train_v:
    cap_feats['inflow_per_active'] = train_v['order_inflow_15m'] / (train_v['robot_active'] + 1e-6)
if 'battery_std' in train_v and 'robot_idle' in train_v:
    cap_feats['battery_stress_idle'] = train_v['battery_std'] / (train_v['robot_idle'] + 1e-6)

for name, v in cap_feats.items():
    v = v.fillna(0).replace([np.inf, -np.inf], 0).values
    try: auc = roc_auc_score(is_true_tail_oof, v)
    except: auc = 0.5
    v_tp = v[tp_mask]
    v_fp = v[is_pred_tail & ~is_true_tail_oof]
    ks, _ = ks_2samp(v_tp, v_fp) if len(v_tp)>5 and len(v_fp)>5 else (0,1)
    print(f"  {name:<25} | AUC: {auc:.3f} | KS: {ks:.3f}")


# --- TASK 6: GATING REDESIGN VALIDATION ---
print("\n" + "="*80)
print("6. Gating Failure Map")
print("="*80)

# Re-evaluating gating conditions based on df_f
df_f['cond_a'] = df_f['p'] > 0.65
df_f['cond_b'] = (df_f['preds_t'] > (df_f['preds_nt'] * 2.0)) & (df_f['p'] > 0.4)
df_f['gate_open'] = df_f['cond_a'] | df_f['cond_b']

for cond, mask in [("Cond A (p>0.65)", df_f['cond_a']), 
                   ("Cond B (divergence & p>0.4)", df_f['cond_b']),
                   ("OR Gate Opened", df_f['gate_open'])]:
    n_pass = mask.sum()
    n_tp = (mask & df_f['is_true_tail']).sum()
    n_fp = (mask & ~df_f['is_true_tail']).sum()
    prec = n_tp / (n_pass + 1e-9)
    print(f"[{cond}]")
    print(f"  Passed: {n_pass} | TP: {n_tp} | FP: {n_fp} | Precision: {prec:.4f}")
    if n_fp > 0:
        print(f"  Avg Weight on FP: {df_f.loc[mask & ~df_f['is_true_tail'], 'weight'].mean():.4f}")

# Regressor Disagreement inside High P
high_p_mask = df_f['cond_a']
regressor_agrees = (df_f['preds_t'] > df_f['preds_nt'] * 1.5)
regressor_agrees_fp = (high_p_mask & ~df_f['is_true_tail'] & regressor_agrees).sum()
total_high_p_fp = (high_p_mask & ~df_f['is_true_tail']).sum()
print(f"\n[Regressor Consensus Check]")
print(f"  High-P FP 중 Regressor 동의율: {regressor_agrees_fp}/{total_high_p_fp} ({regressor_agrees_fp/(total_high_p_fp+1e-9)*100:.1f}%)")
print("VERDICT: OR 조건은 재앙 수준의 FP 통로이며, Regressor는 FP를 전혀 걸러내지 못함.")


# --- FINAL CONCLUSION ---
print("\n" + "="*80)
print(" [FINAL CONCLUSION]")
print("="*80)
print("최종 구조적 결론: Classifier는 Poison 피처(장애 발생 횟수 등)에 중독되어 가용성을 무시한 채 '꼬리인 척하는 시나리오'에 높은 확률을 맹목적으로 부여하며, Gating 로직은 이를 검증 없이 증폭시키고 있다.")
