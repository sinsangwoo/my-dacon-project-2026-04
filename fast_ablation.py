import numpy as np
import os
import json

RUN_ID = "run_20260421_125821"
PRED_DIR = f"outputs/{RUN_ID}/predictions"
PROC_DIR = f"outputs/{RUN_ID}/processed"

# 1. Load Artifacts
print("[LOAD] Loading artifacts...")
oof_stable = np.load(f"{PRED_DIR}/oof_stable.npy")
test_stable = np.load(f"{PRED_DIR}/test_stable.npy")
oof_cat = np.load(f"{PRED_DIR}/oof_cat.npy")
test_cat = np.load(f"{PRED_DIR}/test_cat.npy")
regime_te = np.load(f"{PROC_DIR}/regime_proxy_te.npy")
regime_tr = np.load(f"{PROC_DIR}/regime_proxy_tr.npy")
y_train = np.load(f"{PROC_DIR}/y_train.npy")

# Ground Truth for Fast MAE (10% sample)
np.random.seed(42)
sample_idx = np.random.choice(len(y_train), int(len(y_train) * 0.1), replace=False)
y_val = y_train[sample_idx]
oof_stable_val = oof_stable[sample_idx]
oof_cat_val = oof_cat[sample_idx]
regime_tr_val = regime_tr[sample_idx]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_metrics(preds, y_true=None):
    stats = {
        "mean": float(np.mean(preds)),
        "std": float(np.std(preds)),
        "p90": float(np.quantile(preds, 0.90)),
        "p99": float(np.quantile(preds, 0.99))
    }
    if y_true is not None:
        stats["MAE"] = float(np.mean(np.abs(preds - y_true)))
    return stats

# 2. Simulation Components
# Threshold/Temp for Softening
threshold = float(np.median(regime_tr))
temp = float(np.std(regime_tr)) + 1e-6

# Gating for CatBoost
std_ratio = np.std(test_cat) / (np.std(oof_cat) + 1e-9)
gating_factor = 0.7 if std_ratio > 1.15 else 1.0

# 3. Experiment Versions
versions = {}

# A: Baseline (Hard Routing)
is_extreme_te = (regime_te > 0.5).astype(float)
is_extreme_tr = (regime_tr_val > 0.5).astype(float)
test_A = (is_extreme_te * test_cat) + ((1.0 - is_extreme_te) * test_stable)
oof_A = (is_extreme_tr * oof_cat_val) + ((1.0 - is_extreme_tr) * oof_stable_val)
versions["A: Baseline"] = (test_A, oof_A)

# B: +Soft Routing (Sigmoid)
prob_ext_te = sigmoid((regime_te - threshold) / temp)
prob_ext_tr = sigmoid((regime_tr_val - threshold) / temp)
test_B = (prob_ext_te * test_cat) + ((1.0 - prob_ext_te) * test_stable)
oof_B = (prob_ext_tr * oof_cat_val) + ((1.0 - prob_ext_tr) * oof_stable_val)
versions["B: +Soft"] = (test_B, oof_B)

# C: +Soft + Cat Gating
test_C = (prob_ext_te * (test_cat * gating_factor)) + ((1.0 - prob_ext_te) * test_stable)
# Note: Gating is only applied to test in production if drift is detected. 
# For OOF simulation, we keep it 1.0 or apply the same factor if we want to see impact on bias.
# We'll keep it 1.0 for OOF to represent "no drift" scenario.
oof_C = oof_B 
versions["C: +Soft+Gate"] = (test_C, oof_C)

# D: +Soft + Gate + Pruning Simulation (Stability Penalty 0.9x on extreme)
# Pruning removes high-drift features which were mostly in the extreme regime.
prune_penalty = 0.9
test_D = (prob_ext_te * (test_cat * gating_factor * prune_penalty)) + ((1.0 - prob_ext_te) * test_stable)
oof_D = oof_B
versions["D: +Soft+Gate+Prune"] = (test_D, oof_D)

# E: Optimized Weights (0.4 Cat / 0.6 Stable in Extreme)
test_E = (prob_ext_te * (0.4 * test_cat * gating_factor + 0.6 * test_stable)) + ((1.0 - prob_ext_te) * test_stable)
oof_E = (prob_ext_tr * (0.4 * oof_cat_val + 0.6 * oof_stable_val)) + ((1.0 - prob_ext_tr) * oof_stable_val)
versions["E: Optimized"] = (test_E, oof_E)

# 4. Report
print("\n" + "="*70)
print(f"| version | MAE (approx) | mean | std | p90 | p99 |")
print("-" * 70)
for name, (test_p, oof_p) in versions.items():
    m = get_metrics(test_p, None)
    m_val = get_metrics(oof_p, y_val)
    print(f"| {name:<20} | {m_val['MAE']:<12.4f} | {m['mean']:<6.4f} | {m['std']:<6.4f} | {m['p90']:<6.4f} | {m['p99']:<6.4f} |")
print("="*70)

print(f"\n[INFO] CatBoost Std Ratio: {std_ratio:.4f}")
print(f"[INFO] Gating Applied: {'YES (0.7x)' if gating_factor < 1.0 else 'NO'}")
