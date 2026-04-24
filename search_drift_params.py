import numpy as np
import os
import json
import pandas as pd
from itertools import product
import random

RUN_ID = "run_20260421_125821"
PRED_DIR = f"outputs/{RUN_ID}/predictions"
PROC_DIR = f"outputs/{RUN_ID}/processed"

# 1. Load Artifacts
print("[LOAD] Loading artifacts for parametric search...")
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

# Target Stats (from train)
train_mean = float(np.mean(y_train))
train_p99 = float(np.quantile(y_train, 0.99))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def simulate_blend(params, test_stable, test_cat, regime_te, oof_stable_v, oof_cat_v, regime_tr_v):
    # Parameters
    th_scale = params['th_scale']
    temp_scale = params['temp_scale']
    ext_penalty = params['ext_penalty']
    cat_weight = params['cat_weight']
    
    # Base Stats
    base_th = np.median(regime_tr)
    base_temp = np.std(regime_tr) + 1e-6
    
    # Tuned Stats
    th = base_th * th_scale
    temp = base_temp * temp_scale
    
    # Probabilities
    prob_ext_te = sigmoid((regime_te - th) / temp)
    prob_ext_tr = sigmoid((regime_tr_v - th) / temp)
    
    # Blending (Extreme Regime)
    # final_ext = cat_weight * cat + (1 - cat_weight) * stable
    test_ext = (cat_weight * test_cat + (1.0 - cat_weight) * test_stable) * ext_penalty
    oof_ext = (cat_weight * oof_cat_v + (1.0 - cat_weight) * oof_stable_v) # No penalty for OOF (no drift)
    
    # Final Combine
    test_final = (prob_ext_te * test_ext) + ((1.0 - prob_ext_te) * test_stable)
    oof_final = (prob_ext_tr * oof_ext) + ((1.0 - prob_ext_tr) * oof_stable_v)
    
    return test_final, oof_final

def calculate_score(test_p, oof_p, y_true):
    mae = np.mean(np.abs(oof_p - y_true))
    mean_p = np.mean(test_p)
    p99_p = np.quantile(test_p, 0.99)
    
    # Composite Score: MAE + Mean Alignment + P99 Preservation
    score = mae + 0.1 * abs(mean_p - train_mean) + 0.05 * abs(p99_p - train_p99)
    return {
        "score": score,
        "MAE": mae,
        "mean": mean_p,
        "p99": p99_p,
        "std": np.std(test_p)
    }

# 2. Search Space
search_space = {
    "th_scale": [0.8, 0.9, 1.0, 1.1, 1.2],
    "temp_scale": [0.5, 0.75, 1.0, 1.5, 2.0],
    "ext_penalty": [0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
    "cat_weight": [0.2, 0.3, 0.4, 0.5, 0.6]
}

# 3. Execution
print("[SEARCH] Starting Parametric Search (50 iterations)...")
all_results = []
random.seed(42)

# Generate 50 random configs
configs = []
for _ in range(50):
    config = {k: random.choice(v) for k, v in search_space.items()}
    if config not in configs:
        configs.append(config)

for cfg in configs:
    test_p, oof_p = simulate_blend(cfg, test_stable, test_cat, regime_te, oof_stable_val, oof_cat_val, regime_tr_val)
    metrics = calculate_score(test_p, oof_p, y_val)
    all_results.append({**cfg, **metrics})

# 4. Results Processing
df_results = pd.DataFrame(all_results).sort_values("score")

print("\n" + "="*80)
print(f"| rank | MAE | mean | p99 | params |")
print("-" * 80)
for i, row in df_results.head(5).iterrows():
    params_str = f"th:{row['th_scale']}, temp:{row['temp_scale']}, pen:{row['ext_penalty']}, cat:{row['cat_weight']}"
    print(f"| {i+1:<4} | {row['MAE']:<6.4f} | {row['mean']:<6.4f} | {row['p99']:<6.4f} | {params_str} |")
print("="*80)

# 5. Sensitivity Analysis
print("\n[ANALYSIS] Parameter Sensitivity (Correlation with Score):")
corrs = df_results[["th_scale", "temp_scale", "ext_penalty", "cat_weight", "score"]].corr()["score"].drop("score")
for param, corr in corrs.items():
    print(f"  * {param:<15}: {corr:.4f} ({'Sensitive' if abs(corr) > 0.3 else 'Low Impact'})")

best_cfg = df_results.iloc[0].to_dict()
print(f"\n[BEST] Optimal Config found with Score: {best_cfg['score']:.4f}")
