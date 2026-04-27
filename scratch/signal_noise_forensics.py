import pandas as pd
import numpy as np
import json
import os
import sys
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

def run_forensics(run_id):
    print(f"=== [SIGNAL NOISE FORENSICS: {run_id}] ===")
    
    # 1. Load Data
    data_dir = f"outputs/{run_id}/processed"
    train_df = pd.read_pickle(f"{data_dir}/train_base.pkl")
    y = np.load(f"{data_dir}/y_train.npy")
    
    with open(f"{data_dir}/signal_validation_logs.json", 'r') as f:
        val_data = json.load(f)
    val_logs = pd.DataFrame(val_data['val_logs'])
    
    # 2. Identify the 136 features (Relaxed Logic)
    # We re-apply the relaxation logic to define our current 'Selection'
    gains = val_logs['gain'].values
    min_gain_threshold = np.mean(gains) * 0.05
    
    def is_passed(r):
        gain = r['gain']
        ks = r['ks_stat']
        marg_corr = r['marg_corr']
        perm_delta = r['perm_delta']
        splits = r['splits']
        avg_depth = r['avg_depth']
        
        # Tier 2 Relaxed Rules
        f1 = gain > min_gain_threshold
        ratio_threshold = 0.0001 if not (0.5 < ks <= 0.85) else 0.0002
        f2 = perm_delta > 0 and (perm_delta / (gain + 1e-9)) > ratio_threshold
        f3 = abs(marg_corr) >= 0.001
        f4 = splits >= 1
        f5 = avg_depth < 12
        f6 = ks <= 0.85
        return all([f1, f2, f3, f4, f5, f6])

    val_logs['relaxed_passed'] = val_logs.apply(is_passed, axis=1)
    current_selection = val_logs[val_logs['relaxed_passed']]['feature'].tolist()
    print(f"Current Selection Size: {len(current_selection)}")

    # 3. TASK 1: SIGNAL VS NOISE RATIO ESTIMATION
    print("\n--- TASK 1: SIGNAL VS NOISE RATIO ---")
    
    def evaluate_set(features):
        if not features: return 999.0
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        maes = []
        for tr_idx, val_idx in kf.split(train_df):
            X_tr, X_val = train_df.iloc[tr_idx][features], train_df.iloc[val_idx][features]
            y_tr, y_val = y[tr_idx], y[val_idx]
            model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
            model.fit(X_tr, y_tr)
            maes.append(mean_absolute_error(y_val, model.predict(X_val)))
        return np.mean(maes)

    baseline_mae = evaluate_set(current_selection)
    print(f"Baseline MAE (136 feats): {baseline_mae:.6f}")

    results_subset = []
    for pct in [0.1, 0.2, 0.3]:
        subset_maes = []
        for seed in range(3):
            np.random.seed(seed)
            keep_count = int(len(current_selection) * (1 - pct))
            subset = np.random.choice(current_selection, keep_count, replace=False).tolist()
            subset_maes.append(evaluate_set(subset))
        results_subset.append({'drop_pct': pct, 'mae': np.mean(subset_maes), 'degradation': np.mean(subset_maes) - baseline_mae})

    for r in results_subset:
        print(f"Drop {r['drop_pct']*100:.0f}%: MAE {r['mae']:.6f} (Degradation: {r['degradation']:.6f})")

    # TASK 2: GENERALIZATION FAILURE DETECTION
    print("\n--- TASK 2: GENERALIZATION VALIDATION ---")
    # Corr(Gain, Permutation)
    corr_gp = val_logs['gain'].corr(val_logs['perm_delta'])
    print(f"Correlation (Gain, Permutation): {corr_gp:.4f}")
    
    # Suspicious features: High Gain, but very low Permutation relative to gain
    val_logs['ratio'] = val_logs['perm_delta'] / (val_logs['gain'] + 1e-9)
    suspicious = val_logs[val_logs['relaxed_passed'] & (val_logs['ratio'] < val_logs['ratio'].median())].sort_values('gain', ascending=False).head(10)
    print("\nTop 10 Suspicious (High Gain, Low Ratio) features:")
    print(suspicious[['feature', 'gain', 'perm_delta', 'ratio']])

    # TASK 3: GAIN FLOOR VALIDATION
    print("\n--- TASK 3: GAIN FLOOR VALIDATION ---")
    g_sorted = np.sort(val_logs['gain'].values)[::-1]
    g_mean = np.mean(g_sorted)
    print(f"Gain Stats: Mean={g_mean:.2f}, Median={np.median(g_sorted):.2f}, Std={np.std(g_sorted):.2f}")
    
    # Analyze MAE contribution by Gain bin
    val_logs['gain_bin'] = pd.qcut(val_logs['gain'], 10, labels=False, duplicates='drop')
    bin_impact = val_logs.groupby('gain_bin')['perm_delta'].mean()
    print("\nAvg Permutation Impact by Gain Decile (0=lowest):")
    print(bin_impact)

    # TASK 6: DATA-DRIVEN THRESHOLD LEARNING
    print("\n--- TASK 6: LEARNED THRESHOLDS ---")
    # Identify the 'Noise Floor' for Gain
    # We injected noise in SignalValidator (logs usually contain them if we didn't filter them out)
    noise_feats = val_logs[val_logs['feature'].str.contains('__noise_')]
    if not noise_feats.empty:
        max_noise_gain = noise_feats['gain'].max()
        max_noise_perm = noise_feats['perm_delta'].max()
        print(f"Empirical Noise Ceiling (Gain): {max_noise_gain}")
        print(f"Empirical Noise Ceiling (Permutation): {max_noise_perm}")
        
        learned_gain_floor = max_noise_gain * 1.5
        learned_perm_floor = max(max_noise_perm, 0) * 1.2
    else:
        # Fallback to percentile logic
        learned_gain_floor = np.percentile(val_logs['gain'], 20)
        learned_perm_floor = 0.0001
        print("Noise features missing from logs. Using percentile-based floor.")

    print(f"Learned Gain Floor: {learned_gain_floor:.4f}")
    print(f"Learned Permutation Floor: {learned_perm_floor:.4f}")

if __name__ == "__main__":
    run_forensics('run_20260426_235007')
