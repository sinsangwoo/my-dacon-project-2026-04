import pandas as pd
import numpy as np
import json
import os
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

def run_deep_audit(run_id):
    print(f"=== [DEEP SIGNAL AUDIT: {run_id}] ===")
    data_dir = f"outputs/{run_id}/processed"
    train_df = pd.read_pickle(f"{data_dir}/train_base.pkl")
    y = np.load(f"{data_dir}/y_train.npy")
    with open(f"{data_dir}/signal_validation_logs.json", 'r') as f:
        val_data = json.load(f)
    val_logs = pd.DataFrame(val_data['val_logs'])
    
    # 1. Re-calculate survived set (Tier 2)
    gains = val_logs['gain'].values
    min_gain_threshold = np.mean(gains) * 0.05
    def is_passed(r):
        gain = r['gain']
        ks = r['ks_stat']
        marg_corr = r['marg_corr']
        perm_delta = r['perm_delta']
        f1 = gain > min_gain_threshold
        f2 = perm_delta > 0.0001 # Slightly stricter for this audit
        f3 = abs(marg_corr) >= 0.001
        return all([f1, f2, f3])
    val_logs['passed_relaxed'] = val_logs.apply(is_passed, axis=1)
    survived = val_logs[val_logs['passed_relaxed']]['feature'].tolist()
    print(f"Survived features: {len(survived)}")

    # 2. Interaction Strength (Interaction = Gain - Marginal Power?)
    # A better proxy: High Gain / Low Correlation
    val_logs['interaction_proxy'] = val_logs['gain'] / (abs(val_logs['marg_corr']) + 1e-9)
    top_interactions = val_logs[val_logs['passed_relaxed']].sort_values('interaction_proxy', ascending=False).head(10)
    print("\nTop 10 Interaction-Driven Features (High Gain, Low Corr):")
    print(top_interactions[['feature', 'gain', 'marg_corr', 'perm_delta']])

    # 3. Stability Check (Sign Stability and Gain CV)
    # We need to simulate consistency since it's not in the flat log
    # But wait, SignalValidator does 3-fold.
    # For now, we use perm_delta as the primary signal.

    # 4. DATA-DRIVEN THRESHOLD: The 'Elbow' of the gain curve
    g_sorted = np.sort(val_logs['gain'].values)[::-1]
    # Simple elbow: where the rate of change drops
    diffs = np.diff(g_sorted)
    elbow_idx = np.argmin(diffs) # largest drop
    elbow_gain = g_sorted[elbow_idx]
    print(f"\nGain Elbow detected at: {elbow_gain:.2f}")

    # 5. NOISE FLOOR from Proof metrics
    noise_max_gain = val_data['noise_proof']['noise_max_gain']
    print(f"Noise Proof Max Gain: {noise_max_gain}")
    
    learned_gain_threshold = max(elbow_gain * 0.5, noise_max_gain * 0.5) # Balanced
    print(f"Proposed Learned Gain Threshold: {learned_gain_threshold:.2f}")

if __name__ == "__main__":
    run_deep_audit('run_20260426_235007')
