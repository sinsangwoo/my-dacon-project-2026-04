import json
import pandas as pd
import numpy as np

run_id = "run_20260427_023656"
log_path = f"outputs/{run_id}/processed/signal_validation_logs.json"

with open(log_path, 'r') as f:
    data = json.load(f)

val_logs = data['val_logs']
noise_proof = data['noise_proof']

df = pd.DataFrame(val_logs)

# 1. Noise Gate Audit
max_noise_gain = noise_proof['noise_max_gain']
# In current code: beats_noise_gain = gain > max_noise_gain * 0.5
# In current code: beats_noise_perm = p_data['delta'] > max(max_noise_perm * 2.0, 0.0005)
# Wait, max_noise_perm is not explicitly in the root but I can find it from logs if I look at __noise_ features
# But the logs I have are only for candidates. 
# Wait, let's see if __noise_ features are in val_logs.
noise_entries = [v for v in val_logs if v['feature'].startswith('__noise_')]
if noise_entries:
    max_noise_perm = max([v['perm_delta'] for v in noise_entries])
else:
    # Estimate from beats_noise_perm logic
    max_noise_perm = 0 # Fallback

# Features that pass ONLY via gain
pass_only_gain = df[(df['beats_noise_gain'] == True) & (df['beats_noise_perm'] == False)]
# Features that pass ONLY via perm
pass_only_perm = df[(df['beats_noise_gain'] == False) & (df['beats_noise_perm'] == True)]
# Features that pass BOTH
pass_both = df[(df['beats_noise_gain'] == True) & (df['beats_noise_perm'] == True)]

print(f"--- NOISE GATE AUDIT ---")
print(f"Total Evaluated: {len(df)}")
print(f"Pass Both: {len(pass_both)}")
print(f"Pass Only Gain: {len(pass_only_gain)}")
print(f"Pass Only Perm: {len(pass_only_perm)}")

# 2. Signal Gate Audit (perm_delta > 0)
small_positive = df[(df['perm_delta'] > 0) & (df['perm_delta'] < 0.0005)]
print(f"\n--- SIGNAL GATE AUDIT ---")
print(f"Features with small positive perm_delta (< 0.0005): {len(small_positive)}")

# 3. KS Audit
high_ks = df[df['ks_stat'] > 0.5]
v_high_ks = df[df['ks_stat'] > 0.8]
print(f"\n--- KS AUDIT ---")
print(f"KS > 0.5: {len(high_ks)}")
print(f"KS > 0.8: {len(v_high_ks)}")

# 4. Generalization Audit (Stability Factor)
low_stability = df[df['sign_stability'] < 1.0] # Not all folds have same sign
print(f"\n--- GENERALIZATION AUDIT ---")
print(f"Features with sign instability (sign_stability < 1.0): {len(low_stability)}")

# Sample of questionable features (Passed but weak/unstable)
questionable = df[(df['passed'] == True) & ((df['beats_noise_gain'] == False) | (df['sign_stability'] < 1.0) | (df['ks_stat'] > 0.5))]
print(f"\n--- QUESTIONABLE SURVIVORS ---")
print(questionable[['feature', 'gain', 'perm_delta', 'ks_stat', 'sign_stability']].head(20))
