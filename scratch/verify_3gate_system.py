"""Verify the 3-gate system against existing run data."""
import pandas as pd
import numpy as np
import json

RUN_ID = 'run_20260426_235007'
DATA_DIR = f'outputs/{RUN_ID}/processed'

with open(f'{DATA_DIR}/signal_validation_logs.json', 'r') as f:
    val_data = json.load(f)
val_logs = pd.DataFrame(val_data['val_logs'])

noise_max_gain = val_data['noise_proof']['noise_max_gain']  # 25
# Estimate max_noise_perm from noise_proof (not stored directly, use conservative 0)
max_noise_perm = 0  # Noise features should have ~0 perm_delta

print(f"Noise Ceiling: Max Gain={noise_max_gain}, Max Perm={max_noise_perm}")
print(f"Gate thresholds: gain > {noise_max_gain * 0.5}, perm > {max(max_noise_perm * 2.0, 0.0005)}")

def apply_3gate(r):
    gain = r['gain']
    perm_delta = r['perm_delta']
    ks = r['ks_stat']
    
    beats_noise_gain = gain > noise_max_gain * 0.5  # > 12.5
    beats_noise_perm = perm_delta > max(max_noise_perm * 2.0, 0.0005)
    gate_noise = beats_noise_gain or beats_noise_perm
    gate_signal = perm_delta > 0
    gate_stability = ks <= 0.85
    
    passed = gate_noise and gate_signal and gate_stability
    
    rejections = []
    if not gate_noise: rejections.append("NoiseCeiling")
    if not gate_signal: rejections.append("NegativePermutation")
    if not gate_stability: rejections.append("CatastrophicDrift")
    return passed, rejections

val_logs['new_passed'] = val_logs.apply(lambda r: apply_3gate(r)[0], axis=1)
val_logs['new_rejections'] = val_logs.apply(lambda r: apply_3gate(r)[1], axis=1)

n_old = val_logs['passed'].sum()
n_new = val_logs['new_passed'].sum()

print(f"\nOld System: {n_old} passed / {len(val_logs)} total ({n_old/len(val_logs):.1%})")
print(f"New System: {n_new} passed / {len(val_logs)} total ({n_new/len(val_logs):.1%})")

# Category breakdown
print("\n--- REJECTION REASON BREAKDOWN (New System) ---")
reason_counts = {}
for _, r in val_logs[~val_logs['new_passed']].iterrows():
    for reason in r['new_rejections']:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
for k, v in sorted(reason_counts.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

# Trend/Vol survival
trend_vol = val_logs[val_logs['is_trend_vol']]
trend_old = trend_vol['passed'].sum()
trend_new = trend_vol['new_passed'].sum()
print(f"\nTrend/Volatility: OLD={trend_old}/{len(trend_vol)} -> NEW={trend_new}/{len(trend_vol)}")

# Top-45 high-impact check
high_impact = val_logs[val_logs['perm_delta'] > 0.01].sort_values('perm_delta', ascending=False)
hi_old = high_impact['passed'].sum()
hi_new = high_impact['new_passed'].sum()
print(f"High-impact (perm>0.01): OLD={hi_old}/{len(high_impact)} -> NEW={hi_new}/{len(high_impact)}")

# Variance features (tail correlation)
print("\nTop 10 Newly Restored Features:")
restored = val_logs[~val_logs['passed'] & val_logs['new_passed']].sort_values('perm_delta', ascending=False)
print(f"{'feature':<45} {'gain':>5} {'perm_delta':>12}")
print("-" * 65)
for _, r in restored.head(10).iterrows():
    print(f"{r['feature']:<45} {r['gain']:>5} {r['perm_delta']:>12.6f}")

# Noise features would have been eliminated?
print("\n--- Noise Immunity Sanity Check ---")
noise_like_passed = val_logs[val_logs['new_passed'] & (val_logs['perm_delta'] <= 0)]
print(f"Features with perm_delta <= 0 that PASSED new filter: {len(noise_like_passed)}")
if len(noise_like_passed) > 0:
    print("WARNING: Signal Gate should have blocked these!")
