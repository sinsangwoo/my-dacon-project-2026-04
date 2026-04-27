import json
import pandas as pd
import numpy as np

run_id = "run_20260427_092836"
log_path = f"outputs/{run_id}/processed/signal_validation_logs.json"

with open(log_path, 'r') as f:
    data = json.load(f)

val_logs = data['val_logs']
df = pd.DataFrame(val_logs)

# Task 1 & 2: Why were they rejected?
# Rejected by NoiseCeiling but has positive perm_delta
noise_rejected_but_positive = df[(df['passed'] == False) & (df['rejection_reasons'].apply(lambda x: 'NoiseCeiling' in x)) & (df['perm_delta'] > 0)]

# Rejected by SignInstability but has high perm_delta
instability_rejected_but_high = df[(df['passed'] == False) & (df['rejection_reasons'].apply(lambda x: 'SignInstability' in x)) & (df['perm_delta'] > 0.001)]

print(f"--- REJECTION ANALYSIS ---")
print(f"Total Rejected: {len(df[df['passed'] == False])}")
print(f"Rejected by NoiseCeiling but Positive Perm: {len(noise_rejected_but_positive)}")
print(f"Rejected by SignInstability but High Perm (> 0.001): {len(instability_rejected_but_high)}")

# Task 3: Category Survival
categories = {
    'Trend': ['_slope_', '_rate_', '_diff_'],
    'Volatility': ['_std_', '_volatility_'],
    'Interaction': ['inter_'],
    'Base': [] # Base are BASE_COLS (not in val_logs usually because they bypass)
}

def get_cat(feat):
    for k, suffixes in categories.items():
        if k == 'Interaction':
            if feat.startswith('inter_'): return k
        else:
            if any(s in feat for s in suffixes): return k
    return 'Other'

df['category'] = df['feature'].apply(get_cat)

cat_stats = df.groupby('category').agg(
    total=('feature', 'count'),
    survived=('passed', 'sum'),
    rejected=('passed', lambda x: (x == False).sum())
).reset_index()

print(f"\n--- CATEGORY SURVIVAL ANALYSIS ---")
print(cat_stats)

# Find top rejected features by perm_delta
top_rejected = df[df['passed'] == False].sort_values('perm_delta', ascending=False)
print(f"\n--- TOP REJECTED SIGNALS (Potential False Negatives) ---")
print(top_rejected[['feature', 'category', 'perm_delta', 'rejection_reasons']].head(20))
