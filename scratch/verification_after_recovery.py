import sys
import os
sys.path.append(os.getcwd())
from src.schema import FEATURE_SCHEMA
import logging

# Setup basic logging to avoid errors if any module expects it
logging.basicConfig(level=logging.INFO)

print(f"--- Schema Audit ---")
print(f"Raw features count: {len(FEATURE_SCHEMA['raw_features'])}")
print(f"Embed features count: {len(FEATURE_SCHEMA['embed_features'])}")
print(f"Total features count: {len(FEATURE_SCHEMA['all_features'])}")

# Check for new features
new_suffixes = ['_rolling_mean_3', '_rolling_std_3', '_slope_5', '_rate_1']
found_new = {s: 0 for s in new_suffixes}

for f in FEATURE_SCHEMA['raw_features']:
    for s in new_suffixes:
        if f.endswith(s):
            found_new[s] += 1

print("\n--- Expansion Verification ---")
for s, count in found_new.items():
    print(f"Suffix {s}: {count} features found")

# Test data_loader imports and basic logic
try:
    from src.data_loader import build_base_features
    print("\n--- Data Loader Import: SUCCESS ---")
except Exception as e:
    print(f"\n--- Data Loader Import: FAILED ---\n{e}")

# Test trainer imports
try:
    from src.trainer import Trainer
    print("--- Trainer Import: SUCCESS ---")
except Exception as e:
    print(f"--- Trainer Import: FAILED ---\n{e}")
