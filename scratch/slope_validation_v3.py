import pandas as pd
import numpy as np
import sys
import os

# Mock data generation to ensure enough scenarios and boundaries
def generate_mock_data():
    np.random.seed(42)
    data = {
        'scenario_id': [1]*20 + [2]*20,
        'val': np.random.randn(40).cumsum()
    }
    return pd.DataFrame(data)

def validate_slope_semantic():
    df = generate_mock_data()
    col = 'val'
    
    # OLD LOGIC (PROVEN BROKEN IN BOUNDARIES)
    series = df.groupby("scenario_id")[col]
    # In OLD logic, shift(4) happened on the GroupBy object, returning a FLAT series with original index
    old_shifted = series.shift(4)
    old_logic = old_shifted.rolling(5, min_periods=1).mean()
    
    # NEW LOGIC (APPLIED FIX)
    rm5 = series.rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    new_logic = rm5.groupby(df["scenario_id"]).shift(4)
    
    # Comparison
    diff = np.abs(old_logic.values - new_logic.values)
    mask = ~np.isnan(old_logic.values) & ~np.isnan(new_logic.values)
    
    print("=== TASK 1: slope_5 Semantic Validation ===")
    print(f"Max Absolute Difference (non-NaN): {np.max(diff[mask]) if mask.any() else 'N/A'}")
    print(f"Mean Absolute Difference (non-NaN): {np.mean(diff[mask]) if mask.any() else 'N/A'}")
    
    # NaN check
    old_nan_idx = np.where(np.isnan(old_logic.values))[0]
    new_nan_idx = np.where(np.isnan(new_logic.values))[0]
    
    print(f"Old NaN indices count: {len(old_nan_idx)}")
    print(f"New NaN indices count: {len(new_nan_idx)}")
    
    mismatch_nan = set(old_nan_idx) ^ set(new_nan_idx)
    print(f"NaN index mismatch count: {len(mismatch_nan)}")
    
    # Boundary Analysis
    print("\n--- Boundary Analysis (Scenario Transitions) ---")
    # Transition point is index 20 (start of S2)
    # Indices 0-3 of S1 should be NaN in both
    # Index 20-23 of S2 should be NaN in NEW, but NOT in OLD (LEAKAGE PROOF)
    print("Index | Scenario | OLD Value | NEW Value | Status")
    for i in range(18, 26):
        old_v = old_logic.iloc[i]
        new_v = new_logic.iloc[i]
        status = "MATCH" if (np.isnan(old_v) and np.isnan(new_v)) or (abs(old_v - new_v) < 1e-9) else "LEAKAGE/DIFF"
        print(f"{i:5} | {df.scenario_id.iloc[i]:8} | {old_v:9.4f} | {new_v:9.4f} | {status}")

    # Interior check
    print("\n--- Interior Stability Check (Scenario 1, deep) ---")
    for i in range(10, 15):
        old_v = old_logic.iloc[i]
        new_v = new_logic.iloc[i]
        print(f"{i:5} | {old_v:9.4f} | {new_v:9.4f} | {'OK' if abs(old_v - new_v) < 1e-9 else 'FAIL'}")

if __name__ == "__main__":
    validate_slope_semantic()
