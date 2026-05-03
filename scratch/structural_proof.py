
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.getcwd())

from src.config import Config

def prove_temporal_mismatch():
    print("--- [PROOF: MISSION 1] Temporal Mismatch ---")
    df = pd.read_csv('data/train.csv')
    
    # Current logic: sorted(scenario_id)
    current_order = sorted(df['scenario_id'].unique())
    
    # Real logic: min(ID) per scenario
    # We assume ID is chronological (TRAIN_000000, TRAIN_000001, ...)
    real_order_df = df.groupby('scenario_id')['ID'].min().reset_index()
    real_order_df['id_num'] = real_order_df['ID'].str.extract('(\d+)').astype(int)
    real_order = real_order_df.sort_values('id_num')['scenario_id'].tolist()
    
    print(f"Current Order (first 10): {current_order[:10]}")
    print(f"Real Order (first 10):    {real_order[:10]}")
    
    mismatch_count = 0
    for c, r in zip(current_order, real_order):
        if c != r: mismatch_count += 1
    
    print(f"Mismatch Count: {mismatch_count} / {len(current_order)}")
    return real_order

def prove_layout_leakage():
    print("\n--- [PROOF: MISSION 2] Layout Leakage ---")
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')
    
    train_layouts = set(df_train['layout_id'].unique())
    test_layouts = set(df_test['layout_id'].unique())
    
    new_test_layouts = test_layouts - train_layouts
    print(f"Total Train Layouts: {len(train_layouts)}")
    print(f"Total Test Layouts: {len(test_layouts)}")
    print(f"New Layouts in Test: {len(new_test_layouts)}")
    
    if len(new_test_layouts) > 0:
        print(f"Leakage confirmed: {len(new_test_layouts)} layouts in test will have fillna(0.0) stats.")
    else:
        print("No new layouts in test? Double checking...")
        # Check if they overlap in a way that fillna(0.0) still happens (e.g. per fold)
    
    # Simulate a fold split
    unique_scenarios = df_train['scenario_id'].unique()
    fold0_scenarios = unique_scenarios[:len(unique_scenarios)//5]
    tr_df = df_train[~df_train['scenario_id'].isin(fold0_scenarios)]
    val_df = df_train[df_train['scenario_id'].isin(fold0_scenarios)]
    
    tr_l = set(tr_df['layout_id'].unique())
    val_l = set(val_df['layout_id'].unique())
    new_val_l = val_l - tr_l
    print(f"New Layouts in Fold 0 Validation: {len(new_val_l)}")

def analyze_ks_distribution():
    print("\n--- [PROOF: MISSION 3] KS Distribution ---")
    # This requires calculated features, let's look at raw for now
    df_train = pd.read_csv('data/train.csv').sample(10000, random_state=42)
    df_test = pd.read_csv('data/test.csv').sample(10000, random_state=42)
    
    from scipy.stats import ks_2samp
    common_cols = [c for c in df_train.columns if c in df_test.columns and c not in Config.ID_COLS and c != Config.TARGET]
    
    ks_values = []
    for col in common_cols:
        stat, _ = ks_2samp(df_train[col].dropna(), df_test[col].dropna())
        ks_values.append(stat)
    
    ks_series = pd.Series(ks_values, index=common_cols).sort_values(ascending=False)
    print("Top 10 KS (Drift) Features (Raw):")
    print(ks_series.head(10))
    print(f"Mean KS: {ks_series.mean():.4f}")
    print(f"P90 KS: {ks_series.quantile(0.9):.4f}")

if __name__ == "__main__":
    prove_temporal_mismatch()
    prove_layout_leakage()
    analyze_ks_distribution()
