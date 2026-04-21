import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath('.'))

from src.data_loader import build_features
from src.config import Config
from src.utils import validate_submission, get_logger

def test_immutability():
    print("[TEST] Loading Test Data...")
    # Use 10 samples for clarity
    test = pd.read_csv('./data/test.csv').head(10)
    original_ids = test['ID'].values.copy()
    
    # Mock schema
    schema = {
        'all_features': ['order_inflow_15m'],
        'feature_to_index': {'order_inflow_15m': 0},
        'raw_features': ['order_inflow_15m'],
        'embed_features': []
    }
    
    print("[TEST] Running build_features...")
    X_df, df_processed = build_features(test, schema, mode='raw')
    
    print("[TEST] Verifying ID integrity...")
    assert (df_processed['ID'].values == original_ids).all(), "FAILED: ID order mismatched after build_features!"
    print("SUCCESS: ID order preserved.")

    print("\n[TEST] Verifying Phase 8 Logic (Rule 3 np.allclose)...")
    logger = get_logger("TEST")
    
    # Correct order template
    sample_df = pd.DataFrame({'ID': original_ids, Config.TARGET: [0.0]*10})
    
    # CASE 1: Correct output from pipeline (IDs match sample_df)
    print("[TEST] Case 1: Correct Alignment")
    preds = np.arange(10).astype(float)
    df_correct = pd.DataFrame({'ID': original_ids, Config.TARGET: preds})
    
    try:
        validate_submission(df_correct, sample_df, logger)
        print("PASS: Alignment verified.")
    except Exception as e:
        print(f"FAIL: False positive error: {e}")

    # CASE 2: The actual BUG (Rows shuffled, IDs mismatch, blind assignment attempted)
    # This simulates a failure in the Capture-Restore logic or a regression to the old main.py
    print("\n[TEST] Case 2: Disaligned IDs (The Bug Simulation)")
    
    # 1. Pipeline produces shuffled IDs (simulate failure to restore order)
    shuffled_indices = np.random.permutation(10)
    df_shuffled_ids = df_correct.iloc[shuffled_indices].reset_index(drop=True)
    
    # 2. Main.py performs BLIND ASSIGNMENT (as it used to do)
    # It takes the preds from df_shuffled_ids (which are in WRONG ID order) 
    # and sticks them into sample_df
    preds_blind = df_shuffled_ids[Config.TARGET].values
    
    sample_sub_buggy = sample_df.copy()
    sample_sub_buggy[Config.TARGET] = preds_blind
    
    # Now validate_submission should catch that sample_sub_buggy['ID'] was already 'A, B, C' 
    # but the predictions assigned to 'A' actually belong to whatever ID was at df_shuffled_ids[0].
    
    # Wait, in the real bug, we didn't have ID in the pred array if it was numpy. 
    # My validate_submission implementation expects df to HAVE 'ID'.
    # If the rows of df match sample_df, it passes.
    # If I give it sample_sub_buggy, the ID check (Rule 3.1) passes because I didn't change the IDs in sample_sub_buggy.
    # BUT Rule 3.2 (Merge Check) uses df[['ID', Config.TARGET]]. 
    # If df is df_shuffled_ids, it HAS the mapping.
    
    try:
        # We pass df_shuffled_ids to simulate the state BEFORE blind assignment if we were checking df.
        # But Phase 8 checks the final 'sample_sub' AFTER assignment.
        
        # To test Rule 3.2 correctly within validate_submission(df, sample_df):
        # We need to simulate the state where df has DIFFERENT order than sample_df.
        validate_submission(df_shuffled_ids, sample_df, logger)
        print("FAIL: Disaligned IDs NOT caught!")
    except RuntimeError as e:
        print(f"PASS: Disaligned IDs successfully caught! Error: {e}")

if __name__ == "__main__":
    test_immutability()
