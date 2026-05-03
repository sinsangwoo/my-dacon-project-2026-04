
import logging
import pandas as pd
import numpy as np
import os
import sys

# Setup logging to see our data-driven messages
logging.basicConfig(level=logging.INFO)

# Add src to path
sys.path.append(os.getcwd())

from src.config import Config
from src.trainer import Trainer
from src.data_loader import load_data

def run_integration_validation():
    print("--- [MISSION 4] Integration Validation ---")
    
    # Load data with layout info
    df_train, df_test = load_data()
    
    # 1. Verify Layout Type Merge
    if 'layout_type' in df_train.columns:
        print("Layout Type Merge: SUCCESS")
    else:
        print("Layout Type Merge: FAILED")
        return

    # 2. Run Leakage-Free CV
    trainer = Trainer(df_train, df_train[Config.TARGET], df_test)
    Config.NFOLDS = 2 
    Config.RAW_LGBM_PARAMS['n_estimators'] = 200
    
    mae, oof = trainer.fit_leakage_free_model()
    
    # 3. ADV AUC Measure
    adv_auc = trainer.perform_adversarial_audit()
    
    # 4. Results
    print("\n[FINAL INTEGRATION RESULTS]")
    print(f"CV MAE: {mae:.4f}")
    print(f"ADV AUC: {adv_auc:.4f}")
    
    # 5. Verify Bias Reduction (Heuristic check)
    # df_test should have type-specific values
    layout_cols = [c for c in trainer.df_test.columns if '_layout_mean' in c]
    if layout_cols:
        test_std = trainer.df_test[layout_cols[0]].std()
        print(f"Test Layout Stat Variation (Std): {test_std:.4f}")
        if test_std > 0:
            print("Layout Fallback Bias Reduction: SUCCESS (Type-specific variation detected)")
        else:
            print("Layout Fallback Bias Reduction: FAILED (All unknown layouts have same value)")

if __name__ == "__main__":
    run_integration_validation()
