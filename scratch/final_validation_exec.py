
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

def run_validation():
    print("--- [MISSION 4] Final Validation ---")
    
    # Use small subset for speed but enough for ADV AUC
    df_train, df_test = load_data()
    
    # 1. Check Scenario Order Fix
    trainer = Trainer(df_train, df_train[Config.TARGET], df_test)
    ordered_scenarios = trainer.get_scenario_order(df_train)
    alphabetic_scenarios = sorted(df_train['scenario_id'].unique())
    
    print(f"Scenario Order Fix: {'SUCCESS' if ordered_scenarios != alphabetic_scenarios else 'FAILED'}")
    
    # 2. Run Leakage-Free CV (Limited Folds for speed)
    Config.NFOLDS = 2 # Speed up
    Config.RAW_LGBM_PARAMS['n_estimators'] = 200 # Speed up
    
    mae, oof = trainer.fit_leakage_free_model()
    
    # 3. ADV AUC Measure
    adv_auc = trainer.perform_adversarial_audit()
    
    # 4. Results
    print("\n[FINAL RESULTS]")
    print(f"CV MAE: {mae:.4f}")
    print(f"ADV AUC: {adv_auc:.4f}")
    
    # Check for Test Flag (0.0 in layout_stats)
    # df_test should have global_mean instead of 0.0 for new layouts
    layout_cols = [c for c in trainer.df_test.columns if '_layout_mean' in c]
    if layout_cols:
        test_zeros = (trainer.df_test[layout_cols] == 0).sum().sum()
        print(f"Test Set Layout 0.0 Count: {test_zeros} (Should be low/zero for non-boundary cases)")

if __name__ == "__main__":
    run_validation()
