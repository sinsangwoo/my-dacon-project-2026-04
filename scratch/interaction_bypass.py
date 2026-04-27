import pandas as pd
import numpy as np
import os
import json
import logging
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import lightgbm as lgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InteractionBypass")

def run_bypass_test():
    RUN_ID = "run_20260427_114054"
    PROCESSED_PATH = f"outputs/{RUN_ID}/processed"
    
    # Load Data
    train_df = pd.read_pickle(f"{PROCESSED_PATH}/train_base.pkl")
    y_train = pd.read_csv("c:/Github_public/my_dacon_project/my-dacon-project-2026-04/data/train.csv")['avg_delay_minutes_next_30m'].iloc[:500]
    y_log = np.log1p(y_train)
    
    # Identify interactions
    prefixes = ['inter_', 'ratio_', 'diff_', 'logprod_']
    inter_cols = [c for c in train_df.columns if any(c.startswith(p) for p in prefixes)]
    # Filter for numeric only
    base_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    base_cols = [c for c in base_cols if not any(c.startswith(p) for p in prefixes) and c != 'ID' and 'id' not in c.lower()]
    
    # 1. Base Only
    logger.info(f"Testing Base Only ({len(base_cols)} features)")
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    oof_base = np.zeros(len(y_log))
    for tr_idx, val_idx in kf.split(train_df):
        m = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, verbose=-1)
        m.fit(train_df.iloc[tr_idx][base_cols], y_log.iloc[tr_idx])
        oof_base[val_idx] = m.predict(train_df.iloc[val_idx][base_cols])
    mae_base = mean_absolute_error(y_train, np.expm1(oof_base))
    
    # 2. Base + ALL Interactions (Bypass Validator)
    all_cols = base_cols + inter_cols
    logger.info(f"Testing Base + ALL Interactions ({len(all_cols)} features)")
    oof_all = np.zeros(len(y_log))
    for tr_idx, val_idx in kf.split(train_df):
        m = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, verbose=-1)
        m.fit(train_df.iloc[tr_idx][all_cols], y_log.iloc[tr_idx])
        oof_all[val_idx] = m.predict(train_df.iloc[val_idx][all_cols])
    mae_all = mean_absolute_error(y_train, np.expm1(oof_all))
    
    logger.info(f"[BYPASS_RESULT] Base MAE: {mae_base:.4f} | Base+Inter MAE: {mae_all:.4f} | Delta: {mae_base - mae_all:.4f}")
    
    # Check std improvement
    std_base = np.std(np.expm1(oof_base))
    std_all = np.std(np.expm1(oof_all))
    logger.info(f"[BYPASS_RESULT] Base Std: {std_base:.4f} | Base+Inter Std: {std_all:.4f}")

if __name__ == "__main__":
    run_bypass_test()
