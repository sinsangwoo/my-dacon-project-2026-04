import os
import pandas as pd
import numpy as np
import json
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from src.config import Config
from src.utils import get_logger, load_npy

logger = get_logger("INTERACTION_AUDIT")

def audit_interactions(run_id):
    proc_dir = f"outputs/{run_id}/processed"
    train_base = pd.read_pickle(f"{proc_dir}/train_base.pkl")
    y_train = load_npy(f"{proc_dir}/y_train.npy")
    
    # Base set (approx 90 features as found optimal)
    with open(f"{proc_dir}/signal_validation_logs.json", 'r') as f:
        data = json.load(f)
    df_val = pd.DataFrame(data['val_logs'])
    
    from src.schema import BASE_COLS
    base_set = [f for f in train_base.columns if f in BASE_COLS]
    rejected = df_val[df_val['passed'] == False].sort_values('perm_delta', ascending=False)
    expansion = rejected['feature'].tolist()[:60]
    
    full_pool = list(set(base_set) | set(expansion))
    
    # Target Interactions
    interactions = [f for f in train_base.columns if f.startswith('inter_')]
    
    results = []
    
    # Baseline with all potential interactions in pool
    def get_mae(cols):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        oof = np.zeros(len(train_base))
        for tr, val in kf.split(train_base):
            model = LGBMRegressor(**Config.RAW_LGBM_PARAMS)
            model.fit(train_base.iloc[tr][cols], y_train[tr])
            oof[val] = model.predict(train_base.iloc[val][cols])
        return mean_absolute_error(y_train, oof)

    baseline_mae = get_mae(full_pool)
    logger.info(f"Baseline MAE (with potential interactions): {baseline_mae:.4f}")
    
    for inter in interactions:
        if inter not in full_pool: continue
        
        # Test WITHOUT this interaction
        test_pool = [f for f in full_pool if f != inter]
        mae_without = get_mae(test_pool)
        mae_delta = mae_without - baseline_mae # Positive means interaction was HELPFUL
        
        results.append({
            'interaction': inter,
            'mae_delta': float(mae_delta),
            'verdict': 'HELPFUL' if mae_delta > 0 else 'REDUNDANT'
        })
        logger.info(f"  {inter}: delta={mae_delta:.6f} | {results[-1]['verdict']}")

    print("\n[INTERACTION_ABLATION_REPORT]")
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    audit_interactions("run_20260427_092836")
