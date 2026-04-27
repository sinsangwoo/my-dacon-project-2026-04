import pandas as pd
import numpy as np
import os
import json
import logging
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import lightgbm as lgb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InteractionExperiment")

def run_experiment():
    RUN_ID = "run_20260427_105332"
    PROCESSED_PATH = f"outputs/{RUN_ID}/processed"
    
    # Load validation logs
    with open(f"{PROCESSED_PATH}/signal_validation_logs.json", "r") as f:
        val_data = json.load(f)
    
    val_logs = val_data['val_logs']
    
    # Identify interactions and their perm_deltas
    inter_logs = [v for v in val_logs if v['feature'].startswith('inter_')]
    
    # Configs to test
    # Baseline: passed == True
    # Relaxed x1.2: perm_delta > 0.001 / 1.2
    # Relaxed x1.5: perm_delta > 0.001 / 1.5
    # No gate: all inter_ with perm_delta > 0
    
    thresholds = {
        'baseline': 0.001,
        'relaxed_1.2': 0.001 / 1.2,
        'relaxed_1.5': 0.001 / 1.5,
        'no_gate': 0.000001
    }
    
    # Load base features
    train_df = pd.read_pickle(f"{PROCESSED_PATH}/train_base.pkl")
    y_train = pd.read_csv("c:/Github_public/my_dacon_project/my-dacon-project-2026-04/data/train.csv")['avg_delay_minutes_next_30m']
    
    # Select non-interaction approved features
    base_approved = [v['feature'] for v in val_logs if not v['feature'].startswith('inter_') and v['passed']]
    
    results = []
    
    for name, thresh in thresholds.items():
        inter_approved = [v['feature'] for v in inter_logs if v['perm_delta'] > thresh]
        selected_features = base_approved + inter_approved
        
        # Train simple model (3-fold CV)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        oof = np.zeros(len(y_train))
        
        for tr_idx, val_idx in kf.split(train_df):
            X_tr, X_val = train_df.iloc[tr_idx][selected_features], train_df.iloc[val_idx][selected_features]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            
            model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, verbose=-1, n_jobs=-1)
            model.fit(X_tr, y_tr)
            oof[val_idx] = model.predict(X_val)
            
        mae = mean_absolute_error(y_train, oof)
        std_ratio = np.std(oof) / np.std(y_train)
        p99_ratio = np.percentile(oof, 99) / np.percentile(y_train, 99)
        
        results.append({
            'config': name,
            'n_interactions': len(inter_approved),
            'MAE': mae,
            'std_ratio': std_ratio,
            'p99_ratio': p99_ratio
        })
        logger.info(f"[{name}] n_inter={len(inter_approved)} | MAE={mae:.4f} | std_ratio={std_ratio:.4f} | p99_ratio={p99_ratio:.4f}")

    print("\n--- FINAL EXPERIMENT RESULTS ---")
    print(pd.DataFrame(results).to_markdown())

if __name__ == "__main__":
    run_experiment()
