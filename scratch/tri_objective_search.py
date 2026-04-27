import os
import gc
import json
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, roc_auc_score

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("FeasibilitySweep")

# Suppress lightgbm warnings
import warnings
warnings.filterwarnings("ignore")

RUN_ID = "run_20260427_122102"
OUTPUT_DIR = f"outputs/{RUN_ID}"
BASE_COLS_CACHE = None

def load_data_sample(sample_size=3000):
    """Load and sample data for the sweep."""
    logger.info("Loading base data...")
    raw_train = pd.read_csv("c:/Github_public/my_dacon_project/my-dacon-project-2026-04/data/train.csv")
    raw_test = pd.read_csv("c:/Github_public/my_dacon_project/my-dacon-project-2026-04/data/test.csv")
    
    y = raw_train['avg_delay_minutes_next_30m'].values
    
    # Stratified sampling based on extreme values
    is_extreme = y > np.percentile(y, 90)
    
    np.random.seed(42)
    extreme_idx = np.where(is_extreme)[0]
    normal_idx = np.where(~is_extreme)[0]
    
    n_extreme = int(sample_size * 0.15)
    n_normal = sample_size - n_extreme
    
    sample_extreme = np.random.choice(extreme_idx, min(n_extreme, len(extreme_idx)), replace=False)
    sample_normal = np.random.choice(normal_idx, min(n_normal, len(normal_idx)), replace=False)
    
    sample_idx = np.concatenate([sample_extreme, sample_normal])
    np.random.shuffle(sample_idx)
    
    train_sample = raw_train.iloc[sample_idx].reset_index(drop=True)
    y_sample = train_sample.pop('avg_delay_minutes_next_30m').values
    test_sample = raw_test.sample(n=min(3000, len(raw_test)), random_state=42).reset_index(drop=True)
    
    logger.info("Generating features for train sample...")
    # Just basic numeric columns for simplicity in grid search, simulating full feature generation
    train_num = train_sample.select_dtypes(include=[np.number])
    test_num = test_sample.select_dtypes(include=[np.number])

    
    # Add interactions manually for speed
    priority_sensors = ['order_inflow_15m', 'robot_utilization', 'congestion_score', 'battery_std', 'heavy_item_ratio']
    for c1, c2 in itertools.combinations(priority_sensors, 2):
        if c1 in train_num.columns and c2 in train_num.columns:
            # Train
            train_num[f'inter_{c1}_x_{c2}'] = train_num[c1] * train_num[c2]
            train_num[f'ratio_{c1}_to_{c2}'] = train_num[c1] / (train_num[c2] + 1e-9)
            train_num[f'diff_{c1}_{c2}'] = train_num[c1] - train_num[c2]
            train_num[f'logprod_{c1}_{c2}'] = np.log1p(train_num[c1] * train_num[c2])
            
            # Test
            test_num[f'inter_{c1}_x_{c2}'] = test_num[c1] * test_num[c2]
            test_num[f'ratio_{c1}_to_{c2}'] = test_num[c1] / (test_num[c2] + 1e-9)
            test_num[f'diff_{c1}_{c2}'] = test_num[c1] - test_num[c2]
            test_num[f'logprod_{c1}_{c2}'] = np.log1p(test_num[c1] * test_num[c2])
    
    global BASE_COLS_CACHE
    BASE_COLS_CACHE = [c for c in train_num.columns if not any(c.startswith(p) for p in ('inter_', 'ratio_', 'diff_', 'logprod_')) and c != 'ID']
    
    return train_num, y_sample, test_num, train_num

def get_feature_subset(train, interaction_strength, feature_count):
    """Filter features based on grid configuration."""
    available_cols = list(BASE_COLS_CACHE)
    
    if interaction_strength == 'base_only':
        available_cols += [c for c in train.columns if c.startswith('inter_')]
    elif interaction_strength == 'ratio_only':
        available_cols += [c for c in train.columns if c.startswith('ratio_')]
    elif interaction_strength == 'full':
        prefixes = ('inter_', 'ratio_', 'diff_', 'logprod_', 'bucket_')
        available_cols += [c for c in train.columns if any(c.startswith(p) for p in prefixes)]
        
    # Simulate feature_count selection using importance heuristics
    # (Just take the first N columns to simulate a restricted space, prioritizing interactions if they exist)
    # We want to make sure we keep base columns, then pad with interactions up to the limit
    
    # Simple feature selection simulation (variance based)
    if feature_count < len(available_cols):
        # We can't easily compute importance for every combo, so we use a random subset 
        # but ensure core sensors are kept.
        core_sensors = [c for c in available_cols if c in ['avg_trip_distance', 'congestion_score', 'robot_utilization']]
        other_cols = [c for c in available_cols if c not in core_sensors]
        
        np.random.seed(42)
        selected_others = np.random.choice(other_cols, feature_count - len(core_sensors), replace=False).tolist()
        final_cols = core_sensors + selected_others
    else:
        final_cols = available_cols[:feature_count]
        
    return final_cols

def run_evaluation(train_df, y, test_df, features, weight_mult, tail_pct):
    """Run CV and return metrics."""
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    test_preds = np.zeros(len(test_df))
    
    for tr_idx, val_idx in kf.split(train_df):
        X_tr, y_tr = train_df.iloc[tr_idx][features], y[tr_idx]
        X_val, y_val = train_df.iloc[val_idx][features], y[val_idx]
        
        # Apply weights
        weights = np.ones(len(y_tr))
        cutoff = np.percentile(y_tr, tail_pct)
        weights[y_tr > cutoff] *= weight_mult
        
        model = lgb.LGBMRegressor(
            n_estimators=150, 
            learning_rate=0.05, 
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=2,
            verbose=-1
        )
        model.fit(X_tr, y_tr, sample_weight=weights)
        
        oof[val_idx] = model.predict(X_val)
        test_preds += model.predict(test_df[features]) / 3
        
    # Distribution Metrics
    mae = mean_absolute_error(y, oof)
    std_ratio = np.std(oof) / np.std(y)
    
    # Tail metrics
    q99_target = np.percentile(y, 99)
    q99_pred = np.percentile(oof, 99)
    p99_ratio = q99_pred / q99_target
    
    tail_idx = y >= q99_target
    if np.sum(tail_idx) > 0:
        q99_mae = mean_absolute_error(y[tail_idx], oof[tail_idx])
    else:
        q99_mae = 0.0
        
    # ADV AUC
    X_adv = np.concatenate([oof.reshape(-1, 1), test_preds.reshape(-1, 1)])
    y_adv = np.concatenate([np.zeros(len(oof)), np.ones(len(test_preds))])
    
    adv_model = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1, n_jobs=2)
    adv_kf = KFold(n_splits=3, shuffle=True, random_state=42)
    adv_preds = np.zeros(len(X_adv))
    
    for tr_idx, val_idx in adv_kf.split(X_adv):
        adv_model.fit(X_adv[tr_idx], y_adv[tr_idx])
        adv_preds[val_idx] = adv_model.predict_proba(X_adv[val_idx])[:, 1]
        
    adv_auc = roc_auc_score(y_adv, adv_preds)
    
    return {
        'MAE': mae,
        'std_ratio': std_ratio,
        'p99_ratio': p99_ratio,
        'Q99_MAE': q99_mae,
        'ADV_AUC': adv_auc
    }

def main():
    logger.info("Starting Feasibility Grid Search...")
    train_sample, y_sample, test_sample, train_full = load_data_sample(3000)
    
    weights = [1.0, 1.5, 2.0, 3.0]
    tail_pcts = [90, 95, 97]
    feature_counts = [30, 60, 100]
    interactions = ['none', 'base_only', 'ratio_only', 'full']
    
    results = []
    total_runs = len(weights) * len(tail_pcts) * len(feature_counts) * len(interactions)
    current_run = 0
    
    # Create results directory
    os.makedirs('logs/feasibility', exist_ok=True)
    
    for w, pct, fc, inter in itertools.product(weights, tail_pcts, feature_counts, interactions):
        current_run += 1
        logger.info(f"Run {current_run}/{total_runs} | w={w}, tail=P{pct}, fc={fc}, inter={inter}")
        
        features = get_feature_subset(train_full, inter, fc)
        metrics = run_evaluation(train_sample, y_sample, test_sample, features, w, pct)
        
        results.append({
            'weight': w,
            'tail_pct': pct,
            'feature_count': fc,
            'interaction': inter,
            **metrics
        })
        
        # Save intermediate results in case of crash
        if current_run % 10 == 0:
            pd.DataFrame(results).to_csv('logs/feasibility/grid_results_temp.csv', index=False)
            gc.collect()
            
    df_results = pd.DataFrame(results)
    df_results.to_csv('logs/feasibility/grid_results_final.csv', index=False)
    logger.info("Grid Search Completed. Results saved to logs/feasibility/grid_results_final.csv")

if __name__ == "__main__":
    main()
