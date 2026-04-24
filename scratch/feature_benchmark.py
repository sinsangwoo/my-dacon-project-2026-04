
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
import logging
import gc
import os
import json
import time
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import ks_2samp

# Internal project imports
from src.config import Config
from src.schema import FEATURE_SCHEMA
from src.utils import DriftShieldScaler, memory_guard
from src.data_loader import SuperchargedPCAReconstructor, apply_latent_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FeatureKBench")

def calculate_psi(expected, actual, buckets=10):
    """Calculate Population Stability Index (PSI)."""
    def scale_range(data, min_val, max_val):
        return (data - min_val) / (max_val - min_val + 1e-9)

    min_v = min(expected.min(), actual.min())
    max_v = max(expected.max(), actual.max())
    
    e_scaled = scale_range(expected, min_v, max_v)
    a_scaled = scale_range(actual, min_v, max_v)
    
    e_counts, bin_edges = np.histogram(e_scaled, bins=buckets, range=(0, 1))
    a_counts, _ = np.histogram(a_scaled, bins=buckets, range=(0, 1))
    
    e_percents = e_counts / len(expected) + 1e-9
    a_percents = a_counts / len(actual) + 1e-9
    
    psi = np.sum((e_percents - a_percents) * np.log(e_percents / a_percents))
    return psi

def run_benchmark():
    # 1. Load data
    logger.info("Loading base data for benchmarking...")
    train_base = pd.read_pickle(f'{Config.PROCESSED_PATH}/train_base.pkl')
    test_base = pd.read_pickle(f'{Config.PROCESSED_PATH}/test_base.pkl')
    y = np.load(f'{Config.PROCESSED_PATH}/y_train.npy').astype(np.float32)
    sid = np.load(f'{Config.PROCESSED_PATH}/scenario_id.npy', allow_pickle=True)
    
    # 2. Extract Feature Importance (Reference: Phase 3/Fold 0 style)
    # To save time, we use a single fold to get global importance ranking for candidates
    logger.info("Extracting reference feature importance (Fold 0)...")
    kf = GroupKFold(n_splits=5)
    tr_idx, val_idx = next(kf.split(train_base, y, groups=sid))
    
    tr_df, val_df = train_base.iloc[tr_idx], train_base.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    
    scaler = DriftShieldScaler()
    scaler.fit(tr_df, FEATURE_SCHEMA['raw_features'])
    reconstructor = SuperchargedPCAReconstructor(input_dim=len(FEATURE_SCHEMA['raw_features']))
    
    # Build full features for importance ranking
    tr_df_scaled = scaler.transform(tr_df[Config.EMBED_BASE_COLS], Config.EMBED_BASE_COLS)
    reconstructor.fit(tr_df_scaled.values)
    reconstructor.build_fold_cache(tr_df)
    
    tr_full = apply_latent_features(tr_df, reconstructor, scaler=scaler)
    val_full = apply_latent_features(val_df, reconstructor, scaler=scaler)
    
    all_features = FEATURE_SCHEMA['all_features']
    X_tr = tr_full[all_features].values.astype(np.float32)
    X_val = val_full[all_features].values.astype(np.float32)
    
    model = LGBMRegressor(**Config.EMBED_LGBM_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=0)])
    
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({'feature': all_features, 'importance': importances}).sort_values('importance', ascending=False)
    feat_imp_df['cum_importance'] = feat_imp_df['importance'].cumsum() / feat_imp_df['importance'].sum()
    
    # 3. Generate Cutoff Candidates
    cutoffs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    candidates = []
    for c in cutoffs:
        subset = feat_imp_df[feat_imp_df['cum_importance'] <= c]['feature'].tolist()
        if not subset: subset = [feat_imp_df.iloc[0]['feature']]
        candidates.append({'cutoff': c, 'k': len(subset), 'features': subset})
    
    # 4. Benchmarking
    results = []
    logger.info(f"Starting benchmark for {len(candidates)} candidates...")
    
    for cand in candidates:
        k = cand['k']
        feats = cand['features']
        logger.info(f"Testing k={k} (Cutoff={cand['cutoff']})")
        
        start_time = time.time()
        
        # We benchmark on Fold 0 for speed, but full report would need 5-fold
        X_tr_sub = tr_full[feats].values.astype(np.float32)
        X_val_sub = val_full[feats].values.astype(np.float32)
        
        # Memory tracking (approximate via peak)
        import psutil
        process = psutil.Process(os.getpid())
        mem_start = process.memory_info().rss / 1024**2
        
        m = LGBMRegressor(**Config.EMBED_LGBM_PARAMS)
        m.fit(X_tr_sub, y_tr, eval_set=[(X_val_sub, y_val)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=0)])
        
        mem_end = process.memory_info().rss / 1024**2
        wall_clock = time.time() - start_time
        preds = m.predict(X_val_sub)
        mae = mean_absolute_error(y_val, preds)
        
        # Reliability: PSI (Population Stability Index) for first embedding feature in subset
        embed_cols = [f for f in feats if 'embed' in f]
        psi_val = 0.0
        if embed_cols:
            psi_val = calculate_psi(tr_full[embed_cols[0]].values, val_full[embed_cols[0]].values)
            
        results.append({
            'cutoff': cand['cutoff'],
            'k': k,
            'mae': mae,
            'wall_clock': wall_clock,
            'peak_mem_delta': mem_end - mem_start,
            'psi': psi_val
        })
        
        del X_tr_sub, X_val_sub, m, preds; gc.collect()

    # 5. Report
    res_df = pd.DataFrame(results)
    print("\n--- BENCHMARK RESULTS ---")
    print(res_df.to_string(index=False))
    
    # Save to file
    res_df.to_csv('feature_benchmark_results.csv', index=False)
    logger.info("Benchmark complete. Results saved to feature_benchmark_results.csv")

if __name__ == "__main__":
    run_benchmark()
