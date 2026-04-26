import pandas as pd
import numpy as np
import logging
import json
import lightgbm as lgb
from scipy.stats import ks_2samp, skew, kurtosis
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import wasserstein_distance
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("AdversarialAudit")

def load_data():
    logger.info("Loading data...")
    train_path = f"{Config.PROCESSED_PATH}/train_leakage_free.csv"
    test_path = f"{Config.PROCESSED_PATH}/test_leakage_free.csv"
    if not os.path.exists(train_path):
        train_path = "data/train.csv"
        test_path = "data/test.csv"
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test

def task1_consistency_failure(df_train):
    logger.info("=== TASK 1: CONSISTENCY FILTER FAILURE ===")
    y = df_train[Config.TARGET]
    X = pd.DataFrame()
    
    X['__random_noise__'] = np.random.normal(0, 1, len(df_train))
    
    n = len(df_train)
    X['__consistent_noise__'] = np.random.normal(0, 1, n)
    # create consistent structural leakage
    X['__consistent_noise__'].iloc[:int(n*0.3)] += y.iloc[:int(n*0.3)] * 0.1
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_gains = []
    
    params = Config.RAW_LGBM_PARAMS.copy()
    params['verbose'] = -1
    for tr_idx, val_idx in kf.split(X, y):
        model = lgb.LGBMRegressor(**params)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        cv_gains.append(model.feature_importances_[1]) # consistent_noise
        
    logger.info(f"Consistent Noise CV Gains: {cv_gains}")
    cv_std = np.std(cv_gains) / (np.mean(cv_gains) + 1e-9)
    logger.info(f"Consistent Noise Gain CV (lower is 'more consistent'): {cv_std:.4f}")

def task4_tier1_danger_zone(df_train, df_test):
    logger.info("=== TASK 4: TIER 1 DANGER ZONE ===")
    features = [c for c in df_train.columns if c not in Config.ID_COLS and c != Config.TARGET][:200]
    
    ks_vals = []
    for f in features:
        ks, _ = ks_2samp(df_train[f].dropna(), df_test[f].dropna())
        ks_vals.append(ks)
        
    ks_vals = np.array(ks_vals)
    logger.info(f"Total features evaluated: {len(ks_vals)}")
    logger.info(f"KS > 0.90 (Current Drop): {np.sum(ks_vals > 0.90)}")
    logger.info(f"KS 0.40 - 0.80 (Danger Zone): {np.sum((ks_vals >= 0.40) & (ks_vals <= 0.80))}")
    logger.info(f"KS 0.80 - 0.90 (Warning Zone): {np.sum((ks_vals > 0.80) & (ks_vals <= 0.90))}")

def task3_distribution_metrics(df_train, df_test):
    logger.info("=== TASK 3: SKEW/KURTOSIS VS WASSERSTEIN ===")
    features = [c for c in df_train.columns if c not in Config.ID_COLS and c != Config.TARGET][:5]
    
    for f in features:
        tr = df_train[f].dropna()
        te = df_test[f].dropna()
        if len(tr) < 10 or len(te) < 10: continue
        sk = skew(tr)
        ku = kurtosis(tr)
        wd = wasserstein_distance(tr, te)
        logger.info(f"Feature {f}: Skew={sk:.2f}, Kurt={ku:.2f}, Wasserstein={wd:.4f}")

if __name__ == "__main__":
    df_tr, df_te = load_data()
    task1_consistency_failure(df_tr)
    task4_tier1_danger_zone(df_tr, df_te)
    task3_distribution_metrics(df_tr, df_te)
