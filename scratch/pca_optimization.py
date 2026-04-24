import pandas as pd
import numpy as np
import logging
import sys
import os
from sklearn.decomposition import PCA

# Add src to path
sys.path.append(os.getcwd())

from src.config import Config
from src.utils import DriftShieldScaler
from src.schema import BASE_COLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PCA_OPTIMIZATION")

def optimize_pca_subset():
    logger.info("Optimizing PCA Feature Subset...")
    
    # 1. Load Data
    train = pd.read_csv(Config.DATA_PATH + 'train.csv', nrows=50000)
    
    # 2. Base Audit
    scaler = DriftShieldScaler()
    scaler.fit(train, BASE_COLS)
    df_drifted = scaler.transform(train.copy(), BASE_COLS)
    
    # Standardize
    X_full = df_drifted[BASE_COLS].fillna(0).values
    X_full = (X_full - X_full.mean(axis=0)) / (X_full.std(axis=0) + 1e-8)
    
    current_features = list(BASE_COLS)
    best_var = 0
    
    # Iteratively remove features that contribute LEAST to the top 8 components
    # until we reach 0.8 or we have too few features.
    while len(current_features) > 8:
        X = df_drifted[current_features].fillna(0).values
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        pca = PCA(n_components=8)
        pca.fit(X)
        var_sum = np.sum(pca.explained_variance_ratio_)
        
        logger.info(f"Features: {len(current_features)} | Variance: {var_sum:.4f}")
        
        if var_sum >= 0.8:
            logger.info(f"SUCCESS: Threshold reached with {len(current_features)} features.")
            logger.info(f"FINAL OPTIMIZED FEATURES: {current_features}")
            return current_features, var_sum
            
        # Find feature with least contribution (heuristic: smallest absolute sum of loadings in top 8)
        loadings = np.sum(np.abs(pca.components_), axis=0)
        worst_idx = np.argmin(loadings)
        worst_feat = current_features[worst_idx]
        
        logger.info(f"Removing worst feature: {worst_feat} (loading sum: {loadings[worst_idx]:.4f})")
        current_features.remove(worst_feat)
        
    logger.info(f"FINAL OPTIMIZED FEATURES: {current_features}")
    return current_features, var_sum

if __name__ == "__main__":
    optimize_pca_subset()
