import pandas as pd
import numpy as np
import logging
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.config import Config
from src.utils import DriftShieldScaler
from src.data_loader import load_data
from src.schema import BASE_COLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PCA_SANITIZATION_FAST")

def audit_pca_features_fast():
    logger.info("Starting FAST PCA Feature Sanitization Audit...")
    
    # 1. Load Data (Raw columns only)
    train = pd.read_csv(Config.DATA_PATH + 'train.csv')
    
    # 2. Fit DriftShield on BASE_COLS
    scaler = DriftShieldScaler()
    scaler.fit(train, BASE_COLS)
    
    # 3. Transform and Compute Ratios
    df_drifted = scaler.transform(train.copy(), BASE_COLS)
    
    report = []
    for col in BASE_COLS:
        s = scaler.stats[col]
        raw_std = s['std']
        
        x_drifted = df_drifted[col].values
        drifted_std = np.std(x_drifted)
        
        ratio = drifted_std / (raw_std + 1e-9)
        variance = np.var(x_drifted)
        
        report.append({
            "feature": col,
            "variance": variance,
            "std_ratio": ratio
        })
        
    audit_df = pd.DataFrame(report)
    
    # 4. Identify features to remove
    to_remove = audit_df[(audit_df['variance'] < 1e-6) | (audit_df['std_ratio'] < 0.6)]['feature'].tolist()
    
    logger.info("\nAudit Report:")
    logger.info(audit_df.sort_values("std_ratio"))
    logger.info(f"\nFeatures to REMOVE: {to_remove}")
    
    # 5. Check if removing them improves PCA
    # We'll use a subset of features that are "good"
    good_features = [f for f in BASE_COLS if f not in to_remove]
    
    from sklearn.decomposition import PCA
    X = df_drifted[good_features].fillna(0).values
    # Standardize as PCA is sensitive to scale (though DriftShield handles some)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    pca = PCA(n_components=8)
    pca.fit(X)
    var_sum = np.sum(pca.explained_variance_ratio_)
    logger.info(f"New PCA Variance with {len(good_features)} features: {var_sum:.4f}")
    
    return to_remove, good_features, var_sum

if __name__ == "__main__":
    audit_pca_features_fast()
