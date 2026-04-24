import pandas as pd
import numpy as np
import logging
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.config import Config
from src.utils import DriftShieldScaler
from src.data_loader import load_data, build_base_features
from src.schema import BASE_COLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PCA_SANITIZATION_FULL")

def audit_all_features():
    logger.info("Starting FULL PCA Feature Sanitization Audit (on subset)...")
    
    # 1. Load Data (Subset for speed)
    train, _ = load_data()
    train_subset = train.head(50000).copy()
    train_full = build_base_features(train_subset)
    
    all_cols = train_full.columns.tolist()
    # Filter out ID and target
    features = [c for c in all_cols if c not in Config.ID_COLS and c != Config.TARGET]
    
    # 2. Fit DriftShield on ALL features
    scaler = DriftShieldScaler()
    scaler.fit(train_full, features)
    
    # 3. Transform and Compute Ratios
    df_drifted = scaler.transform(train_full.copy(), features)
    
    report = []
    for col in features:
        s = scaler.stats[col]
        raw_std = s['std']
        
        x_drifted = df_drifted[col].values
        drifted_std = np.std(x_drifted)
        
        ratio = drifted_std / (raw_std + 1e-9)
        variance = np.var(x_drifted)
        
        report.append({
            "feature": col,
            "variance": variance,
            "std_ratio": ratio,
            "is_raw": col in BASE_COLS,
            "is_derivative": any(suffix in col for suffix in ["rate_", "slope_", "diff_", "accel_"])
        })
        
    audit_df = pd.DataFrame(report)
    
    # 4. Filter Good Features
    # Rule: variance > 1e-6 AND std_ratio >= 0.6
    good_mask = (audit_df['variance'] > 1e-6) & (audit_df['std_ratio'] >= 0.6)
    good_features_df = audit_df[good_mask].copy()
    
    logger.info(f"Total features: {len(features)}")
    logger.info(f"Good features: {len(good_features_df)}")
    
    # 5. Find features that contribute most to PCA
    X = df_drifted[good_features_df['feature'].tolist()].fillna(0).values
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    from sklearn.decomposition import PCA
    # We want 8 components to capture >= 0.8 variance.
    # If it doesn't, we need to select a better subset.
    
    pca = PCA(n_components=8)
    pca.fit(X)
    var_sum = np.sum(pca.explained_variance_ratio_)
    logger.info(f"PCA Variance with ALL good features ({len(good_features_df)}): {var_sum:.4f}")
    
    # If it's still low, maybe we have too many "independent" features.
    # PCA captures variance better when features are CORRELATED.
    # If we want 8 components to capture 80%, we need the features to be highly redundant.
    
    # Let's try selecting the top N features by some metric or just raw signals + best derivatives.
    raw_good = good_features_df[good_features_df['is_raw']]['feature'].tolist()
    derivatives_good = good_features_df[good_features_df['is_derivative']]['feature'].tolist()
    
    # Try PCA on raw_good only
    X_raw = df_drifted[raw_good].fillna(0).values
    X_raw = (X_raw - X_raw.mean(axis=0)) / (X_raw.std(axis=0) + 1e-8)
    pca_raw = PCA(n_components=8)
    pca_raw.fit(X_raw)
    logger.info(f"PCA Variance with raw_good ({len(raw_good)}): {np.sum(pca_raw.explained_variance_ratio_):.4f}")
    
    return audit_df, good_features_df

if __name__ == "__main__":
    audit_all_features()
