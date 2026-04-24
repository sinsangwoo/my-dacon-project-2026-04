import pandas as pd
import numpy as np
import logging
import sys
import os
import pickle

# Add src to path
sys.path.append(os.getcwd())

from src.config import Config
from src.utils import DriftShieldScaler
from src.data_loader import load_data, build_base_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PCA_SANITIZATION")

def audit_pca_features():
    logger.info("Starting PCA Feature Sanitization Audit...")
    
    # 1. Load Data
    train, _ = load_data()
    train_base = build_base_features(train)
    
    # 2. Fit DriftShield on BASE_COLS
    scaler = DriftShieldScaler()
    scaler.fit(train_base, Config.EMBED_BASE_COLS)
    
    # 3. Transform and Compute Ratios
    df_drifted = scaler.transform(train_base.copy(), Config.EMBED_BASE_COLS)
    
    report = []
    for col in Config.EMBED_BASE_COLS:
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
            "is_raw": True
        })
        
    audit_df = pd.DataFrame(report)
    logger.info("\nAudit Report for BASE_COLS:")
    logger.info(audit_df.sort_values("std_ratio"))
    
    # Identify bad features
    bad_features = audit_df[(audit_df['variance'] < 1e-6) | (audit_df['std_ratio'] < 0.6)]['feature'].tolist()
    logger.info(f"Bad Features (Variance < 1e-6 or Ratio < 0.6): {bad_features}")
    
    return audit_df, bad_features

if __name__ == "__main__":
    audit_pca_features()
