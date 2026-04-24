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
logger = logging.getLogger("PCA_HYPOTHESIS")

def test_pca_reconstruction():
    logger.info("Testing PCA Reconstruction Hypothesis...")
    
    # 1. Load Data
    train = pd.read_csv(Config.DATA_PATH + 'train.csv', nrows=50000)
    
    # 2. Base Audit (Already know task_reassign_15m is bad)
    bad_raw = ['task_reassign_15m']
    good_raw = [f for f in BASE_COLS if f not in bad_raw]
    
    # 3. Add simple, high-quality derivatives: Rolling Mean and Std
    df = train.copy()
    new_features = {}
    for col in good_raw:
        series = df.groupby("scenario_id")[col]
        new_features[f"{col}_rm5"] = series.rolling(5, min_periods=1).mean().values
        new_features[f"{col}_rs5"] = series.rolling(5, min_periods=1).std().fillna(0).values
        
    df_full = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
    all_features = good_raw + list(new_features.keys())
    
    # 4. DriftShield and Variance Audit
    scaler = DriftShieldScaler()
    scaler.fit(df_full, all_features)
    df_drifted = scaler.transform(df_full.copy(), all_features)
    
    report = []
    for col in all_features:
        s = scaler.stats[col]
        ratio = np.std(df_drifted[col]) / (s['std'] + 1e-9)
        variance = np.var(df_drifted[col])
        report.append({"feature": col, "variance": variance, "std_ratio": ratio})
    
    audit_df = pd.DataFrame(report)
    final_good = audit_df[(audit_df['variance'] > 1e-6) & (audit_df['std_ratio'] >= 0.6)]['feature'].tolist()
    
    logger.info(f"Final feature count: {len(final_good)}")
    
    # 5. PCA
    from sklearn.decomposition import PCA
    X = df_drifted[final_good].fillna(0).values
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    pca = PCA(n_components=8)
    pca.fit(X)
    var_sum = np.sum(pca.explained_variance_ratio_)
    logger.info(f"PCA Variance with 8 components: {var_sum:.4f}")
    
    if var_sum >= 0.8:
        logger.info("HYPOTHESIS CONFIRMED: 0.8 threshold reached.")
    else:
        logger.warning("HYPOTHESIS FAILED: Threshold not reached.")
        
    return final_good, var_sum

if __name__ == "__main__":
    test_pca_reconstruction()
