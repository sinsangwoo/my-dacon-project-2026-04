import numpy as np
import pandas as pd
import logging
import os
import json
from scipy.stats import ks_2samp
from .config import Config

logger = logging.getLogger(__name__)

class DomainShiftAudit:
    """Systematic distribution comparator across Train, Val, and Test (v16.0)."""
    
    @staticmethod
    def calculate_drift(train_df, test_df, features):
        """Compute KS-statistic and statistical moments for drift detection."""
        drift_results = []
        logger.info(f"[DRIFT_AUDIT] Analyzing {len(features)} features...")
        
        for col in features:
            if col not in train_df.columns or col not in test_df.columns:
                continue
            
            s_tr = train_df[col].dropna()
            s_te = test_df[col].dropna()
            
            if len(s_tr) == 0 or len(s_te) == 0:
                continue
            
            # 1. KS Test (D-statistic)
            ks_stat, p_val = ks_2samp(s_tr, s_te)
            
            # 2. Moments comparison
            drift_results.append({
                "feature": col,
                "ks_stat": float(ks_stat),
                "p_value": float(p_val),
                "tr_mean": float(s_tr.mean()),
                "te_mean": float(s_te.mean()),
                "tr_std": float(s_tr.std()),
                "te_std": float(s_te.std()),
                "tr_p99": float(np.quantile(s_tr, 0.99)),
                "te_p99": float(np.quantile(s_te, 0.99)),
                "drift_ratio": float(abs(s_tr.mean() - s_te.mean()) / (s_tr.std() + 1e-9))
            })
            
        drift_df = pd.DataFrame(drift_results).sort_values(by="ks_stat", ascending=False)
        return drift_df

    @staticmethod
    def save_report(drift_df, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        drift_df.to_csv(path, index=False)
        logger.info(f"[DRIFT_AUDIT] Drift report saved to {path}")

class FeatureStabilityFilter:
    """Auto-isolation of unstable features based on drift audit results."""
    
    def __init__(self, threshold=0.25):
        self.threshold = threshold
        self.unstable_features = []
        self.stable_features = []

    def fit(self, drift_df):
        """Identify features exceeding the drift threshold."""
        self.unstable_features = drift_df[drift_df['ks_stat'] > self.threshold]['feature'].tolist()
        self.stable_features = drift_df[drift_df['ks_stat'] <= self.threshold]['feature'].tolist()
        
        logger.info(f"[STABILITY_FILTER] Identified {len(self.unstable_features)} unstable features (KS > {self.threshold})")
        logger.info(f"[STABILITY_FILTER] Top unstable: {self.unstable_features[:5]}")
        
    def transform(self, features):
        """Filter list of features to exclude unstable ones."""
        return [f for f in features if f not in self.unstable_features]
