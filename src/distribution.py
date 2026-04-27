import numpy as np
import pandas as pd
import logging
import os
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

class DomainShiftAudit:
    """Detects distribution drift between Train and Test sets."""
    def calculate_drift(self, train_df, test_df, columns):
        results = []
        for col in columns:
            if col not in train_df.columns or col not in test_df.columns:
                continue
            
            ks_stat, p_val = ks_2samp(train_df[col].dropna(), test_df[col].dropna())
            
            results.append({
                'feature': col,
                'ks_stat': ks_stat,
                'p_value': p_val,
                'is_drifted': ks_stat > 0.1 # Threshold can be tuned
            })
            
        return pd.DataFrame(results)

    def save_report(self, df, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"[DOMAIN_SHIFT] Drift report saved to {path}")

class FeatureStabilityFilter:
    """Prunes features based on statistical drift metrics."""
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.stable_features = []
        self.unstable_features = []

    def fit(self, drift_df, protected_cols=None):
        protected_cols = protected_cols or []
        
        # Features are stable if ks_stat < threshold OR if they are protected
        mask_stable = (drift_df['ks_stat'] < self.threshold) | (drift_df['feature'].isin(protected_cols))
        
        self.stable_features = drift_df[mask_stable]['feature'].tolist()
        self.unstable_candidates = drift_df[~mask_stable]['feature'].tolist()
        
        logger.info(f"[STABILITY_FILTER] Stable: {len(self.stable_features)} | Unstable: {len(self.unstable_candidates)}")

class VarianceMonitor:
    """Monitors feature variance to detect collapse or explosion."""
    @staticmethod
    def audit_variance(df, stats, columns):
        variances = df[columns].var()
        zero_var = variances[variances == 0].index.tolist()
        if zero_var:
            logger.warning(f"[VARIANCE_MONITOR] Detected {len(zero_var)} zero-variance features.")
        return {
            'zero_var': zero_var,
            'total': len(columns),
            'natural_shift': 0,
            'collapse': len(zero_var)
        }

class DistributionAuditor:
    """
    [WHY_THIS_DESIGN] Absolute Distribution Validation.
    [GOAL] Eliminate single-value misinterpretation by comparing predictions against y_train ground truth.
    """
    @staticmethod
    def audit(y_true, y_pred, fold_name="Global"):
        # 1. Std Ratio (Absolute)
        std_true = np.std(y_true) + 1e-9
        std_pred = np.std(y_pred)
        std_ratio = std_pred / std_true
        
        # 2. P99 Ratio (Absolute Tail)
        p99_true = np.percentile(y_true, 99) + 1e-9
        p99_pred = np.percentile(y_pred, 99)
        p99_ratio = p99_pred / p99_true
        
        # 3. Tail Error (Q99 MAE)
        mask_q99 = y_true > np.percentile(y_true, 99)
        q99_mae = np.mean(np.abs(y_true[mask_q99] - y_pred[mask_q99])) if mask_q99.any() else 0
        
        logger.info(f"[DIST_AUDIT] {fold_name:10} | std_ratio={std_ratio:.4f} | p99_ratio={p99_ratio:.4f} | Q99_MAE={q99_mae:.4f}")
        
        return {
            'std_ratio': std_ratio,
            'p99_ratio': p99_ratio,
            'q99_mae': q99_mae
        }
