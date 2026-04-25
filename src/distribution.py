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
    """Auto-isolation of unstable features based on drift audit results.

    [WHY_THIS_CHANGE]
    Problem:
        __init__ had a hardcoded default threshold of 0.25 with no statistical basis.
        The value was passed in from Config.STABILITY_THRESHOLD (also a fixed constant)
        making both the default and the override arbitrary.
    Root Cause:
        Original design treated 0.25 as a universal KS-statistic cutoff, ignoring that
        the actual KS distribution varies with dataset size and feature correlation.
    Decision:
        If threshold is None (default), derive it from the KS-statistic distribution
        using IQR-based outlier detection: threshold = Q3 + 1.5 * IQR.
        -> Treats high-drift features as "outliers" in the drift distribution.
        -> Adapts to datasets where all features are slightly drifted (raises bar)
           or where drift is severe (lowers bar).
    Why IQR (not alternatives):
        - Fixed 0.25: violates RULE 1.
        - Simple percentile: another form of hardcoding.
        - IQR: robust, interpretable, scale-invariant.
    Expected Impact:
        Stability threshold adapts to the actual drift severity in each fold.
        Derivation is logged for full traceability.
    """

    def __init__(self, threshold=None):
        # [WHY_THIS_CHANGE] threshold=None by default; if None, derived at fit() time.
        # Previously: threshold=0.25 (hardcoded). Now: data-driven or caller-supplied.
        self.threshold = threshold
        self.unstable_features = []
        self.stable_features = []
        self.derived_threshold = None
        self.derivation_log = None

    def fit(self, drift_df):
        """Identify features exceeding the drift threshold.

        If self.threshold is None, derives it from the KS-statistic distribution
        using IQR-based outlier detection.
        """
        ks_vals = drift_df['ks_stat'].dropna().values

        if self.threshold is None:
            # [WHY_THIS_CHANGE] Adaptive KS threshold derivation
            # Problem: No threshold provided → old code would crash or use 0.25 default.
            # Root Cause: Hardcoded default.
            # Decision: Derive from IQR of the KS distribution.
            if len(ks_vals) >= 4:
                q1 = float(np.percentile(ks_vals, 25))
                q3 = float(np.percentile(ks_vals, 75))
                iqr = q3 - q1
                derived = float(np.clip(q3 + 1.5 * iqr, 0.05, 0.99))
            else:
                # Fallback: median of KS values when sample too small for IQR
                derived = float(np.median(ks_vals)) if len(ks_vals) > 0 else 0.25
                logger.warning(
                    "[STABILITY_FILTER] Too few features for IQR derivation. "
                    "Using median KS = {:.4f} as threshold.".format(derived)
                )
            self.derived_threshold = derived
            self.derivation_log = (
                "IQR outlier detection on KS distribution "
                "(n={}, Q3={:.4f}, IQR={:.4f}) -> Q3+1.5*IQR = {:.4f} [clip 0.05-0.99]".format(
                    len(ks_vals),
                    np.percentile(ks_vals, 75) if len(ks_vals) >= 4 else float('nan'),
                    (np.percentile(ks_vals, 75) - np.percentile(ks_vals, 25)) if len(ks_vals) >= 4 else float('nan'),
                    derived
                )
            )
            effective_threshold = derived
            logger.info("[STABILITY_FILTER] Derived adaptive threshold={:.4f} | {}".format(
                derived, self.derivation_log))
        else:
            effective_threshold = self.threshold
            self.derived_threshold = self.threshold
            self.derivation_log = "Caller-supplied threshold={:.4f}".format(self.threshold)
            logger.info("[STABILITY_FILTER] Using caller-supplied threshold={:.4f}".format(self.threshold))

        self.unstable_features = drift_df[drift_df['ks_stat'] > effective_threshold]['feature'].tolist()
        self.stable_features = drift_df[drift_df['ks_stat'] <= effective_threshold]['feature'].tolist()

        logger.info("[STABILITY_FILTER] Identified {} unstable features (KS > {:.4f})".format(
            len(self.unstable_features), effective_threshold))
        logger.info("[STABILITY_FILTER] Top unstable: {}".format(self.unstable_features[:5]))

    def transform(self, features):
        """Filter list of features to exclude unstable ones."""
        return [f for f in features if f not in self.unstable_features]
