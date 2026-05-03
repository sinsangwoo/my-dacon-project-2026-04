import numpy as np
import pandas as pd
import logging
import os
from scipy.stats import ks_2samp
from lightgbm import LGBMClassifier

logger = logging.getLogger(__name__)

class DomainShiftAudit:
    """Detects distribution drift between Train and Test sets."""
    def calculate_drift(self, train_df, test_df, columns):
        """
        [PHASE 3: ADVERSARIAL-AWARE DRIFT AUDIT]
        Now incorporates adversarial importance to derive a TRULY data-driven threshold.
        """
        results = []
        for col in columns:
            if col not in train_df.columns or col not in test_df.columns:
                continue
            
            ks_stat, p_val = ks_2samp(train_df[col].dropna(), test_df[col].dropna())
            
            results.append({
                'feature': col,
                'ks_stat': ks_stat,
                'p_value': p_val
            })
            
        drift_df = pd.DataFrame(results)
        
        # [MISSION 1] TRUE Data-Driven Threshold Selection
        # Step 1: Compute Adversarial Importance on the fly
        logger.info("[DOMAIN_SHIFT] Computing adversarial importance for threshold derivation...")
        adv_importance = self._get_adversarial_importance(train_df, test_df, columns)
        
        # Step 2: Derive threshold from Signal-Efficiency Curvature
        optimal_threshold = self.determine_optimal_threshold(drift_df, adv_importance)
        drift_df['is_drifted'] = drift_df['ks_stat'] > optimal_threshold
        
        return drift_df, optimal_threshold

    def _get_adversarial_importance(self, train_df, test_df, columns):
        """Trains a shallow adversarial classifier to identify drift-contributors."""
        # [SSOT_FIX] Local import removed
        
        # Ensure columns exist in both DFs (defensive filtering)
        columns = [c for c in columns if c in train_df.columns and c in test_df.columns]
        if not columns:
            return {}
        
        # Sample for speed (Zero-Hardcode Rule: adaptive sampling)
        n_tr = min(5000, len(train_df))
        n_te = min(5000, len(test_df))
        
        X_tr = train_df[columns].sample(n_tr, random_state=42).fillna(-999).values
        X_te = test_df[columns].sample(n_te, random_state=42).fillna(-999).values
        
        X = np.vstack([X_tr, X_te])
        y = np.hstack([np.zeros(len(X_tr)), np.ones(len(X_te))])
        
        adv_clf = LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, verbose=-1)
        adv_clf.fit(X, y)
        
        return dict(zip(columns, adv_clf.feature_importances_))

    def determine_optimal_threshold(self, drift_df, adv_importance):
        """
        [MISSION 1 — KS THRESHOLD RECONSTRUCTION]
        
        [WHY FIXED THRESHOLDS FAILED]
        Static thresholds like 0.05 or 0.10 are arbitrary 'guesses' that ignore the 
        actual 'Signal-to-Drift Density' of a specific dataset. 
        Collective Weak Drift (many features with KS 0.03-0.05) can still reach 
        ADV AUC 0.95 if the threshold is too loose, while a too-strict threshold 
        kills predictive variance.
        
        [WHY INFLECTION-BASED IS NECESSARY]
        The optimal cutoff is not a value, but a 'state change'. 
        By finding the point where Cumulative Adversarial Importance starts to 
        accelerate relative to the feature count (Signal Curvature), we identify 
        the statistical transition from 'intrinsic noise' to 'structural drift'.
        
        [LOGIC: Signal Curvature Analysis (SCA)]
        1. Sort features by KS statistic (ascending).
        2. Compute Cumulative Adversarial Importance.
        3. Use the Elbow/Knee method to maximize the distance from the 
           'Unity Line' (Diagonal from (0,0) to (1,1) in normalized space).
        """
        if drift_df.empty: 
            return 0.15 # Absolute fallback if no features
        
        # Map importance to drift_df
        drift_df['importance'] = drift_df['feature'].map(adv_importance).fillna(0)
        
        # Sort by KS to see how signal accumulates as we allow more drift
        df_sorted = drift_df.sort_values('ks_stat').reset_index(drop=True)
        df_sorted['cum_imp'] = df_sorted['importance'].cumsum()
        df_sorted['cum_count'] = np.arange(1, len(df_sorted) + 1)
        
        # Normalize to [0, 1] for geometric analysis
        imp_total = df_sorted['cum_imp'].max()
        if imp_total < 1e-9:
            # If adversarial importance is zero for all, fallback to a strict distribution percentile
            logger.warning("[DOMAIN_SHIFT] Zero adversarial importance detected. Falling back to KS P75.")
            return float(np.percentile(drift_df['ks_stat'], 75))
            
        imp_norm = df_sorted['cum_imp'] / imp_total
        count_norm = df_sorted['cum_count'] / len(df_sorted)
        
        # Geometric Inflection: Maximize distance to the diagonal
        distances = np.abs(count_norm - imp_norm)
        
        # Find the peak (conservative elbow)
        knee_idx = distances.idxmax()
        optimal_threshold = float(df_sorted.loc[knee_idx, 'ks_stat'])
        
        # [DATA-DRIVEN SAFETY BOUNDING]
        # Bounding by the distribution itself (P90) to ensure we never drop 100% of the signal.
        p90_ks = np.percentile(drift_df['ks_stat'], 90)
        final_optimal = min(optimal_threshold, p90_ks)
        
        # Absolute floor: Never go below 0.03 (intrinsic noise level)
        final_optimal = max(final_optimal, 0.03)
        
        logger.info(f"[DOMAIN_SHIFT] SCA Threshold: {final_optimal:.4f} (Inflection at idx {knee_idx}/{len(df_sorted)})")
        logger.info(f"[DOMAIN_SHIFT] Drift signal coverage at threshold: {imp_norm[knee_idx]:.2%}")
        
        return float(final_optimal)

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
        # [OOF_GAP_FIX] NaN-safe distribution audit
        mask = ~np.isnan(y_pred)
        if not mask.any():
            logger.warning(f"[DIST_AUDIT] {fold_name} | No valid predictions to audit.")
            return {'std_ratio': 0.0, 'p99_ratio': 0.0, 'q99_mae': 0.0}
            
        y_t = y_true[mask]
        y_p = y_pred[mask]
        
        # 1. Std Ratio (Absolute)
        std_true = np.std(y_t) + 1e-9
        std_pred = np.std(y_p)
        std_ratio = std_pred / std_true
        
        # 2. P99 Ratio (Absolute Tail)
        p99_true = np.percentile(y_t, 99) + 1e-9
        p99_pred = np.percentile(y_p, 99)
        p99_ratio = p99_pred / p99_true
        
        # 3. Tail Error (Q99 MAE)
        p99_val = np.percentile(y_t, 99)
        mask_q99 = y_t > p99_val
        q99_mae = np.mean(np.abs(y_t[mask_q99] - y_p[mask_q99])) if mask_q99.any() else 0
        
        logger.info(f"[DIST_AUDIT] {fold_name:10} | std_ratio={std_ratio:.4f} | p99_ratio={p99_ratio:.4f} | Q99_MAE={q99_mae:.4f}")
        
        return {
            'std_ratio': std_ratio,
            'p99_ratio': p99_ratio,
            'q99_mae': q99_mae
        }
