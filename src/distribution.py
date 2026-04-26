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
            
            # [TASK 3] Wasserstein Distance for robust distribution shift
            from scipy.stats import wasserstein_distance
            wd = wasserstein_distance(s_tr, s_te)
            
            # 2. Moments comparison
            drift_results.append({
                "feature": col,
                "ks_stat": float(ks_stat),
                "p_value": float(p_val),
                "wasserstein": float(wd),
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

class VarianceMonitor:
    """
    [WHY_THIS_CHANGE] Decoupled variance diagnostics from DriftShieldScaler (Task 1).
    [ROOT_CAUSE] Scaling logic was generating false-positive warnings by comparing clipped to unclipped std.
    [EXPECTED_IMPACT] Scaling is purely functional. Diagnostics happen here and distinguish natural shift from collapse.
    """
    def __init__(self):
        self.logger = logging.getLogger("VarianceMonitor")

    def audit_variance(self, df, baseline_stats, feature_cols):
        self.logger.info("[VARIANCE_MONITOR] Auditing variance shifts...")
        
        stats_summary = {'total': 0, 'natural_shift': 0, 'collapse': 0}
        
        for col in feature_cols:
            if col not in df.columns or col not in baseline_stats: continue
            
            s = baseline_stats[col]
            current_vals = df[col].dropna().values
            if len(current_vals) < 10: continue
            
            stats_summary['total'] += 1
            current_std = np.std(current_vals) + 1e-9
            baseline_std = s.get('clipped_std', s['std']) + 1e-9
            ratio = current_std / baseline_std
            
            # [TASK 5] Quantile Drift
            curr_p1 = np.percentile(current_vals, 1)
            curr_p99 = np.percentile(current_vals, 99)
            base_p1 = s['p1']
            base_p99 = s['p99']
            
            base_range = abs(base_p99 - base_p1) + 1e-9
            drift_p1 = abs(curr_p1 - base_p1) / base_range
            drift_p99 = abs(curr_p99 - base_p99) / base_range
            max_q_drift = max(drift_p1, drift_p99)
            
            # [TASK 5] Differentiate collapse vs natural shift using combined metrics
            is_variance_collapse = ratio < 0.2
            is_shape_collapse = max_q_drift > 0.50
            
            if is_variance_collapse and is_shape_collapse:
                self.logger.error(f"!!! [TRUE_DISTRIBUTION_COLLAPSE] {col} variance dropped to {ratio:.2%} AND shape collapsed (Q-Drift: {max_q_drift:.2%})")
                stats_summary['collapse'] += 1
            elif ratio < 0.6:
                # Expected for rate/slope features after clipping or in new regimes
                self.logger.info(f"[NATURAL_REGIME_SHIFT] {col} variance compressed to {ratio:.2%} but shape is stable (Q-Drift: {max_q_drift:.2%})")
                stats_summary['natural_shift'] += 1
                
        return stats_summary
class FeatureStabilityFilter:
    """Auto-isolation of unstable features based on drift audit results.

    [WHY_THIS_CHANGE]
    Redesigned to be a "Soft Gate" and signal-type aware (Task 2 & 3).
    [ROOT_CAUSE]
    High-sensitivity signals like rate_1 and slope_5 were hard-filtered due to natural distribution shifts, falsely flagged as instability.
    [EXPECTED_IMPACT]
    Valid signals are allowed to pass to the model evaluation phase. Trend/Volatility features are evaluated with relaxed rules.
    """
    def __init__(self, threshold=None):
        self.threshold = threshold
        self.unstable_features = []
        self.stable_features = []
        self.unstable_candidates = [] # [TASK 2D] Soft gate tags
        self.derived_threshold = None
        self.derivation_log = None

    def _get_feature_type(self, feature_name):
        """Classify feature to apply specific stability rules (Task 2A)."""
        if any(x in feature_name for x in ['_rate_', '_slope_', '_diff_']):
            return 'TREND'
        if any(x in feature_name for x in ['_std_', '_volatility_']):
            return 'VOLATILITY'
        return 'LEVEL'

    def fit(self, drift_df, protected_cols=None):
        ks_vals = drift_df['ks_stat'].dropna().values
        protected_cols = set(protected_cols) if protected_cols else set()

        if self.threshold is None:
            if len(ks_vals) >= 4:
                q1 = float(np.percentile(ks_vals, 25))
                q3 = float(np.percentile(ks_vals, 75))
                iqr = q3 - q1
                derived = float(np.clip(q3 + 1.5 * iqr, 0.05, 0.99))
            else:
                derived = float(np.median(ks_vals)) if len(ks_vals) > 0 else 0.25
            self.derived_threshold = derived
            effective_threshold = derived
        else:
            effective_threshold = self.threshold
            self.derived_threshold = self.threshold

        final_unstable = []
        self.unstable_candidates = []
        
        for idx, row in drift_df.iterrows():
            f = row['feature']
            ks = row['ks_stat']
            tr_std = row.get('tr_std', 1.0)
            
            f_type = self._get_feature_type(f)
            
            # [TASK 2] Tier 1A: Hard Filter (Catastrophic Features)
            if ks > 0.85 or tr_std < 1e-5:
                logger.warning(f"[STABILITY_TIER1A_DROP] {f} dropped due to catastrophic drift or zero variance (KS={ks:.4f}, STD={tr_std:.4f})")
                final_unstable.append(f)
                continue
                
            # [TASK 2] Tier 1B: Danger Zone
            # No adjusted_KS bias. Pure global thresholding.
            thresh = effective_threshold
            
            if ks > thresh:
                # [TASK 2 & 4] Tier 2: Soft Gate (Danger Zone)
                # We tag them as unstable_candidates. They will require 2x strict validation downstream.
                self.unstable_candidates.append(f)
                logger.info(f"[STABILITY_TIER1B_DANGER] {f} (Type: {f_type}) flagged as unstable_candidate (KS={ks:.4f} > {thresh:.4f})")
                
        self.unstable_features = final_unstable
        self.stable_features = drift_df[~drift_df['feature'].isin(self.unstable_features) & ~drift_df['feature'].isin(self.unstable_candidates)]['feature'].tolist()

        logger.info("[STABILITY_FILTER] Identified {} perfectly stable, {} unstable_candidates, {} catastrophic drops".format(
            len(self.stable_features), len(self.unstable_candidates), len(self.unstable_features)))

    def transform(self, features):
        """Filter list of features to exclude unstable ones."""
        # Because we use a soft gate, we DO NOT filter them out here anymore.
        # They are passed downstream.
        return features
