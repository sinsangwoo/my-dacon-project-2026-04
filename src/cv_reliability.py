"""
[CV RELIABILITY RECONSTRUCTION MODULE]
Author: Forensic ML Consultant
Date: 2026-04-30

[MISSION]
This module addresses the fundamental measurement system failure where:
  - CV MAE and LB MAE show near-zero correlation (~0.05)
  - Fold-wise MAE appears stable (std ≈ 0.25) but does NOT translate to LB performance
  - Adversarial AUC remains 0.80-0.95, indicating unresolved distribution shift
  - std_ratio = 0.46~0.62 across all runs → predictions are variance-compressed
  - mean_ratio = 0.65~0.85 → predictions consistently underpredict ground truth

[DIAGNOSIS]
The root cause is a three-layered failure in the current CV pipeline:
  1. Expanding-window asymmetry: fold 0 trains on ~17% of data, fold 4 on ~83%.
     OOF MAE averages across fundamentally different model states.
  2. Homogeneous validation domain: ALL fold validation sets are drawn from training
     scenarios. The test distribution (higher order_inflow, different robot_active, etc.)
     is NEVER represented in any validation fold.
  3. Collective drift masquerade: features individually pass KS filter but collectively
     create AUC 0.80-0.95. CV measures performance within the training manifold while
     LB measures generalization to the test manifold.

[REFERENCES]
  - trainer.py: _get_time_aware_splits() → expanding window logic
  - experiment_registry.json: fold_stats, adv_auc, std_ratio evidence
  - forensic_drift.json: avg_items_per_order KS=0.356, robot_idle KS=0.337
  - config.py: Config.NFOLDS = 5, Config.ADV_TARGET_AUC = 0.75
"""

import logging
import numpy as np
import pandas as pd
import os
from lightgbm import LGBMClassifier
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1-A: CV Pipeline Analyzer
# Quantifies structural bias in the existing _get_time_aware_splits() logic.
# ─────────────────────────────────────────────────────────────────────────────

class CVPipelineAnalyzer:
    """
    [WHY THIS EXISTS]
    The existing _get_time_aware_splits() uses an expanding window where:
      chunks = np.array_split(unique_scenarios, NFOLDS + 1)  # 6 chunks for 5 folds
      Fold k trains on chunks[0..k], validates on chunks[k+1]
    
    This creates two structural problems:
      (a) Fold 0: ~17% train data → biased toward underfitting → low MAE artifact
      (b) All validation folds are drawn from train-distribution scenarios.
          Test scenarios have significantly different distributions (KS 0.26-0.36).
    
    This class quantifies these biases numerically without changing the pipeline.
    """

    def __init__(self, df_train, y, df_test, n_folds=5):
        self.df_train = df_train
        self.y = y
        self.df_test = df_test
        self.n_folds = n_folds
        self.analysis_results = {}

    def get_scenario_order(self, df):
        # [PRESERVED] Same logic as trainer.py get_scenario_order — DO NOT MODIFY
        temp_df = df[["scenario_id", "ID"]].copy()
        temp_df["id_num"] = temp_df["ID"].str.extract(r"(\d+)").astype(int)
        scenario_time = temp_df.groupby("scenario_id")["id_num"].min().sort_values()
        return scenario_time.index.tolist()

    def analyze_fold_size_asymmetry(self):
        """
        [DIAGNOSTIC 1: Expanding Window Size Asymmetry]
        
        Problem: Fold 0 has ~1/6 of training data. Fold 4 has ~5/6.
        Impact: OOF MAE is an average of predictions from models trained on
                vastly different dataset sizes → reliability degrades for early folds.
        
        Numeric Evidence (5-fold case):
          Fold 0: ~17% train, ~17% val  → model undertrained
          Fold 4: ~83% train, ~17% val  → model overtrained relative to fold 0
        
        Result: OOF MAE is NOT measuring a single consistent model state.
        """
        unique_scenarios = self.get_scenario_order(self.df_train)
        chunks = np.array_split(unique_scenarios, self.n_folds + 1)

        fold_stats = []
        train_scenarios = list(chunks[0])
        for fold in range(self.n_folds):
            val_scenarios = list(chunks[fold + 1])
            tr_idx = self.df_train[self.df_train["scenario_id"].isin(train_scenarios)].index
            val_idx = self.df_train[self.df_train["scenario_id"].isin(val_scenarios)].index

            fold_stats.append({
                "fold": fold,
                "n_train": len(tr_idx),
                "n_val": len(val_idx),
                "train_pct": len(tr_idx) / len(self.df_train),
                "train_scenario_count": len(train_scenarios),
                "val_scenario_count": len(val_scenarios),
            })
            train_scenarios.extend(val_scenarios)

        df_stats = pd.DataFrame(fold_stats)

        # [EVIDENCE] Compute train size ratio between fold 0 and fold N-1
        fold0_train_pct = df_stats.iloc[0]["train_pct"]
        fold_last_train_pct = df_stats.iloc[-1]["train_pct"]
        asymmetry_ratio = fold_last_train_pct / (fold0_train_pct + 1e-9)

        logger.info(f"[CV_ANALYZER] Fold 0 train size: {fold0_train_pct:.1%}")
        logger.info(f"[CV_ANALYZER] Fold {self.n_folds - 1} train size: {fold_last_train_pct:.1%}")
        logger.info(f"[CV_ANALYZER] Train size asymmetry ratio: {asymmetry_ratio:.2f}x")
        logger.info(f"[CV_ANALYZER] => OOF combines models from {fold0_train_pct:.0%} to {fold_last_train_pct:.0%} data. NOT a single measurement.")

        self.analysis_results["fold_size_stats"] = df_stats.to_dict(orient="records")
        self.analysis_results["train_asymmetry_ratio"] = float(asymmetry_ratio)
        return df_stats

    def compute_per_fold_distribution_similarity(self, feature_cols=None):
        """
        [DIAGNOSTIC 2: Per-Fold Distribution Similarity to Test]
        
        For each fold's validation set, compute average KS distance to test.
        The fold with LOWEST average KS is the most test-like fold.
        
        If the most test-like fold has significantly lower KS than others,
        it can be used as a proxy for LB performance.
        
        [WHY KS DISTANCE]
        KS statistic directly quantifies distribution shift — it's the same
        metric used in forensic_drift.json to identify top drift features.
        A fold with KS → 0 across all features approximates the test distribution.
        """
        if feature_cols is None:
            # [SAFE DEFAULT] Use only numeric columns present in both train and test
            feature_cols = [
                c for c in self.df_train.columns
                if c in self.df_test.columns
                and pd.api.types.is_numeric_dtype(self.df_train[c])
                and c not in ["ID", "scenario_id", "layout_id"]
            ]

        unique_scenarios = self.get_scenario_order(self.df_train)
        chunks = np.array_split(unique_scenarios, self.n_folds + 1)

        fold_ks_results = []
        train_scenarios = list(chunks[0])
        for fold in range(self.n_folds):
            val_scenarios = list(chunks[fold + 1])
            val_df = self.df_train[self.df_train["scenario_id"].isin(val_scenarios)]

            ks_scores = []
            for col in feature_cols[:30]:  # Limit to 30 features for speed
                if col in val_df.columns and col in self.df_test.columns:
                    stat, _ = ks_2samp(
                        val_df[col].dropna(),
                        self.df_test[col].dropna()
                    )
                    ks_scores.append(stat)

            mean_ks = float(np.mean(ks_scores)) if ks_scores else 1.0
            fold_ks_results.append({
                "fold": fold,
                "mean_ks_to_test": mean_ks,
                "min_ks_to_test": float(np.min(ks_scores)) if ks_scores else 1.0,
                "max_ks_to_test": float(np.max(ks_scores)) if ks_scores else 1.0,
                "n_features_checked": len(ks_scores),
            })
            train_scenarios.extend(val_scenarios)

        df_ks = pd.DataFrame(fold_ks_results).sort_values("mean_ks_to_test")

        # [FINDING] Which fold most approximates test distribution?
        best_fold = int(df_ks.iloc[0]["fold"])
        best_ks = float(df_ks.iloc[0]["mean_ks_to_test"])
        worst_ks = float(df_ks.iloc[-1]["mean_ks_to_test"])

        logger.info(f"[CV_ANALYZER] Fold {best_fold} is most test-like (mean KS={best_ks:.4f})")
        logger.info(f"[CV_ANALYZER] Fold {df_ks.iloc[-1]['fold']} is least test-like (mean KS={worst_ks:.4f})")
        logger.info(f"[CV_ANALYZER] KS range across folds: [{best_ks:.4f}, {worst_ks:.4f}]")

        if best_ks > 0.10:
            logger.warning(f"[CV_ANALYZER] WARNING: Even the best fold has mean KS={best_ks:.4f} to test.")
            logger.warning(f"[CV_ANALYZER] => No fold adequately approximates test distribution.")
            logger.warning(f"[CV_ANALYZER] => CV cannot be trusted as LB proxy under current split strategy.")

        self.analysis_results["fold_ks_to_test"] = fold_ks_results
        self.analysis_results["best_proxy_fold"] = best_fold
        self.analysis_results["best_proxy_fold_ks"] = best_ks
        return df_ks

    def compute_per_fold_adversarial_auc(self, feature_cols=None):
        """
        [DIAGNOSTIC 3: Per-Fold Adversarial AUC vs Test]
        
        For each validation fold, train an adversarial classifier to distinguish
        fold_val from test. High AUC = fold is NOT representative of test.
        
        This directly answers: "Which fold best approximates test distribution?"
        
        [NUMERIC THRESHOLD]
        AUC < 0.60 → fold is indistinguishable from test (SAFE to use as proxy)
        AUC 0.60-0.75 → fold has moderate distribution shift (USE WITH CAUTION)
        AUC > 0.75 → fold has severe distribution shift (DO NOT USE as LB proxy)
        
        [REFERENCE]
        Config.ADV_TARGET_AUC = 0.75 in config.py — same threshold used in
        CollectiveDriftPruner. Consistent thresholding principle applied here.
        """
        # [SSOT_FIX] Local import removed

        if feature_cols is None:
            feature_cols = [
                c for c in self.df_train.columns
                if c in self.df_test.columns
                and pd.api.types.is_numeric_dtype(self.df_train[c])
                and c not in ["ID", "scenario_id", "layout_id"]
            ]

        unique_scenarios = self.get_scenario_order(self.df_train)
        chunks = np.array_split(unique_scenarios, self.n_folds + 1)

        fold_adv_results = []
        train_scenarios = list(chunks[0])
        for fold in range(self.n_folds):
            val_scenarios = list(chunks[fold + 1])
            val_df = self.df_train[self.df_train["scenario_id"].isin(val_scenarios)]

            # Sample for speed
            n_val_sample = min(3000, len(val_df))
            n_test_sample = min(3000, len(self.df_test))

            safe_cols = [c for c in feature_cols if c in val_df.columns and c in self.df_test.columns][:50]

            X_val = val_df[safe_cols].sample(n_val_sample, random_state=42).fillna(-999).values
            X_test = self.df_test[safe_cols].sample(n_test_sample, random_state=42).fillna(-999).values

            X = np.vstack([X_val, X_test])
            y_adv = np.hstack([np.zeros(len(X_val)), np.ones(len(X_test))])

            # 3-fold stratified CV for AUC estimate
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            aucs = []
            for tr_idx, te_idx in skf.split(X, y_adv):
                clf = LGBMClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    random_state=42, verbose=-1, n_jobs=4
                )
                clf.fit(X[tr_idx], y_adv[tr_idx])
                preds = clf.predict_proba(X[te_idx])[:, 1]
                aucs.append(roc_auc_score(y_adv[te_idx], preds))

            avg_auc = float(np.mean(aucs))
            status = "SAFE_PROXY" if avg_auc < 0.60 else ("MODERATE_SHIFT" if avg_auc < 0.75 else "SEVERE_SHIFT")

            fold_adv_results.append({
                "fold": fold,
                "adv_auc_vs_test": avg_auc,
                "status": status,
                "n_val_samples": n_val_sample,
            })

            logger.info(f"[CV_ANALYZER] Fold {fold} Adv AUC vs Test: {avg_auc:.4f} [{status}]")
            train_scenarios.extend(val_scenarios)

        df_adv = pd.DataFrame(fold_adv_results).sort_values("adv_auc_vs_test")

        best_fold = int(df_adv.iloc[0]["fold"])
        best_auc = float(df_adv.iloc[0]["adv_auc_vs_test"])

        logger.info(f"[CV_ANALYZER] Best proxy fold: Fold {best_fold} (AUC={best_auc:.4f})")

        self.analysis_results["fold_adv_auc"] = fold_adv_results
        self.analysis_results["best_adv_proxy_fold"] = best_fold
        self.analysis_results["best_adv_proxy_auc"] = best_auc
        return df_adv


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1-B: CV Reliability Quantification
# Determines whether CV MAE can be trusted as a proxy for LB MAE.
# ─────────────────────────────────────────────────────────────────────────────

class CVReliabilityQuantifier:
    """
    [WHY THIS EXISTS]
    The experiment_registry.json shows all runs are "RISKY" with high risk scores
    despite apparently stable fold MAEs. This class quantifies the disconnect
    between CV performance and expected LB performance.
    
    [KEY METRICS COMPUTED]
    1. OOF Variance Proxy: std_ratio and mean_ratio of OOF predictions vs y_train
       Evidence: std_ratio = 0.46-0.62, mean_ratio = 0.65-0.85 across all runs
    2. Fold Consistency: CoV of fold MAEs (should be < 0.05 for stable CV)
    3. Proxy LB MAE: Weighted OOF using test-similarity weights
    
    [FAILURE EVIDENCE FROM REGISTRY]
    Run run_20260430_120906:
      - fold_stats: [9.08, 9.18, 9.73, 9.52, 9.13] → std=0.25, CoV=0.026 (STABLE)
      - mean_ratio: 0.7526 → predictions are 25% below ground truth
      - std_ratio: 0.6206 → prediction spread is 38% narrower than ground truth
      - adv_auc: 0.8072 → test distribution is significantly different from train
      => Conclusion: CV is measuring within-distribution performance. LB measures
         generalization to a shifted distribution. They are measuring different things.
    """

    def __init__(self, fold_maes, oof_preds, y_train, adv_auc, mean_ratio, std_ratio, p99_ratio):
        self.fold_maes = np.array(fold_maes)
        
        # [OOF_GAP_FIX] Mask NaN values from unvalidated initial chunk
        valid_mask = ~np.isnan(oof_preds)
        self.oof_preds = np.array(oof_preds)[valid_mask]
        self.y_train = np.array(y_train)[valid_mask]
        
        self.adv_auc = adv_auc
        self.mean_ratio = mean_ratio
        self.std_ratio = std_ratio
        self.p99_ratio = p99_ratio

    def compute_cv_reliability_score(self):
        """
        [FORMULA]
        CV Reliability Score = 1.0 / (1.0 + distribution_penalty)
        
        Where distribution_penalty combines:
          - Adversarial penalty: how much the CV domain differs from test
          - Variance compression penalty: how compressed predictions are
          - Bias penalty: systematic mean shift in predictions
        
        Score = 1.0 → CV is a perfect proxy for LB
        Score → 0.0 → CV is useless as an LB proxy
        
        [CURRENT ESTIMATE]
        Using run_20260430_120906 as reference:
          adv_auc = 0.807 → adversarial_penalty = (0.807 - 0.5) / 0.5 = 0.614
          std_ratio = 0.621 → variance_penalty = 1 - 0.621 = 0.379
          mean_ratio = 0.753 → bias_penalty = |1 - 0.753| = 0.247
          distribution_penalty = 0.614 * 1.5 + 0.379 * 1.0 + 0.247 * 0.5 = 1.423
          CV Reliability Score = 1.0 / 2.423 ≈ 0.413
        """
        # [WHY THESE WEIGHTS]
        # adversarial penalty is 1.5x because it's the most direct measure of
        # distribution shift — the CV score cannot account for domain gap.
        # variance penalty is 1.0x because compressed predictions mean the model
        # has not learned the full dynamic range of the target.
        # bias penalty is 0.5x because systematic bias can partially be corrected
        # by recalibration, unlike domain gap.
        adversarial_penalty = max(0.0, (self.adv_auc - 0.5) / 0.5) * 1.5
        variance_penalty = max(0.0, 1.0 - self.std_ratio) * 1.0
        bias_penalty = abs(1.0 - self.mean_ratio) * 0.5

        distribution_penalty = adversarial_penalty + variance_penalty + bias_penalty
        reliability_score = 1.0 / (1.0 + distribution_penalty)

        # [FOLD CONSISTENCY ANALYSIS]
        fold_cov = float(np.std(self.fold_maes) / (np.mean(self.fold_maes) + 1e-9))
        # [OOF_GAP_FIX] NaN-safe OOF MAE for reliability analysis
        valid_mask = ~np.isnan(self.oof_preds)
        if valid_mask.any():
            oof_mae = float(mean_absolute_error(self.y_train[valid_mask], self.oof_preds[valid_mask]))
        else:
            oof_mae = 0.0
            logger.warning("[RELIABILITY] No valid OOF predictions for MAE calculation.")

        logger.info(f"[RELIABILITY] Adversarial Penalty: {adversarial_penalty:.4f}")
        logger.info(f"[RELIABILITY] Variance Compression Penalty: {variance_penalty:.4f}")
        logger.info(f"[RELIABILITY] Mean Bias Penalty: {bias_penalty:.4f}")
        logger.info(f"[RELIABILITY] Total Distribution Penalty: {distribution_penalty:.4f}")
        logger.info(f"[RELIABILITY] CV Reliability Score: {reliability_score:.4f} (1.0=perfect, 0.0=useless)")
        logger.info(f"[RELIABILITY] Fold CoV: {fold_cov:.4f} (stable if < 0.05)")
        logger.info(f"[RELIABILITY] OOF MAE: {oof_mae:.4f}")

        # [INTERPRETATION]
        if reliability_score > 0.7:
            interpretation = "CV is a RELIABLE proxy for LB MAE."
        elif reliability_score > 0.4:
            interpretation = "CV is a PARTIALLY reliable proxy. Use directional guidance only."
        else:
            interpretation = "CV is UNRELIABLE as LB proxy. Distribution mismatch is dominant."

        logger.info(f"[RELIABILITY] Interpretation: {interpretation}")

        return {
            "reliability_score": reliability_score,
            "adversarial_penalty": adversarial_penalty,
            "variance_penalty": variance_penalty,
            "bias_penalty": bias_penalty,
            "fold_cov": fold_cov,
            "oof_mae": oof_mae,
            "interpretation": interpretation,
        }


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1-C: Proposed Validation Redesign
# New split strategy: Test-Tail Holdout + Distribution-Weighted OOF
# ─────────────────────────────────────────────────────────────────────────────

class TestProxyValidator:
    """
    [PROPOSED SOLUTION: Test-Tail Holdout CV]
    
    [WHY CURRENT STRATEGY FAILS]
    The expanding window validates on training-distribution scenarios.
    Test scenarios are drawn from a DIFFERENT distribution (KS 0.26-0.36 for
    key features: order_inflow_15m, robot_active, avg_items_per_order).
    
    [NEW STRATEGY]
    Replace the expanding window with a HOLDOUT split where:
      1. Identify the N_HOLDOUT most "test-like" scenarios from train data,
         using adversarial classification as the similarity metric.
      2. Use these scenarios as the single validation set.
      3. Train on ALL remaining scenarios (no expanding window complexity).
    
    This creates a validation set that explicitly approximates the test distribution,
    making CV MAE a more reliable proxy for LB MAE.
    
    [JUSTIFICATION]
    The core insight is that LB measures performance on the test distribution.
    If validation data has the same distribution as test, CV ≈ LB (by definition
    of what we're measuring). The expanding window makes CV measure within-distribution
    performance, which is provably different from LB when AUC > 0.75.
    
    [EXPECTED IMPACT]
    - CV-LB correlation should increase from ~0.05 toward > 0.60
    - Fold variance should increase (because we're now measuring a harder task)
    - But the single held-out proxy metric will be more reliable than averaged OOF
    
    [TRADE-OFF ACKNOWLEDGED]
    This reduces training data for the proxy fold. However, the alternative
    (using all folds) is provably unreliable given AUC 0.80-0.95.
    One reliable measurement > five unreliable measurements averaged together.
    """

    def __init__(self, df_train, y, df_test, n_folds=4, holdout_pct=0.20):
        """
        Args:
            n_folds: Number of CV folds for the main training loop (default 4,
                     reduced because 1 chunk is reserved for holdout).
            holdout_pct: Fraction of training scenarios to use as test-proxy holdout.
                         0.20 means the 20% most test-like scenarios become the proxy val set.
        """
        self.df_train = df_train
        self.y = y
        self.df_test = df_test
        self.n_folds = n_folds
        self.holdout_pct = holdout_pct

    def get_scenario_order(self, df):
        # [PRESERVED] Same logic as trainer.py — DO NOT MODIFY
        temp_df = df[["scenario_id", "ID"]].copy()
        temp_df["id_num"] = temp_df["ID"].str.extract(r"(\d+)").astype(int)
        scenario_time = temp_df.groupby("scenario_id")["id_num"].min().sort_values()
        return scenario_time.index.tolist()

    def identify_test_proxy_scenarios(self, feature_cols=None):
        """
        [STEP 1: Find the most test-like training scenarios]
        
        Uses adversarial classification to assign each training scenario a
        "test-similarity score". Scenarios with the highest scores are
        most likely to produce validation performance that correlates with LB.
        
        [METHOD]
        1. Train an adversarial classifier (train=0, test=1) on scenario-level
           aggregated features.
        2. Predict the probability of each training scenario being "test-like".
        3. Rank scenarios and select the top HOLDOUT_PCT as the proxy val set.
        
        [WHY SCENARIO-LEVEL AGGREGATION]
        Individual rows within a scenario are highly correlated (same warehouse,
        same time block). Aggregating to scenario-level prevents data leakage
        within the adversarial classifier.
        """
        # [SSOT_FIX] Local import removed

        if feature_cols is None:
            feature_cols = [
                c for c in self.df_train.columns
                if c in self.df_test.columns
                and pd.api.types.is_numeric_dtype(self.df_train[c])
                and c not in ["ID", "scenario_id", "layout_id"]
            ]

        # Aggregate to scenario level
        train_scenario_agg = self.df_train.groupby("scenario_id")[feature_cols].agg(
            ["mean", "std", "max", "min"]
        )
        train_scenario_agg.columns = [f"{col}_{stat}" for col, stat in train_scenario_agg.columns]
        train_scenario_agg = train_scenario_agg.fillna(-999)

        test_scenario_agg = self.df_test.groupby("scenario_id")[feature_cols].agg(
            ["mean", "std", "max", "min"]
        )
        test_scenario_agg.columns = [f"{col}_{stat}" for col, stat in test_scenario_agg.columns]
        test_scenario_agg = test_scenario_agg.fillna(-999)

        # Ensure same columns
        common_cols = [c for c in train_scenario_agg.columns if c in test_scenario_agg.columns]
        X_train_sc = train_scenario_agg[common_cols].values
        X_test_sc = test_scenario_agg[common_cols].values

        X = np.vstack([X_train_sc, X_test_sc])
        y_adv = np.hstack([np.zeros(len(X_train_sc)), np.ones(len(X_test_sc))])

        # Train adversarial classifier
        clf = LGBMClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=42, verbose=-1, n_jobs=4
        )
        clf.fit(X, y_adv)

        # Score each training scenario
        train_test_probs = clf.predict_proba(X_train_sc)[:, 1]
        train_scenario_ids = list(train_scenario_agg.index)

        scenario_scores = pd.DataFrame({
            "scenario_id": train_scenario_ids,
            "test_similarity_score": train_test_probs
        }).sort_values("test_similarity_score", ascending=False)

        # Select top HOLDOUT_PCT as proxy holdout
        n_holdout = max(1, int(len(scenario_scores) * self.holdout_pct))
        proxy_scenarios = set(scenario_scores.head(n_holdout)["scenario_id"].tolist())

        # [VALIDATION] Check that proxy scenarios have lower KS to test
        proxy_df = self.df_train[self.df_train["scenario_id"].isin(proxy_scenarios)]
        remaining_df = self.df_train[~self.df_train["scenario_id"].isin(proxy_scenarios)]

        proxy_ks_scores = []
        remaining_ks_scores = []
        for col in feature_cols[:20]:
            if col in proxy_df.columns and col in self.df_test.columns:
                ks_proxy, _ = ks_2samp(proxy_df[col].dropna(), self.df_test[col].dropna())
                ks_remain, _ = ks_2samp(remaining_df[col].dropna(), self.df_test[col].dropna())
                proxy_ks_scores.append(ks_proxy)
                remaining_ks_scores.append(ks_remain)

        mean_ks_proxy = float(np.mean(proxy_ks_scores)) if proxy_ks_scores else 1.0
        mean_ks_remaining = float(np.mean(remaining_ks_scores)) if remaining_ks_scores else 1.0

        logger.info(f"[PROXY_VALIDATOR] Proxy val set size: {len(proxy_df)} rows ({n_holdout} scenarios)")
        logger.info(f"[PROXY_VALIDATOR] Proxy val mean KS to test: {mean_ks_proxy:.4f}")
        logger.info(f"[PROXY_VALIDATOR] Remaining train mean KS to test: {mean_ks_remaining:.4f}")
        logger.info(f"[PROXY_VALIDATOR] KS improvement: {mean_ks_remaining - mean_ks_proxy:.4f} lower in proxy vs remaining")

        if mean_ks_proxy >= mean_ks_remaining:
            logger.warning(f"[PROXY_VALIDATOR] WARNING: Proxy selection did NOT reduce KS to test.")
            logger.warning(f"[PROXY_VALIDATOR] => Training distribution is uniformly different from test.")
            logger.warning(f"[PROXY_VALIDATOR] => No subset of training data can reliably approximate test.")

        return proxy_scenarios, scenario_scores, {
            "mean_ks_proxy_to_test": mean_ks_proxy,
            "mean_ks_remaining_to_test": mean_ks_remaining,
            "n_proxy_scenarios": n_holdout,
            "n_remaining_scenarios": len(scenario_scores) - n_holdout,
        }

    def get_test_proxy_splits(self, proxy_scenarios):
        """
        [STEP 2: Generate splits using test-proxy holdout]
        
        Structure:
          - Holdout set: proxy_scenarios (most test-like) → used for reliability measurement
          - Main set: remaining scenarios → standard k-fold CV for model selection
        
        Returns:
          - holdout_idx: indices of proxy validation set (use for LB correlation estimate)
          - cv_splits: list of (tr_idx, val_idx) for k-fold on remaining data
        
        [TEMPORAL LEAKAGE PREVENTION]
        Within the main CV splits, we still use time-aware ordering to prevent
        temporal leakage (same guarantee as the existing expanding window).
        The key change is that we ALSO have a holdout set that approximates test.
        """
        holdout_idx = self.df_train[self.df_train["scenario_id"].isin(proxy_scenarios)].index.values
        main_df = self.df_train[~self.df_train["scenario_id"].isin(proxy_scenarios)]

        # Time-aware splits on remaining data
        temp_df = main_df[["scenario_id", "ID"]].copy()
        temp_df["id_num"] = temp_df["ID"].str.extract(r"(\d+)").astype(int)
        scenario_time = temp_df.groupby("scenario_id")["id_num"].min().sort_values()
        unique_scenarios_main = scenario_time.index.tolist()
        chunks = np.array_split(unique_scenarios_main, self.n_folds + 1)

        cv_splits = []
        train_scenarios = list(chunks[0])
        for fold in range(self.n_folds):
            val_scenarios = list(chunks[fold + 1])
            tr_idx = main_df[main_df["scenario_id"].isin(train_scenarios)].index.values
            val_idx = main_df[main_df["scenario_id"].isin(val_scenarios)].index.values

            if len(tr_idx) > 0 and len(val_idx) > 0:
                cv_splits.append((tr_idx, val_idx))

            train_scenarios.extend(val_scenarios)

        logger.info(f"[PROXY_VALIDATOR] Holdout size: {len(holdout_idx)} rows")
        logger.info(f"[PROXY_VALIDATOR] Main CV folds generated: {len(cv_splits)}")

        return holdout_idx, cv_splits


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1-D: OOF Reliability Report
# Generates a comprehensive diagnosis of CV-LB mismatch
# ─────────────────────────────────────────────────────────────────────────────

def generate_cv_reliability_report(
    fold_maes,
    oof_preds,
    y_train,
    adv_auc,
    mean_ratio,
    std_ratio,
    p99_ratio,
    output_path=None
):
    """
    [ENTRY POINT FOR CV RELIABILITY DIAGNOSIS]
    
    Generates a structured report that quantifies:
    1. Whether current CV is biased
    2. The root cause of CV-LB mismatch
    3. The expected impact of the proposed redesign
    
    [INPUT FROM EXPERIMENT REGISTRY]
    Reference run: run_20260430_120906
      fold_stats: [9.08, 9.18, 9.73, 9.52, 9.13]
      adv_auc: 0.8072
      mean_ratio: 0.7526
      std_ratio: 0.6206
      p99_ratio: 0.6070
    
    [USAGE]
    Call this after trainer.fit_leakage_free_model() to get a reliability report.
    """
    quantifier = CVReliabilityQuantifier(
        fold_maes=fold_maes,
        oof_preds=oof_preds,
        y_train=y_train,
        adv_auc=adv_auc,
        mean_ratio=mean_ratio,
        std_ratio=std_ratio,
        p99_ratio=p99_ratio,
    )
    result = quantifier.compute_cv_reliability_score()

    report_lines = [
        "=" * 70,
        "[CV RELIABILITY RECONSTRUCTION REPORT]",
        "=" * 70,
        "",
        "[CURRENT CV FAILURE DIAGNOSIS]",
        f"  Fold MAEs: {[round(m, 4) for m in fold_maes]}",
        f"  Fold MAE std: {np.std(fold_maes):.4f} (CoV: {result['fold_cov']:.4f})",
        f"  OOF MAE: {result['oof_mae']:.4f}",
        f"  Adversarial AUC (train vs test): {adv_auc:.4f}",
        f"  Mean Ratio (pred/true): {mean_ratio:.4f}  [Target: 1.0]",
        f"  Std Ratio (pred/true): {std_ratio:.4f}   [Target: 1.0]",
        f"  P99 Ratio (pred/true): {p99_ratio:.4f}  [Target: 1.0]",
        "",
        "[ROOT CAUSE OF CV-LB MISMATCH]",
        "  (1) EXPANDING WINDOW ASYMMETRY:",
        f"      Fold 0 trains on ~17% of data; fold {len(fold_maes)-1} on ~83%.",
        "      OOF averages predictions from fundamentally different model states.",
        "      This creates WITHIN-DISTRIBUTION measurement artifacts.",
        "",
        "  (2) HOMOGENEOUS VALIDATION DOMAIN:",
        "      All validation folds use training-distribution scenarios.",
        "      Test distribution differs significantly (KS 0.26-0.36 for key features).",
        "      Validation domain ≠ Test domain → CV measures the wrong thing.",
        "",
        "  (3) COLLECTIVE DRIFT:",
        "      Individual features pass KS filter but collectively achieve AUC > 0.80.",
        "      The combination of 200+ weakly-shifted features creates a strong",
        "      train-test discriminant that CV cannot detect.",
        "",
        "[RELIABILITY QUANTIFICATION]",
        f"  Adversarial Penalty: {result['adversarial_penalty']:.4f}",
        f"  Variance Compression Penalty: {result['variance_penalty']:.4f}",
        f"  Mean Bias Penalty: {result['bias_penalty']:.4f}",
        f"  CV Reliability Score: {result['reliability_score']:.4f}",
        f"  Interpretation: {result['interpretation']}",
        "",
        "[PROPOSED VALIDATION REDESIGN]",
        "  Strategy: Test-Tail Holdout CV (TestProxyValidator)",
        "  1. Identify most test-like training scenarios via adversarial scoring.",
        "  2. Reserve these as the PROXY HOLDOUT (single held-out val set).",
        "  3. Run standard k-fold CV on remaining data for model selection.",
        "  4. Use PROXY HOLDOUT MAE as the primary LB correlation metric.",
        "",
        "[EXPECTED IMPACT ON LB CORRELATION]",
        "  Current: CV-LB correlation ≈ 0.05 (near-zero, as reported)",
        "  Expected: Proxy holdout MAE should achieve correlation > 0.60 with LB",
        "            because validation domain explicitly approximates test domain.",
        "  Caveat: If mean_ks_proxy_to_test > 0.10, no subset of train can",
        "          reliably approximate test → problem is in data collection,",
        "          not in the CV split strategy.",
        "",
        "=" * 70,
    ]

    report_text = "\n".join(report_lines)
    logger.info("\n" + report_text)

    if output_path:
        # [SSOT_FIX] Local import removed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        logger.info(f"[CV_RELIABILITY] Report saved to {output_path}")

    result["report_text"] = report_text
    return result
