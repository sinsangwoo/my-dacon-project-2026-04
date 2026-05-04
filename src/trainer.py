import logging
import numpy as np
import pandas as pd
import os
import gc
import pickle
import re
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from .config import Config
from .schema import FEATURE_SCHEMA
from src.distribution import DistributionAuditor
from . import utils
from .utils import (
    save_npy, load_npy, memory_guard, DriftShieldScaler, 
    build_metrics, calculate_risk_score, 
    save_json, calculate_std_ratio
)
from .data_loader import (
    apply_latent_features,
    SuperchargedPCAReconstructor,
    add_time_series_features,
    add_extreme_detection_features,
    get_protected_candidates,
    FEATURE_LINEAGE,
    add_scenario_context_features,
)
from .signal_validation import CollectiveDriftPruner, SignalValidator
from .cv_reliability import CVPipelineAnalyzer, CVReliabilityQuantifier, generate_cv_reliability_report
from .intelligence import ExperimentIntelligence

logger = logging.getLogger(__name__)

def _sigmoid_gate(p, k, theta):
    """Sigmoid-based blending gate to sharpen tail transition."""
    return 1.0 / (1.0 + np.exp(-k * (p - theta)))

def _multi_signal_blend(p, preds_t, preds_nt):
    """
    [ARCHITECTURAL RECONSTRUCTION: MULTI-SIGNAL GATING]
    Relaxed to allow tail signal to pass when either classifier is highly confident
    OR the regressors show a massive divergence with moderate confidence.
    """
    # Condition A: Classifier is highly confident
    cond_a = p > 0.65
    # Condition B: Regressors disagree massively AND classifier isn't totally dead
    cond_b = (preds_t > (preds_nt * 2.0)) & (p > 0.4)
    
    # Base weight is less suppressive (Config.BLENDING_POWER = 1.5 instead of 2.0)
    base_weight = p ** Config.BLENDING_POWER
    
    # Apply aggressive Sigmoid when EITHER condition is met
    multi_signal_mask = cond_a | cond_b
    
    final_weight = base_weight.copy()
    if np.any(multi_signal_mask):
        final_weight[multi_signal_mask] = _sigmoid_gate(
            p[multi_signal_mask], 
            Config.SIGMOID_K, 
            Config.SIGMOID_THETA
        )
    return final_weight

class TailRiskController:
    """Controls aggressive tail predictions when classifier confidence is low."""
    def __init__(self, gap_threshold, clf_floor, damp_factor):
        self.gap_threshold = gap_threshold
        self.clf_floor = clf_floor
        self.damp_factor = damp_factor

    def compute_gap(self, preds_t, preds_nt):
        """Measures divergence between tail and non-tail regressors."""
        return np.abs(preds_t - preds_nt) / (preds_nt + 1e-6)

    def apply(self, weight, p, gap):
        """Applies damping to weight if tail prediction is risky."""
        danger_mask = (gap > self.gap_threshold) & (p < self.clf_floor)
        
        # If in danger, damp the weight proportional to distrust
        # danger_intensity: how much gap exceeds threshold (maxed at 2x threshold)
        danger_intensity = np.clip((gap - self.gap_threshold) / self.gap_threshold, 0, 2)
        # p_distrust: how much p is below floor
        p_distrust = np.clip((self.clf_floor - p) / self.clf_floor, 0, 1)
        
        # damping range: [damp_factor, 1.0]
        damping = 1.0 - (danger_intensity * p_distrust * (1.0 - self.damp_factor))
        damping = np.clip(damping, self.damp_factor, 1.0)
        
        return weight * damping

# Base columns that must be present in every fold (raw features)
BASE_COLS = FEATURE_SCHEMA["raw_features"]


class Trainer:
    def __init__(self, df_train, y, df_test, groups=None, full_df=None, test_df=None, drift_df=None, ks_threshold=None, manifest=None):
        self.df_train = df_train if df_train is not None else full_df
        self.y = y
        self.df_test = df_test if df_test is not None else test_df
        self.groups = groups
        self.full_df = self.df_train
        self.test_df = self.df_test
        self.drift_df = drift_df
        self.ks_threshold = ks_threshold
        self.manifest = manifest # [v7.0] Stability manifest for inference fallback
        self.models = []
        self.oof = np.full(len(self.y), np.nan)
        self.oof_base = np.full(len(self.y), np.nan) # [v4.0] Baseline tracking
        self.test_preds = {}
        self.fold_scores = []
        self.fold_stats = []
        self.metadata = {}

    def get_scenario_order(self, df):
        """Sort scenarios chronologically based on the smallest ID within each scenario."""
        temp_df = df[["scenario_id", "ID"]].copy()
        temp_df["id_num"] = temp_df["ID"].str.extract(r"(\d+)").astype(int)
        scenario_time = temp_df.groupby("scenario_id")["id_num"].min().sort_values()
        return scenario_time.index.tolist()

    def _get_time_aware_splits(self):
        """
        [v4.1] Warm-Start Expanding Window Split
        Why: Fold 0 previously trained on ~17% data, leading to severe underfitting 
             and biased OOF metrics.
        Fix: Use NFOLDS + 2 chunks and start with 2 chunks for Fold 0 training.
        """
        unique_scenarios = self.get_scenario_order(self.df_train)
        # Increase granularity to allow for a larger initial training set
        chunks = np.array_split(unique_scenarios, Config.NFOLDS + 2)
        splits = []
        
        # Initial training set: Chunks 0 and 1
        train_scenarios = list(chunks[0]) + list(chunks[1])
        
        for fold in range(Config.NFOLDS):
            # Validation set starts from Chunk 2
            val_scenarios = list(chunks[fold + 2])
            tr_idx = self.df_train[self.df_train["scenario_id"].isin(train_scenarios)].index.values
            val_idx = self.df_train[self.df_train["scenario_id"].isin(val_scenarios)].index.values
            
            if len(tr_idx) > 0 and len(val_idx) > 0:
                tr_id_max = (
                    self.df_train.iloc[tr_idx]["ID"].str.extract(r"(\d+)").astype(int).max().iloc[0]
                )
                val_id_min = (
                    self.df_train.iloc[val_idx]["ID"].str.extract(r"(\d+)").astype(int).min().iloc[0]
                )
                assert (
                    tr_id_max < val_id_min
                ), f"Temporal leakage in Fold {fold}: train_max_id {tr_id_max} >= val_min_id {val_id_min}"
            
            splits.append((tr_idx, val_idx))
            # Expand training set with the validation data we just used
            train_scenarios.extend(val_scenarios)
            
        return splits

    def _compute_fold_layout_stats(self, tr_df):
        layout_target_cols = [
            "order_inflow_15m",
            "robot_utilization",
            "congestion_score",
            "avg_trip_distance",
            "pack_utilization",
        ]
        layout_stats = {}
        if "layout_id" in tr_df.columns:
            for col in layout_target_cols:
                if col in tr_df.columns:
                    grp = tr_df.groupby("layout_id")[col]
                    layout_stats[f"{col}_layout_mean"] = grp.mean().to_dict()
                    layout_stats[f"{col}_layout_std"] = grp.std().to_dict()
                    if "layout_type" in tr_df.columns:
                        type_grp = tr_df.groupby("layout_type")[col]
                        layout_stats[f"{col}_type_mean"] = type_grp.mean().to_dict()
                    layout_stats[f"{col}_global_mean"] = float(tr_df[col].mean())
                    layout_stats[f"{col}_global_std"] = float(tr_df[col].std())
        return layout_stats

    def _apply_layout_stats(self, df, layout_stats):
        df = df.copy()
        for feat, mapping in layout_stats.items():
            if not (feat.endswith("_layout_mean") or feat.endswith("_layout_std")):
                continue
            target_col = feat
            df[target_col] = df["layout_id"].map(mapping)
            base_col = feat.replace("_layout_mean", "").replace("_layout_std", "")
            stat_type = "mean" if "mean" in feat else "std"
            if df[target_col].isna().any() and f"{base_col}_type_{stat_type}" in layout_stats:
                type_mapping = layout_stats[f"{base_col}_type_{stat_type}"]
                if "layout_type" in df.columns:
                    df[target_col] = df[target_col].fillna(df["layout_type"].map(type_mapping))
            global_fallback = layout_stats.get(f"{base_col}_global_{stat_type}", 0.0)
            df[target_col] = df[target_col].fillna(global_fallback)
        return df

    def fit_raw_model(self):
        logger.info("[TRAINER] Starting raw baseline training...")
        raw_cols = [c for c in FEATURE_SCHEMA["raw_features"] if c in self.df_train.columns]
        scaler = DriftShieldScaler()
        scaler.fit(self.df_train, raw_cols)
        train_df_drifted = scaler.transform(self.df_train, raw_cols)
        # [SSOT_FIX] Local import removed
        norm_scaler = StandardScaler()
        train_df_scaled = train_df_drifted.copy()
        train_df_scaled[raw_cols] = norm_scaler.fit_transform(train_df_drifted[raw_cols])
        splits = self._get_time_aware_splits()
        for fold, (tr_idx, val_idx) in enumerate(splits):
            logger.info(f"[RAW_FOLD {fold}] Processing...")
            X_tr = train_df_scaled.iloc[tr_idx][raw_cols].values.astype(np.float32)
            y_tr = self.y[tr_idx]
            X_val = train_df_scaled.iloc[val_idx][raw_cols].values.astype(np.float32)
            y_val = self.y[val_idx]
            
            y_tr_log = np.log1p(y_tr)
            y_val_log = np.log1p(y_val)
            
            model = LGBMRegressor(**Config.RAW_LGBM_PARAMS)
            utils.SAFE_FIT(model, X_tr, y_tr_log, eval_set=[(X_val, y_val_log)], eval_metric="mae")
            self.oof[val_idx] = np.expm1(utils.SAFE_PREDICT(model, X_val))
            fold_mae = mean_absolute_error(y_val, self.oof[val_idx])
            logger.info(f"[RAW_FOLD {fold}] MAE: {fold_mae:.4f}")
        # [OOF_GAP_FIX] NaN-safe MAE calculation
        valid_mask = ~np.isnan(self.oof)
        if valid_mask.any():
            overall_mae = mean_absolute_error(self.y[valid_mask], self.oof[valid_mask])
            logger.info(f"[TRAINER] Raw baseline complete. Overall MAE: {overall_mae:.4f}")
        else:
            overall_mae = 0.0
            logger.warning("[TRAINER] No valid OOF predictions. MAE set to 0.0")
            
        # [PHASE_ARTIFACT_SYNC] Persist OOF after baseline completion
        save_npy(self.oof, f"{Config.PREDICTIONS_PATH}/oof_final.npy")
        
        return overall_mae, self.oof

    def fit_leakage_free_model(self):
        """Two‑pass training ensuring identical feature set across folds and no temporal leakage."""
        logger.info("[TRAINER] Starting structural corrected leakage‑free training (2‑pass)...")
        # [SSOT] Initialize drift data and thresholds for use in both PCA and pruning
        drift_df = self.drift_df if self.drift_df is not None else pd.DataFrame(columns=["feature", "ks_stat"])
        # [FORENSIC #32] Relax PCA threshold slightly from 0.15 to 0.20 to avoid trivial dimensionality reduction
        ks_threshold = self.ks_threshold if self.ks_threshold is not None else 0.20
        
        raw_cols = [c for c in FEATURE_SCHEMA["raw_features"] if c in self.df_train.columns]
        splits = self._get_time_aware_splits()
        # ---------- First Pass: collect stable features per fold ----------
        per_fold_features = []
        per_fold_data = []
        for fold, (tr_idx, val_idx) in enumerate(splits):
            logger.info(f"━━━━━━━━━━━━━━━━ FIRST‑PASS FOLD {fold} ━━━━━━━━━━━━━━━━")
            # [FORENSIC #24] Do NOT clear FEATURE_LINEAGE per fold — lineage must persist for conditional protection
            tr_df = self.df_train.iloc[tr_idx].copy()
            val_df = self.df_train.iloc[val_idx].copy()
            test_df_fold = self.df_test.copy()
            # Layout stats (MUST be fold-specific)
            layout_stats = self._compute_fold_layout_stats(tr_df)
            tr_df = self._apply_layout_stats(tr_df, layout_stats)
            val_df = self._apply_layout_stats(val_df, layout_stats)
            test_df_fold = self._apply_layout_stats(test_df_fold, layout_stats)
            
            # [FORENSIC #27/#28] RESTORED calls to prevent SCHEMA_CRITICAL_FAILURE.
            # While redundant if Phase 2 kept them, they are MANDATORY if Phase 2 dropped them
            # to ensure the full FEATURE_SCHEMA is populated for downstream audit checks.
            tr_df = add_time_series_features(tr_df)
            val_df = add_time_series_features(val_df)
            test_df_fold = add_time_series_features(test_df_fold)
            
            tr_df = add_scenario_context_features(tr_df)
            val_df = add_scenario_context_features(val_df)
            test_df_fold = add_scenario_context_features(test_df_fold)
            
            bucket_edges = {}
            extreme_quantiles = {}
            tr_df = add_extreme_detection_features(tr_df, extreme_quantiles=extreme_quantiles, bucket_edges=bucket_edges)
            val_df = add_extreme_detection_features(val_df, extreme_quantiles=extreme_quantiles, bucket_edges=bucket_edges)
            test_df_fold = add_extreme_detection_features(test_df_fold, extreme_quantiles=extreme_quantiles, bucket_edges=bucket_edges)
            # Scaling
            scaler = DriftShieldScaler()
            scaler.fit(tr_df, raw_cols)
            # [SSOT_FIX] Local import removed
            norm_scaler = StandardScaler()
            
            tr_df_scaled = scaler.transform(tr_df, raw_cols)
            tr_df_scaled[raw_cols] = norm_scaler.fit_transform(tr_df_scaled[raw_cols])
            
            val_df_scaled = scaler.transform(val_df, raw_cols)
            val_df_scaled[raw_cols] = norm_scaler.transform(val_df_scaled[raw_cols])
            
            test_df_scaled = scaler.transform(test_df_fold, raw_cols)
            test_df_scaled[raw_cols] = norm_scaler.transform(test_df_scaled[raw_cols])
            
            # [FORENSIC #13] Persist train-derived column means for Phase 7
            train_col_means = {col: float(tr_df[col].mean()) for col in raw_cols if col in tr_df.columns}

            # [MISSION 3.1: PCA DRIFT FILTERING]
            # Why: Including drifted features in PCA contaminates ALL latent features.
            # Fix: Dynamic filtering of PCA inputs based on KS statistics.
            pca_inputs = list(Config.PCA_INPUT_COLS)
            if not drift_df.empty:
                ks_lookup = drift_df.set_index('feature')['ks_stat'].to_dict()
                pca_inputs = [f for f in pca_inputs if ks_lookup.get(f, 0) <= ks_threshold]
                logger.info(f"[PCA_AUDIT] Using {len(pca_inputs)}/30 stable features for reconstruction.")
            
            if len(pca_inputs) < 5:
                logger.warning("[PCA_AUDIT] Too many drifted PCA inputs. Falling back to all inputs to prevent dimension collapse.")
                pca_inputs = list(Config.PCA_INPUT_COLS)

            reconstructor = SuperchargedPCAReconstructor(input_dim=len(pca_inputs))
            reconstructor.fit(tr_df_scaled[pca_inputs].values, pca_cols=pca_inputs)
            reconstructor.build_fold_cache(tr_df_scaled, pca_cols=pca_inputs)
            self.metadata["pca_input_count"] = len(pca_inputs)
            
            # [MISSION 2: EMBED RESTORATION]
            # Generate ALL latent features as candidates before pruning
            tr_df_scaled = apply_latent_features(tr_df_scaled, reconstructor, scaler=None, selected_features=None, is_train=True)
            val_df_scaled = apply_latent_features(val_df_scaled, reconstructor, scaler=None, selected_features=None, is_train=False)
            test_df_scaled = apply_latent_features(test_df_scaled, reconstructor, scaler=None, selected_features=None, is_train=False)
            
            # [MISSION 2.5: 100% AUDIT COVERAGE]
            # Why: Static schema lookup caused dynamically added features to bypass drift audit.
            # Fix: Audit EVERY column present in the dataframe before model entry.
            initial_features = [c for c in tr_df_scaled.columns if c not in Config.ID_COLS and c != Config.TARGET]
            
            # [SSOT] Filter by KS before adversarial pruning
            if not drift_df.empty:
                # Use absolute fallback if a feature is missing from drift_df
                initial_features = [f for f in initial_features if drift_df.set_index('feature').get('ks_stat', {}).get(f, 0) <= ks_threshold]
            
            protected_cols = set(FEATURE_SCHEMA["raw_features"]).intersection(set(tr_df_scaled.columns))
            # [MISSION 3: AGGRESSIVE COLLECTIVE DRIFT SUPPRESSION]
            # Why: Previous settings (5/5) were too conservative for 700+ features (AUC stayed at 0.98).
            # Fix: Increase iterations and step size to force AUC below 0.75.
            # Perform iterative pruning to neutralize domain shift artifacts
            # [MISSION_CRITICAL] Prune relative to TEST distribution, not validation fold
            # Pruning relative to validation (within-distribution) fails to detect actual domain shift.
            # [KEYERROR_FIX] Use test_df_scaled (processed) instead of raw self.df_test
            drift_proxy = test_df_scaled if test_df_scaled is not None else val_df_scaled
            
            # [FORENSIC #31] Pass fold index + 42 to ensure different sampling per fold
            pruner = CollectiveDriftPruner(target_auc=Config.ADV_TARGET_AUC, max_iterations=20, prune_step=50)
            stable_features, _ = pruner.prune(tr_df_scaled, drift_proxy, initial_features, 
                                             protected_cols=protected_cols, base_seed=42 + fold)
            
            # Latent features are already applied, just filter by stable set
            tr_df_full = tr_df_scaled[list(set(stable_features) | set(Config.ID_COLS))]
            val_df_full = val_df_scaled[list(set(stable_features) | set(Config.ID_COLS))]
            per_fold_features.append(set(stable_features))
            per_fold_data.append({
                "tr_df_full": tr_df_full,
                "val_df_full": val_df_full,
                "test_df_full": apply_latent_features(test_df_scaled, reconstructor, scaler=None, selected_features=stable_features, is_train=False),
                "scaler": scaler,
                "norm_scaler": norm_scaler,
                "reconstructor": reconstructor,
                "layout_stats": layout_stats,
                "bucket_edges": bucket_edges,
                "extreme_quantiles": extreme_quantiles,
                "train_col_means": train_col_means,
                "raw_cols": raw_cols,
                "fold": fold,
            })
            del tr_df, val_df, test_df_fold, tr_df_scaled
            gc.collect()
        # Compute global intersection of stable features
        if not per_fold_features:
            raise RuntimeError("No folds processed in first pass.")
        global_features = set.intersection(*per_fold_features)
        if not global_features:
            raise RuntimeError("Feature set intersection across folds is empty – structural inconsistency.")
        
        sorted_features = sorted(list(global_features))
        logger.info(f"[FEATURE INTERSECTION] Global stable feature count: {len(sorted_features)}")
        
        # Populate metadata for intelligence report
        self.metadata["num_features"] = len(sorted_features)
        self.metadata["feature_alignment"] = True # Guaranteed by sorted_features
        self.metadata["pruned_count"] = len(initial_features) - len(sorted_features)
        self.metadata["prune_rate"] = self.metadata["pruned_count"] / (len(initial_features) + 1e-9)

        # [OPTIMIZATION] Compute adversarial weights once for the global feature set
        global_adv_weights = None
        if Config.USE_ADVERSARIAL_WEIGHTING:
            global_adv_weights = self.compute_adversarial_weights(sorted(list(global_features)))

        # ---------- Second Pass: train using intersected features ----------
        for fold_data in per_fold_data:
            fold = fold_data["fold"]
            logger.info(f"━━━━━━━━━━━━━━━━ SECOND‑PASS FOLD {fold} ━━━━━━━━━━━━━━━━")
            # [DETERMINISTIC_ALIGNMENT_FIX] Always use sorted() for set->list conversion
            sorted_features = sorted(list(global_features))
            tr_df_full = fold_data["tr_df_full"][sorted_features]
            val_df_full = fold_data["val_df_full"][sorted_features]
            test_df_full = fold_data["test_df_full"][sorted_features]
            scaler = fold_data["scaler"]
            norm_scaler = fold_data["norm_scaler"]
            X_tr_fold = tr_df_full.values.astype(np.float32)
            tr_idx = splits[fold][0]
            val_idx = splits[fold][1]
            y_tr = self.y[tr_idx]
            X_val_fold = val_df_full.values.astype(np.float32)
            y_val = self.y[val_idx]
            
            # Optional adversarial weighting
            sample_weight = None
            if Config.USE_ADVERSARIAL_WEIGHTING:
                sample_weight = global_adv_weights[tr_idx]
            # Target‑aware weighting
            if Config.USE_TARGET_AWARE_WEIGHTING:
                y_p90 = np.quantile(y_tr, 0.90)
                tail_weight = np.where(
                    y_tr > y_p90,
                    1.0 + Config.TARGET_AWARE_ALPHA * (y_tr - y_p90) / (np.std(y_tr) + 1e-6),
                    1.0,
                )
                tail_weight = np.clip(tail_weight, 1.0, Config.TARGET_AWARE_MAX_WEIGHT)
                sample_weight = tail_weight if sample_weight is None else sample_weight * tail_weight
            # Log‑transform target
            y_tr_log = np.log1p(y_tr)
            y_val_log = np.log1p(y_val)
            
            logger.info(f"[DEBUG_SHAPE] Fold {fold} | X_tr: {X_tr_fold.shape} | y_tr: {y_tr_log.shape} | weight: {sample_weight.shape if sample_weight is not None else 'None'}")
            
            # 2-Stage Model Training
            if Config.USE_2STAGE_MODEL:
                # Stage 1: Classifier
                # Stage 1: Tail Classifier (Q90)
                q90_val = np.quantile(y_tr, 0.90)
                y_tr_binary = (y_tr >= q90_val).astype(int)
                y_val_binary = (y_val >= q90_val).astype(int)
                
                # [TARGETING RECONSTRUCTION] Quarantine POISON features for classifier ONLY
                clf_features = [f for f in sorted_features if f not in getattr(Config, 'CLASSIFIER_POISON_FEATURES', [])]
                X_tr_clf = tr_df_full[clf_features].values.astype(np.float32)
                X_val_clf = val_df_full[clf_features].values.astype(np.float32)
                
                model_clf = LGBMClassifier(**Config.TAIL_CLASSIFIER_PARAMS)
                # [SAFE_GATEWAY] Enforce numpy-only, float32-only, zero-hardcode name contract
                utils.SAFE_FIT(model_clf, X_tr_clf, y_tr_binary, eval_set=[(X_val_clf, y_val_binary)], eval_metric="auc")
                
                # Stage 2: Regressors (Tail vs Non-Tail)
                tail_mask = y_tr >= q90_val
                # [ARCHITECTURAL RECONSTRUCTION] Prune Destructive Features
                tail_features = [f for f in sorted_features if f not in Config.TAIL_DESTRUCTIVE_FEATURES]
                X_tr_tail = tr_df_full[tail_features].values.astype(np.float32)
                X_val_tail = val_df_full[tail_features].values.astype(np.float32)
                
                # Tail Regressor
                model_tail = LGBMRegressor(**Config.TAIL_REGRESSOR_PARAMS)
                utils.SAFE_FIT(
                    model_tail, X_tr_tail[tail_mask], np.log1p(y_tr[tail_mask]),
                    sample_weight=sample_weight[tail_mask] if sample_weight is not None else None,
                    eval_set=[(X_val_tail, np.log1p(y_val))], eval_metric="mae",
                    callbacks=[lambda env: None]
                )
                
                # Non-Tail Regressor
                non_tail_params = Config.NON_TAIL_REGRESSOR_PARAMS.copy()
                model_non_tail = LGBMRegressor(**non_tail_params)
                utils.SAFE_FIT(
                    model_non_tail, X_tr_fold[~tail_mask], np.log1p(y_tr[~tail_mask]),
                    sample_weight=sample_weight[~tail_mask] if sample_weight is not None else None,
                    eval_set=[(X_val_fold, np.log1p(y_val))], eval_metric="mae",
                    callbacks=[lambda env: None]
                )
                
                # [FINAL SIMPLICITY: POWER-DAMPED BLENDING]
                p_val_raw = utils.SAFE_PREDICT_PROBA(model_clf, X_val_clf)[:, 1]
                preds_t = np.expm1(utils.SAFE_PREDICT(model_tail, X_val_tail))
                preds_nt = np.expm1(utils.SAFE_PREDICT(model_non_tail, X_val_fold))
                
                # [ARCHITECTURAL RECONSTRUCTION: MULTI-SIGNAL GATING]
                final_weight = _multi_signal_blend(p_val_raw, preds_t, preds_nt)
                
                risk_ctrl = TailRiskController(
                    gap_threshold=Config.GAP_THRESHOLD,
                    clf_floor=Config.CLF_TRUST_FLOOR,
                    damp_factor=Config.SAFE_DAMP_FACTOR
                )
                
                gap = risk_ctrl.compute_gap(preds_t, preds_nt)
                final_weight = risk_ctrl.apply(final_weight, p_val_raw, gap)
                
                self.oof_base[val_idx] = preds_nt
                self.oof[val_idx] = final_weight * preds_t + (1.0 - final_weight) * preds_nt
                
                model = {"clf": model_clf, "tail": model_tail, "non_tail": model_non_tail, "risk_ctrl": risk_ctrl, "clf_features": clf_features, "tail_features": tail_features}
            else:
                model = LGBMRegressor(**Config.RAW_LGBM_PARAMS, early_stopping_rounds=50)
                utils.SAFE_FIT(
                    model, X_tr_fold, np.log1p(y_tr),
                    sample_weight=sample_weight,
                    eval_set=[(X_val_fold, np.log1p(y_val))],
                    eval_metric="mae",
                )
                self.oof[val_idx] = np.expm1(utils.SAFE_PREDICT(model, X_val_fold))
            
            # [OOF_GAP_FIX] NaN-safe fold scoring
            val_preds = self.oof[val_idx]
            mask_val = ~np.isnan(val_preds)
            if mask_val.any():
                self.fold_scores.append(mean_absolute_error(y_val[mask_val], val_preds[mask_val]))
            else:
                self.fold_scores.append(0.0)
            
            # [TASK 7: LEAF VALUE MONITORING]
            target_booster = model.booster_ if not Config.USE_2STAGE_MODEL else model["tail"].booster_
            tree_str = str(target_booster.dump_model()['tree_info'])
            leaf_values = [float(x) for x in re.findall(r"'leaf_value':\s*([-+]?\d*\.\d+|\d+)", tree_str)]
            self.fold_stats.append({"fold": fold, "mae": float(self.fold_scores[-1])})
            
            logger.info(f"━━━━━━━━━━━━━━━━ FOLD {fold} MAE: {self.fold_scores[-1]:.4f} ━━━━━━━━━━━━━━━━")
            # Test inference
            X_te = test_df_full.values.astype(np.float32)
            if "final" not in self.test_preds:
                self.test_preds["final"] = np.zeros(len(self.df_test))
            
            if Config.USE_2STAGE_MODEL:
                X_te_clf = test_df_full[clf_features].values.astype(np.float32)
                X_te_tail = test_df_full[tail_features].values.astype(np.float32)
                
                p_te_raw = utils.SAFE_PREDICT_PROBA(model["clf"], X_te_clf)[:, 1]
                preds_t_te = np.expm1(utils.SAFE_PREDICT(model["tail"], X_te_tail))
                preds_nt_te = np.expm1(utils.SAFE_PREDICT(model["non_tail"], X_te))
                
                final_weight_te = _multi_signal_blend(p_te_raw, preds_t_te, preds_nt_te)
                gap_te = model["risk_ctrl"].compute_gap(preds_t_te, preds_nt_te)
                final_weight_te = model["risk_ctrl"].apply(final_weight_te, p_te_raw, gap_te)
                
                self.test_preds["final"] += (final_weight_te * preds_t_te + (1.0 - final_weight_te) * preds_nt_te) / Config.NFOLDS
            else:
                self.test_preds["final"] += np.expm1(utils.SAFE_PREDICT(model, X_te)) / Config.NFOLDS
            
            # Persistence
            fold_dir = f"{Config.MODELS_PATH}/reconstructors"
            os.makedirs(fold_dir, exist_ok=True)
            os.makedirs(f"{Config.MODELS_PATH}/lgbm", exist_ok=True)
            with open(f"{fold_dir}/recon_fold_{fold}.pkl", "wb") as f:
                pickle.dump(fold_data["reconstructor"], f)
            with open(f"{fold_dir}/scaler_fold_{fold}.pkl", "wb") as f:
                pickle.dump(scaler, f)
            with open(f"{fold_dir}/features_fold_{fold}.pkl", "wb") as f:
                pickle.dump(sorted_features, f)
            with open(f"{fold_dir}/norm_scaler_fold_{fold}.pkl", "wb") as f:
                pickle.dump(norm_scaler, f)
            with open(f"{fold_dir}/layout_stats_fold_{fold}.pkl", "wb") as f:
                pickle.dump(fold_data["layout_stats"], f)
            with open(f"{fold_dir}/bucket_edges_fold_{fold}.pkl", "wb") as f:
                pickle.dump(fold_data["bucket_edges"], f)
            with open(f"{fold_dir}/extreme_quantiles_fold_{fold}.pkl", "wb") as f:
                pickle.dump(fold_data["extreme_quantiles"], f)
            with open(f"{fold_dir}/train_col_means_fold_{fold}.pkl", "wb") as f:
                pickle.dump(fold_data["train_col_means"], f)
            with open(f"{fold_dir}/risk_ctrl_fold_{fold}.pkl", "wb") as f:
                pickle.dump(risk_ctrl, f)
            with open(f"{Config.MODELS_PATH}/lgbm/model_fold_{fold}.pkl", "wb") as f:
                pickle.dump(model, f)
            
            # [FORENSIC #13/#14] Persist per-fold train_col_means and extreme_quantiles
            # for Phase 7 inference to eliminate test distribution leakage
            fold_train_df = self.df_train.iloc[splits[fold][0]]
            fold_train_col_means = {
                col: float(fold_train_df[col].mean())
                for col in fold_train_df.columns
                if col not in Config.ID_COLS and not fold_train_df[col].isna().all()
            }
            with open(f"{fold_dir}/train_col_means_fold_{fold}.pkl", "wb") as f:
                pickle.dump(fold_train_col_means, f)
            
            # Compute and save extreme quantiles from fold training data
            fold_extreme_quantiles = {}
            key_extreme_cols = ['order_inflow_15m', 'robot_utilization', 'congestion_score', 'near_collision_15m']
            for c in key_extreme_cols:
                if c in fold_train_df.columns:
                    fold_extreme_quantiles[c] = float(fold_train_df[c].quantile(0.95))
            # Also save util_rel_p90 if available
            if 'robot_utilization_rel_to_mean_5' in fold_train_df.columns:
                vals = fold_train_df['robot_utilization_rel_to_mean_5'].dropna()
                if len(vals) > 0:
                    fold_extreme_quantiles['util_rel_p90'] = float(np.quantile(vals, 0.90))
            with open(f"{fold_dir}/extreme_quantiles_fold_{fold}.pkl", "wb") as f:
                pickle.dump(fold_extreme_quantiles, f)
            
            # [GLOBAL_FALLBACK_FIX] Save global training means for new-layout inference
            if self.manifest and hasattr(self.manifest, 'train_col_means'):
                means_to_save = self.manifest.train_col_means
            elif isinstance(self.manifest, dict) and 'train_col_means' in self.manifest:
                means_to_save = self.manifest['train_col_means']
            else:
                logger.warning("[TRAINER] No train_col_means found in manifest. Saving empty fallback.")
                means_to_save = {}
                
            with open(f"{Config.MODELS_PATH}/reconstructors/global_means.pkl", "wb") as f:
                pickle.dump(means_to_save, f)
            self.models.append(model)
            # Clean up heavy objects
            del tr_df_full, val_df_full, test_df_full, X_tr_fold, X_val_fold, X_te
            gc.collect()
        # Final metrics & guards
        # [OOF_GAP_FIX] NaN-safe overall MAE calculation
        valid_mask = ~np.isnan(self.oof)
        if valid_mask.any():
            overall_mae = mean_absolute_error(self.y[valid_mask], self.oof[valid_mask])
            logger.info(f"[TRAINER] Structural corrected CV complete. Overall MAE: {overall_mae:.4f}")
        else:
            overall_mae = 0.0
            logger.warning("[TRAINER] No valid OOF predictions found for overall MAE.")
            
        self.metadata["num_features"] = len(global_features)
        self.metadata["oof_valid_rate"] = np.mean(~np.isnan(self.oof))
        
        # Variance‑collapse guard (NaN-safe)
        y_valid = self.y[valid_mask] if valid_mask.any() else np.array([0])
        oof_valid = self.oof[valid_mask] if valid_mask.any() else np.array([0])
        self.validate_distribution(oof_valid, y_valid)
        
        # Auditing
        dist_stats = DistributionAuditor.audit(y_valid, oof_valid, fold_name="FINAL_OOF")
        
        # [CV_RELIABILITY HOOK]
        # [WHY] After every training run, generate a CV reliability report to quantify
        #   whether the CV MAE can be trusted as a proxy for LB MAE.
        # [FAILURE_ADDRESSED] Without this report, the pipeline silently uses CV MAE
        #   as the optimization target despite it being provably unreliable (AUC 0.80-0.95).
        # [WHAT IT DOES] Computes reliability score based on adv_auc, std_ratio, mean_ratio.
        #   Score = 1.0 → CV is perfect LB proxy. Score → 0.0 → CV is useless.
        # [NOTE] The adv_auc here comes from perform_adversarial_audit() called in main.py.
        #   We use a placeholder (1.0 = worst case) if not yet computed, to ensure the
        #   report is always generated even in intermediate states.
        try:
            # Build distribution metrics from auditor output
            y_mean = float(np.nanmean(self.y[valid_mask])) if valid_mask.any() else 1.0
            pred_mean = float(np.nanmean(self.oof[valid_mask])) if valid_mask.any() else 0.0
            mean_ratio_est = pred_mean / (y_mean + 1e-9)
            std_ratio_est = float(dist_stats.get('std_ratio', 0.5))
            p99_ratio_est = float(dist_stats.get('p99_ratio', 0.5))
            
            # [FORENSIC #19] Compute ACTUAL adv_auc instead of hardcoded placeholder
            actual_adv_auc = self.perform_adversarial_audit()
            self.metadata["adv_auc"] = actual_adv_auc
            
            final_metrics = build_metrics(self.y, self.oof, y_base=self.oof_base)
            
            cv_report_path = f"{Config.LOG_DIR}/cv_reliability_report.txt"
            cv_result = generate_cv_reliability_report(
                fold_maes=self.fold_scores,
                oof_preds=self.oof,
                y_train=self.y,
                adv_auc=actual_adv_auc,
                mean_ratio=mean_ratio_est,
                std_ratio=std_ratio_est,
                p99_ratio=p99_ratio_est,
                output_path=cv_report_path,
            )
            self.metadata["cv_reliability_score"] = cv_result.get("reliability_score", 0.0)
            logger.info(f"[CV_RELIABILITY] Score: {cv_result.get('reliability_score', 0.0):.4f} | {cv_result.get('interpretation', 'N/A')}")
        except Exception as cv_err:
            # [NON-BLOCKING] CV reliability report failure must not block training output
            logger.warning(f"[CV_RELIABILITY] Report generation failed (non-critical): {cv_err}")
        
        # [PHASE_ARTIFACT_SYNC] Persist OOF after leakage-free completion
        save_npy(self.oof, f"{Config.PREDICTIONS_PATH}/oof_final.npy")
        
        return overall_mae, self.oof

    def validate_distribution(self, test_preds, train_stats):
        """Final sanity check to ensure test predictions don't collapse relative to train stats."""
        # [FORENSIC #22/#23] Removed unused DomainShiftAudit and fake analyze_model_divergence
        ratio, p_std, t_std, m_ratio = calculate_std_ratio(test_preds, train_stats)
        logger.info(f"[DIST_GUARD] Std Ratio: {ratio:.4f} (Pred: {p_std:.4f}, Train: {t_std:.4f})")
        logger.info(f"[DIST_GUARD] Mean Ratio: {m_ratio:.4f}")
        if ratio < 0.1 or ratio > 2.0:
            logger.error(f"!!! [CRITICAL_DIST_VIOLATION] Std Ratio {ratio:.4f} outside safe bounds [0.1, 2.0] !!!")

    def compute_adversarial_weights(self, features):
        """[PHASE 3: ADVERSARIAL WEIGHTING]"""
        logger.info(f"[ADV_WEIGHT] Computing weights using {len(features)} features...")
        raw_feat = [f for f in features if f in self.df_train.columns and f in self.df_test.columns]
        raw_feat = [f for f in raw_feat if pd.api.types.is_numeric_dtype(self.df_train[f])]
        X_tr = self.df_train[raw_feat].copy().fillna(-999).values.astype(np.float32)
        X_te = self.df_test[raw_feat].copy().fillna(-999).values.astype(np.float32)
        X = np.vstack([X_tr, X_te])
        y = np.hstack([np.zeros(len(X_tr)), np.ones(len(X_te))])
        adv_clf = LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, verbose=-1)
        utils.SAFE_FIT(adv_clf, X.astype(np.float32), y.astype(np.float32))
        probs = utils.SAFE_PREDICT_PROBA(adv_clf, X_tr.astype(np.float32))[:, 1]
        weights = probs / (1.0 - probs)
        if Config.ADV_WEIGHT_POWER != 1.0:
            weights = weights ** Config.ADV_WEIGHT_POWER
        weights = weights / np.mean(weights)
        logger.info(f"[ADV_WEIGHT] Weights computed (Power={Config.ADV_WEIGHT_POWER}). Range: [{weights.min():.4f}, {weights.max():.4f}]")
        return weights

    def perform_adversarial_audit(self):
        """[PHASE 3: POST‑TRAIN ADVERSARIAL AUDIT]"""
        logger.info("[AUDIT] Starting final adversarial audit...")
        s_tr = self.df_train.sample(min(20000, len(self.df_train)))
        s_te = self.df_test.sample(min(20000, len(self.df_test)))
        common_cols = [c for c in s_tr.columns if c in s_te.columns and c not in Config.ID_COLS]
        common_cols = [c for c in common_cols if pd.api.types.is_numeric_dtype(s_tr[c])]
        X_tr = s_tr[common_cols].fillna(-999).values
        X_te = s_te[common_cols].fillna(-999).values
        X = np.vstack([X_tr, X_te])
        y = np.hstack([np.zeros(len(X_tr)), np.ones(len(X_te))])
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        aucs = []
        for tr_idx, val_idx in skf.split(X, y):
            adv_clf = LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, verbose=-1)
            utils.SAFE_FIT(adv_clf, X[tr_idx].astype(np.float32), y[tr_idx].astype(np.float32))
            probs = utils.SAFE_PREDICT_PROBA(adv_clf, X[val_idx].astype(np.float32))[:, 1]
            aucs.append(roc_auc_score(y[val_idx], probs))
        avg_auc = np.mean(aucs)
        logger.info(f"[AUDIT] Final Adversarial AUC: {avg_auc:.4f}")
        return avg_auc
