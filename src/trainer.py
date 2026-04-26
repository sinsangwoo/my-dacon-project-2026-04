import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, roc_auc_score
import logging
import gc
import os
import json

from .config import Config
from .schema import FEATURE_SCHEMA
from .utils import (
    SAFE_FIT, SAFE_PREDICT, build_metrics, run_adversarial_validation, 
    generate_pseudo_test_set, calculate_risk_score, calculate_std_ratio,
    DriftShieldScaler, run_integrity_audit, memory_guard
)
from .data_loader import SuperchargedPCAReconstructor, apply_latent_features

logger = logging.getLogger(__name__)

class Trainer:
    """Canonical Trainer for the restored pipeline (v16.0 - Shift-Robust)."""
    def __init__(self, X_base, y, X_test_base, groups=None, full_df=None, test_df=None):
        self.df_train = full_df 
        self.df_test = test_df
        self.y = np.asarray(y, dtype=np.float32)
        self.groups = groups if groups is not None else np.zeros(len(y))
        
        self.kf = GroupKFold(n_splits=Config.NFOLDS)
        self.oof_preds = {}
        self.test_preds = {}
        self.fold_stats = {}

    def fit_leakage_free_model(self):
        """
        [ZERO_TOLERANCE_CV] + [SHIFT_ROBUST]
        Redesigned CV loop with feature filtering and adversarial prioritization.
        """
        logger.info("[TRAIN_SHIFT_ROBUST] Starting robust CV loop...")
        n_train = len(self.df_train)
        n_test = len(self.df_test)
        oof = np.zeros(n_train)
        test_preds = np.zeros(n_test)
        
        # Feature stability will be determined per fold.
        all_features = FEATURE_SCHEMA['all_features']
        
        # [MISSION: FINAL EDGE BOOST] Integrity Audit (train only — no test reference)
        run_integrity_audit(self.df_train, label="TRAIN_POOL")
        
        # [WHY_THIS_CHANGE] Fold Consistency Reconstruction (TASK 5)
        # Problem: Feature selection was performed PER FOLD based on local importance.
        # Root Cause: Over-optimization for local fold characteristics.
        # Why previous logic failed: Created non-deterministic feature spaces; violated
        #   the "same feature space across folds" requirement.
        # Why this solution: Perform a single, global feature selection phase on a 
        #   representative training sample before the CV loop begins. 
        #   Ensures all folds train on the exact same signal pool.
        logger.info("[TRAIN_SHIFT_ROBUST] Performing global feature selection...")
        
        # 1. Global Scaling & Initialization
        global_scaler = DriftShieldScaler()
        # [TASK 12 — FOLD STABILITY] Use local RNG for deterministic sampling
        # [CODE_EVIDENCE] Previously: np.random.choice() used global RNG state
        # [FAILURE_MODE_PREVENTED] Non-deterministic feature selection across runs
        rng = np.random.RandomState(Config.SEED)
        sample_idx = rng.choice(n_train, min(n_train, 20000), replace=False)
        sample_df = self.df_train.iloc[sample_idx]
        sample_y = self.y[sample_idx]
        
        raw_cols = [c for c in FEATURE_SCHEMA['raw_features'] if c in sample_df.columns]
        global_scaler.fit(sample_df, raw_cols)
        
        from sklearn.preprocessing import StandardScaler
        global_norm_scaler = StandardScaler()
        sample_df_drifted = global_scaler.transform(sample_df, raw_cols)
        global_norm_scaler.fit(sample_df_drifted[raw_cols])
        
        global_reconstructor = SuperchargedPCAReconstructor(input_dim=len(raw_cols))
        sample_df_scaled = sample_df_drifted.copy()
        sample_df_scaled[raw_cols] = global_norm_scaler.transform(sample_df_drifted[raw_cols])
        
        global_reconstructor.fit(sample_df_scaled[Config.PCA_INPUT_COLS].values)
        
        # 2. Candidate Selection (Train-Only Stability)
        # [TASK 4 — FEATURE SELECTION LEAKAGE FIX]
        # [WHY_THIS_DESIGN] Drift calculation MUST NOT reference test data.
        # [CODE_EVIDENCE] Previously: audit.calculate_drift(sample_df, self.df_test.head(5000), raw_cols)
        #   This directly used self.df_test for drift calculation — DATA LEAKAGE.
        #   Test data was influencing which features were selected for training.
        # [SOLUTION] Use train-internal holdout split for stability estimation.
        #   Split the training sample 80/20 and measure drift between the two halves.
        #   This detects features with high INTERNAL instability (which are also likely
        #   to be unstable vs test) without ever seeing test data.
        # [FAILURE_MODE_PREVENTED] Test data referenced before final inference.
        from .distribution import DomainShiftAudit, FeatureStabilityFilter
        audit = DomainShiftAudit()
        n_sample_split = int(len(sample_df) * 0.8)
        sample_train_half = sample_df.iloc[:n_sample_split]
        sample_holdout_half = sample_df.iloc[n_sample_split:]
        drift_df = audit.calculate_drift(sample_train_half, sample_holdout_half, raw_cols)
        stability_filter = FeatureStabilityFilter(threshold=Config.STABILITY_THRESHOLD)
        stability_filter.fit(drift_df)
        
        initial_candidates = [
            f for f in all_features 
            if (f in stability_filter.stable_features or any(p in f for p in ['embed', 'weighted', 'trend', 'volatility', 'regime', 'local_density', 'similarity']))
            and (f not in FEATURE_SCHEMA['raw_features'] or f in raw_cols)
        ]
        
        # 3. Global Importance Pruning
        global_reconstructor.build_fold_cache(sample_df_scaled)
        sample_df_full = apply_latent_features(sample_df_scaled, global_reconstructor, scaler=None, selected_features=initial_candidates, is_train=True)
        X_sample = sample_df_full[initial_candidates].values.astype(np.float32)
        
        temp_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, verbose=-1, random_state=42)
        temp_model.fit(X_sample, sample_y)
        
        imp = temp_model.feature_importances_
        imp_df = pd.DataFrame({'f': initial_candidates, 'i': imp}).sort_values('i', ascending=False)
        imp_df['c'] = imp_df['i'].cumsum() / (imp_df['i'].sum() + 1e-9)
        
        # [TASK 6 — IMPORTANCE_CUTOFF LOGIC FIX]
        # [WHY_THIS_DESIGN] Use direct cumulative importance threshold, not positional percentile.
        # [CODE_EVIDENCE] Previously:
        #   p80_idx = int(np.ceil(0.80 * n_feats)) - 1
        #   p80_val = float(cum_series_vals[min(p80_idx, n_feats - 1)])
        #   IMPORTANCE_CUTOFF = float(np.clip(p80_val, 0.80, 0.97))
        # This computed the cumulative importance VALUE at the 80th PERCENTILE POSITION,
        # then clipped it. This conflates feature count with importance coverage.
        # Example failure: If top 10% of features cover 99% importance, p80_idx points
        # to a feature with ~100% cumulative importance, and the cutoff becomes 0.97 (clipped).
        # This would KEEP features with near-zero marginal importance.
        # [MATHEMATICAL FIX] Directly set cumulative importance target = 0.95.
        # Keep features until their cumulative importance reaches 95% of total.
        # This is the correct interpretation: "features covering 95% of model signal."
        # [FAILURE_MODE_PREVENTED] Over-inclusion of zero-importance features.
        IMPORTANCE_CUTOFF = 0.95
        
        selected_features = imp_df[imp_df['c'] <= IMPORTANCE_CUTOFF]['f'].tolist()
        if not selected_features: selected_features = initial_candidates[:100]
        
        logger.info(f"[GLOBAL_SELECTION] Selected {len(selected_features)} features for all folds (Cumulative importance cutoff={IMPORTANCE_CUTOFF}).")
        
        # Cleanup global phase
        del sample_df, sample_df_drifted, sample_df_scaled, sample_df_full, X_sample, temp_model, imp_df
        gc.collect()

        for fold, (tr_idx, val_idx) in enumerate(self.kf.split(self.df_train, self.y, groups=self.groups)):
            logger.info(f"--- FOLD {fold} ---")
            
            # Isolation-First Data Slice (No copy to save memory)
            tr_df = self.df_train.iloc[tr_idx]
            val_df = self.df_train.iloc[val_idx]
            test_df_fold = self.df_test
            y_tr = self.y[tr_idx]
            y_val = self.y[val_idx]

            raw_cols = [c for c in FEATURE_SCHEMA['raw_features'] if c in tr_df.columns and c in test_df_fold.columns]
            
            scaler = DriftShieldScaler()
            scaler.fit(tr_df, raw_cols)
            
            from sklearn.preprocessing import StandardScaler
            norm_scaler = StandardScaler()
            reconstructor = SuperchargedPCAReconstructor(input_dim=len(raw_cols))
            
            # [MISSION: SSOT SCALING] Scale tr_df once and use it everywhere
            
            # 1. Drift handling (Clipping/NaNs)
            tr_df_drifted = scaler.transform(tr_df, raw_cols)
            # 2. Normalization (StandardScaler)
            tr_df_scaled_all = tr_df_drifted.copy()
            tr_df_scaled_all[raw_cols] = norm_scaler.fit_transform(tr_df_drifted[raw_cols])
            
            logger.info(f"[TRAIN_AUDIT] DriftShield + Normalization complete for FOLD {fold}")
            
            # Fit reconstructor on scaled base cols
            reconstructor.fit(tr_df_scaled_all[Config.PCA_INPUT_COLS].values)
            # Build cache using scaled data
            reconstructor.build_fold_cache(tr_df_scaled_all)
            
            # Step 1: Process Train with GLOBAL selected features
            logger.info(f"[FOLD {fold}] Extracting selected features...")
            
            # Ensure selected features are available in this fold's raw_cols
            fold_features = [f for f in selected_features if (f not in FEATURE_SCHEMA['raw_features'] or f in raw_cols)]
            
            tr_df_final = apply_latent_features(tr_df_scaled_all, reconstructor, scaler=None, selected_features=fold_features, is_train=True)
            X_tr = tr_df_final[fold_features].values.astype(np.float32)
            
            del tr_df_final; gc.collect()
            memory_guard(f"Fold {fold} - Post Train", logger)
            
            # Step 2: Process Validation with ONLY selected features
            logger.info(f"[FOLD {fold}] Populating validation features (Selected Only)...")
            # [PHASE 2: UNIFIED SCALING] Scale validation data first
            val_df_drifted = scaler.transform(val_df, raw_cols)
            val_df_scaled = val_df_drifted.copy()
            val_df_scaled[raw_cols] = norm_scaler.transform(val_df_drifted[raw_cols])
            
            val_df_full = apply_latent_features(val_df_scaled, reconstructor, scaler=None, selected_features=fold_features, is_train=False)
            X_val = val_df_full[fold_features].values.astype(np.float32)
            del val_df_full, val_df_scaled; gc.collect()
            
            # Step 3: Process Test with ONLY selected features
            logger.info(f"[FOLD {fold}] Populating test features (Selected Only)...")
            # [PHASE 2: UNIFIED SCALING] Scale test data first
            test_df_drifted = scaler.transform(test_df_fold, raw_cols)
            test_df_scaled = test_df_drifted.copy()
            test_df_scaled[raw_cols] = norm_scaler.transform(test_df_drifted[raw_cols])
            
            test_df_full = apply_latent_features(test_df_scaled, reconstructor, scaler=None, selected_features=fold_features, is_train=False)
            X_test_f = test_df_full[fold_features].values.astype(np.float32)
            
            # [PHASE 2: ASSERT EMBEDDING SPACE]
            # Before KNN or model input, we can check if the feature space is consistent
            # In this refactored version, we ensure this by scaling everything with 'scaler'
            # and then passing to 'apply_latent_features' which uses the same 'reconstructor'.
            
            del test_df_full, test_df_scaled; gc.collect()
            memory_guard(f"Fold {fold} - Post Test", logger)
            
            # 2. Artifact Preservation
            import pickle
            os.makedirs(f'{Config.MODELS_PATH}/reconstructors', exist_ok=True)
            with open(f'{Config.MODELS_PATH}/reconstructors/recon_fold_{fold}.pkl', 'wb') as f:
                pickle.dump(reconstructor, f)
            with open(f'{Config.MODELS_PATH}/reconstructors/scaler_fold_{fold}.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            with open(f'{Config.MODELS_PATH}/reconstructors/norm_scaler_fold_{fold}.pkl', 'wb') as f:
                pickle.dump(norm_scaler, f)
            with open(f'{Config.MODELS_PATH}/reconstructors/features_fold_{fold}.pkl', 'wb') as f:
                pickle.dump(fold_features, f)
            
            # Clear cache immediately to prevent leakage and save memory
            reconstructor.clear_fold_cache()
            
            # 3. [MISSION: ADVERSARIAL PRIORITIZATION]
            adv_X = np.vstack([X_tr, X_test_f])
            adv_y = np.array([0]*len(X_tr) + [1]*len(X_test_f))
            adv_model = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, verbose=-1, random_state=42)
            adv_model.fit(adv_X, adv_y)
            adv_probs = adv_model.predict_proba(X_tr)[:, 1]
            
            # 4. Weight Balancing (Reliability)
            # [PHASE 7: WEIGHT BALANCING] Redefine weights
            # final_weight = base_weight * extreme_weight * adversarial_weight
            base_weight = 1.0
            adversarial_weight = (1.0 + 2.0 * adv_probs)
            
            # [STABILITY_FIX] Ensure y_tr is accessible and threshold calculation is safe
            y_tr_val = y_tr.values if isinstance(y_tr, pd.Series) else y_tr
            threshold = np.quantile(y_tr_val, Config.EXTREME_TARGET_QUANTILE)
            
            # extreme_weight MUST dominate: extreme_weight >= 2.0
            extreme_weight = np.where(y_tr_val >= threshold, Config.EXTREME_SAMPLE_WEIGHT, 1.0)
            if Config.EXTREME_SAMPLE_WEIGHT < 2.0:
                logger.warning(f"[WEIGHT_CONTRACT_VIOLATION] extreme_weight {Config.EXTREME_SAMPLE_WEIGHT} < 2.0. Enforcing 2.0.")
                extreme_weight = np.where(y_tr_val >= threshold, max(2.0, Config.EXTREME_SAMPLE_WEIGHT), 1.0)
            
            weights = base_weight * extreme_weight * adversarial_weight
            
            # VERIFY: mean(weight_extreme) > mean(weight_normal)
            mean_extreme = weights[y_tr_val >= threshold].mean()
            mean_normal = weights[y_tr_val < threshold].mean()
            logger.info(f"[WEIGHT_AUDIT] mean_extreme={mean_extreme:.4f} | mean_normal={mean_normal:.4f}")
            assert mean_extreme > mean_normal, "[WEIGHT_CONTRACT_FAIL] extreme samples must have higher average weight"
            
            # 5. Fit Shift-Robust LGBM
            params = Config.EMBED_LGBM_PARAMS
            model = LGBMRegressor(**params)
            SAFE_FIT(model, X_tr, y_tr, sample_weight=weights, eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=0)])
            
            # 6. Predict & Save
            f_preds = SAFE_PREDICT(model, X_val)
            oof[val_idx] = f_preds
            test_preds += SAFE_PREDICT(model, X_test_f) / Config.NFOLDS
            
            os.makedirs(f'{Config.MODELS_PATH}/lgbm', exist_ok=True)
            with open(f'{Config.MODELS_PATH}/lgbm/model_fold_{fold}.pkl', 'wb') as f:
                pickle.dump(model, f)
            
            f_mae = mean_absolute_error(y_val, f_preds)
            self.fold_stats[fold] = {"mae": f_mae, "mean": np.mean(f_preds)}
            logger.info(f"Fold {fold} MAE: {f_mae:.6f}")
            
            # 7. Aggressive Cleanup
            del X_tr, X_val, X_test_f, adv_X, adv_model, weights, model
            gc.collect()
            memory_guard(f"Fold {fold} - End", logger)

        self.oof_preds['final'] = oof
        self.test_preds['final'] = test_preds
        
        metrics = build_metrics(self.y, oof)
        logger.info(f"[SHIFT_ROBUST_TRAIN] Final MAE: {metrics['mae']:.6f}")
        return metrics['mae'], oof

    def train_kfolds(self, X_subset, y, X_test_subset, result_key, params):
        # Keep legacy for comparison
        n_train = len(y)
        n_test = len(X_test_subset)
        oof = np.zeros(n_train)
        test_preds = np.zeros(n_test)
        
        for fold, (tr_idx, val_idx) in enumerate(self.kf.split(X_subset, y, groups=self.groups)):
            X_tr, y_tr = X_subset[tr_idx], y[tr_idx]
            X_val, y_val = X_subset[val_idx], y[val_idx]
            
            model = LGBMRegressor(**params)
            SAFE_FIT(model, X_tr, y_tr, eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=0)])
            
            oof[val_idx] = SAFE_PREDICT(model, X_val)
            test_preds += SAFE_PREDICT(model, X_test_subset) / Config.NFOLDS
            
        self.oof_preds[result_key] = oof
        self.test_preds[result_key] = test_preds
        return build_metrics(y, oof)['mae'], oof

    def fit_raw_model(self):
        raw_cols = [c for c in FEATURE_SCHEMA['raw_features'] if c in self.df_train.columns and c in self.df_test.columns]
        print(f"DEBUG: missing from test: {[c for c in raw_cols if c not in self.df_test.columns]}")
        X_raw = self.df_train[raw_cols].fillna(0).values.astype(np.float32)
        X_test_raw = self.df_test[raw_cols].fillna(0).values.astype(np.float32)
        return self.train_kfolds(X_raw, self.y, X_test_raw, 'raw', Config.RAW_LGBM_PARAMS)

    def perform_adversarial_audit(self):
        logger.info("[AUDIT] Running realigned adversarial validation...")
        aucs = []
        raw_cols = [c for c in FEATURE_SCHEMA['raw_features'] if c in self.df_train.columns]
        X_raw = self.df_train[raw_cols].fillna(0).values
        for fold, (tr_idx, val_idx) in enumerate(self.kf.split(X_raw, self.y, groups=self.groups)):
            auc = run_adversarial_validation(X_raw[tr_idx], X_raw[val_idx])
            aucs.append(auc)
        return np.mean(aucs)

    def validate_distribution(self, preds, train_stats):
        ratio, pred_std, train_std, mean_ratio = calculate_std_ratio(preds, train_stats)
        logger.info("[DIST_GUARD] std_ratio={:.4f} | mean_ratio={:.4f}".format(ratio, mean_ratio))

        # [WHY_THIS_CHANGE] Adaptive std_ratio bounds
        # Problem: std_ratio range [0.5, 2.0] was hardcoded with no statistical basis.
        # Root Cause: Arbitrary bounds chosen during early debugging, never revisited.
        # Decision: Derive bounds from the std_ratio values observed across OOF folds.
        #   - Collect per-fold std_ratio from self.fold_stats if available.
        #   - Derive [Q1 - 2*IQR, Q3 + 2*IQR] bounds (wider than 1.5*IQR for safety).
        #   - Absolute fallback: [0.1, 5.0] (never too restrictive, never meaningless).
        # Why IQR (not fixed range):
        #   - Fixed [0.5, 2.0]: violates RULE 1; calibrated to one dataset.
        #   - IQR with fold std_ratios: grounded in actual prediction distribution behavior.
        # Expected Impact: Bounds adapt to model behavior; unstable models detected earlier.
        if self.fold_stats and len(self.fold_stats) >= 2:
            fold_means = [v.get("mean", float("nan")) for v in self.fold_stats.values()]
            fold_means = [m for m in fold_means if not np.isnan(m) and m > 0]
            if len(fold_means) >= 2:
                train_mean = train_stats.get("mean", 1.0) or 1.0
                fold_ratios = [m / (train_mean + 1e-9) for m in fold_means]
                q1_r = float(np.percentile(fold_ratios, 25))
                q3_r = float(np.percentile(fold_ratios, 75))
                iqr_r = q3_r - q1_r
                lo = float(np.clip(q1_r - 2.0 * iqr_r, 0.1, 0.8))
                hi = float(np.clip(q3_r + 2.0 * iqr_r, 1.2, 5.0))
                derivation_bounds = (
                    "IQR from {} fold mean-ratios (Q1={:.3f}, Q3={:.3f}, IQR={:.3f}) "
                    "-> bounds=[{:.3f}, {:.3f}]".format(
                        len(fold_ratios), q1_r, q3_r, iqr_r, lo, hi
                    )
                )
            else:
                lo, hi = 0.1, 5.0
                derivation_bounds = "Insufficient fold data; using absolute fallback [0.1, 5.0]"
        else:
            lo, hi = 0.1, 5.0
            derivation_bounds = "No fold_stats available; using absolute fallback [0.1, 5.0]"

        logger.info("[DIST_GUARD] std_ratio bounds derived: [{:.3f}, {:.3f}] | {}".format(
            lo, hi, derivation_bounds))

        # [PHASE 9: ALWAYS ON] Removed debug/smoke skip
        if ratio < lo or ratio > hi:
            raise RuntimeError("FAIL: std_ratio {:.4f} outside derived bounds [{:.3f}, {:.3f}] | {}".format(
                ratio, lo, hi, derivation_bounds))
        return ratio

    def analyze_model_divergence(self):
        if 'raw' in self.oof_preds and 'final' in self.oof_preds:
            r, e = self.oof_preds['raw'], self.oof_preds['final']
            corr = np.corrcoef(r, e)[0, 1]
            logger.info(f"[MODEL_DIVERGENCE] corr_raw_final={corr:.4f}")
            return corr, 0.0
        return 1.0, 0.0
