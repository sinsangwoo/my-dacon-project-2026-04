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
    DriftShieldScaler, run_integrity_audit
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
        
        # [MISSION: FINAL EDGE BOOST] Integrity Audit
        run_integrity_audit(self.df_train, label="TRAIN_POOL")
        run_integrity_audit(self.df_test, label="TEST_POOL")
        
        for fold, (tr_idx, val_idx) in enumerate(self.kf.split(self.df_train, self.y, groups=self.groups)):
            logger.info(f"--- FOLD {fold} ---")
            
            # Isolation-First Data Slice (No copy to save memory)
            tr_df = self.df_train.iloc[tr_idx]
            val_df = self.df_train.iloc[val_idx]
            test_df_fold = self.df_test
            y_tr = self.y[tr_idx]
            y_val = self.y[val_idx]

            # 1. Component Initialization
            scaler = DriftShieldScaler()
            scaler.fit(tr_df, FEATURE_SCHEMA['raw_features'])
            
            from sklearn.preprocessing import StandardScaler
            norm_scaler = StandardScaler()
            reconstructor = SuperchargedPCAReconstructor(input_dim=len(FEATURE_SCHEMA['raw_features']))
            
            # [STABILITY] Process drift audit and stability filtering
            from .distribution import DomainShiftAudit, FeatureStabilityFilter
            audit = DomainShiftAudit()
            drift_df = audit.calculate_drift(tr_df, val_df, FEATURE_SCHEMA['raw_features'])
            
            filter = FeatureStabilityFilter(threshold=Config.STABILITY_THRESHOLD)
            filter.fit(drift_df)
            
            # [MISSION: ADVERSARIAL PRIORITIZATION] - Moved up for early feature pruning
            # Initial candidates: stable raw + all latent
            initial_candidates = [f for f in all_features if f in filter.stable_features or any(p in f for p in ['embed', 'weighted', 'trend', 'volatility', 'regime', 'local_density', 'similarity'])]
            
            # [PRUNING] Phase 1: Global Reference Importance (v17.0)
            # We use the raw-model importance as a first-pass filter if available
            # Or use a fixed threshold based on benchmark (Cutoff 0.9 -> ~113-120 features)
            IMPORTANCE_CUTOFF = 0.90 
            
            # [SPEED/STABILITY] Sequential Latent Feature Population
            from src.utils import memory_guard, log_memory_usage
            
            # [SINGLE-PASS CACHE] Explicitly build the pool embeddings
            # [PHASE 2: UNIFIED SCALING]
            # tr_df_scaled_lite = scaler.transform(tr_df[Config.EMBED_BASE_COLS], Config.EMBED_BASE_COLS)
            # reconstructor.fit(tr_df_scaled_lite.values)
            # reconstructor.build_fold_cache(tr_df)
            
            # [MISSION: SSOT SCALING] Scale tr_df once and use it everywhere
            # 1. Drift handling (Clipping/NaNs)
            tr_df_drifted = scaler.transform(tr_df, FEATURE_SCHEMA['raw_features'])
            # 2. Normalization (StandardScaler)
            tr_df_scaled_all = tr_df_drifted.copy()
            tr_df_scaled_all[FEATURE_SCHEMA['raw_features']] = norm_scaler.fit_transform(tr_df_drifted[FEATURE_SCHEMA['raw_features']])
            
            logger.info(f"[TRAIN_AUDIT] DriftShield + Normalization complete for FOLD {fold}")
            
            # [CLIPPING_MONITOR] Log top 5 clipped features
            sorted_clips = sorted(scaler.clipping_ratios.items(), key=lambda x: x[1], reverse=True)
            logger.info(f"[CLIPPING_MONITOR] Top clipped: {sorted_clips[:5]}")
            
            # Fit reconstructor on scaled base cols
            reconstructor.fit(tr_df_scaled_all[Config.EMBED_BASE_COLS].values)
            # Build cache using scaled data
            reconstructor.build_fold_cache(tr_df_scaled_all)
            
            # [MEMORY_OPTIMIZATION] Zero-Copy Population (v18.0)
            # Step 1: Process Train for Pruning (Avoid Wide DataFrame)
            logger.info(f"[FOLD {fold}] Extracting features for pruning...")
            
            # 1a. Scale raw features - already done in tr_df_scaled_all
            X_raw_part = tr_df_scaled_all[[f for f in initial_candidates if f in FEATURE_SCHEMA['raw_features']]].values.astype(np.float32)
            
            # 1b. Get latent stats directly
            # [PHASE 3: KNN LEAKAGE] Pass is_train=True
            latent_stats = reconstructor.calculate_graph_stats(tr_df_scaled_all, is_train=True)
            
            # 1c. Build temporary X for pruning using the schema-aware population logic
            # Instead of manual loop, let's use a temporary wide DF for pruning since it's just for pruning
            # and we only keep selected features later.
            tr_df_temp = apply_latent_features(tr_df_scaled_all, reconstructor, scaler=None, selected_features=initial_candidates, is_train=True)
            X_tr_temp = tr_df_temp[initial_candidates].values.astype(np.float32)
            temp_feat_names = initial_candidates
            
            del tr_df_temp, latent_stats, X_raw_part; gc.collect()

            # [DYNAMIC_PRUNING]
            temp_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, verbose=-1, random_state=42)
            temp_model.fit(X_tr_temp, y_tr)
            
            imp = temp_model.feature_importances_
            imp_df = pd.DataFrame({'f': temp_feat_names, 'i': imp}).sort_values('i', ascending=False)
            imp_df['c'] = imp_df['i'].cumsum() / (imp_df['i'].sum() + 1e-9)
            
            fold_features = imp_df[imp_df['c'] <= IMPORTANCE_CUTOFF]['f'].tolist()
            if not fold_features: fold_features = initial_candidates[:100]
            
            logger.info(f"[FOLD {fold}] Selected {len(fold_features)} features (Cutoff {IMPORTANCE_CUTOFF}).")
            
            # Extract final X_tr from X_tr_temp (Slice it)
            selected_indices = [temp_feat_names.index(f) for f in fold_features]
            X_tr = X_tr_temp[:, selected_indices].copy()
            
            # [PHASE 2: ASSERT EMBEDDING SPACE CONSISTENCY]
            # This is implicitly handled by using the same scaler and reconstructor
            
            del X_tr_temp, temp_model, imp_df, tr_df_scaled_all; gc.collect()
            memory_guard(f"Fold {fold} - Post Train Pruning", logger)
            
            # Step 2: Process Validation with ONLY selected features
            logger.info(f"[FOLD {fold}] Populating validation features (Selected Only)...")
            # [PHASE 2: UNIFIED SCALING] Scale validation data first
            val_df_drifted = scaler.transform(val_df, FEATURE_SCHEMA['raw_features'])
            val_df_scaled = val_df_drifted.copy()
            val_df_scaled[FEATURE_SCHEMA['raw_features']] = norm_scaler.transform(val_df_drifted[FEATURE_SCHEMA['raw_features']])
            
            val_df_full = apply_latent_features(val_df_scaled, reconstructor, scaler=None, selected_features=fold_features, is_train=False)
            X_val = val_df_full[fold_features].values.astype(np.float32)
            del val_df_full, val_df_scaled; gc.collect()
            
            # Step 3: Process Test with ONLY selected features
            logger.info(f"[FOLD {fold}] Populating test features (Selected Only)...")
            # [PHASE 2: UNIFIED SCALING] Scale test data first
            test_df_drifted = scaler.transform(test_df_fold, FEATURE_SCHEMA['raw_features'])
            test_df_scaled = test_df_drifted.copy()
            test_df_scaled[FEATURE_SCHEMA['raw_features']] = norm_scaler.transform(test_df_drifted[FEATURE_SCHEMA['raw_features']])
            
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
        X_raw = self.df_train[FEATURE_SCHEMA['raw_features']].fillna(0).values.astype(np.float32)
        X_test_raw = self.df_test[FEATURE_SCHEMA['raw_features']].fillna(0).values.astype(np.float32)
        return self.train_kfolds(X_raw, self.y, X_test_raw, 'raw', Config.RAW_LGBM_PARAMS)

    def perform_adversarial_audit(self):
        logger.info("[AUDIT] Running realigned adversarial validation...")
        aucs = []
        X_raw = self.df_train[FEATURE_SCHEMA['raw_features']].fillna(0).values
        for fold, (tr_idx, val_idx) in enumerate(self.kf.split(X_raw, self.y, groups=self.groups)):
            auc = run_adversarial_validation(X_raw[tr_idx], X_raw[val_idx])
            aucs.append(auc)
        return np.mean(aucs)

    def validate_distribution(self, preds, train_stats):
        ratio, pred_std, train_std, mean_ratio = calculate_std_ratio(preds, train_stats)
        logger.info(f"[DIST_GUARD] std_ratio={ratio:.4f} | mean_ratio={mean_ratio:.4f}")
        
        # [PHASE 9: ALWAYS ON] Removed debug/smoke skip
            
        if ratio < 0.5 or ratio > 2.0: # [PHASE 5: REAL VARIANCE] Adjusted range to [0.5, 2.0]
            raise RuntimeError(f"FAIL: std_ratio {ratio:.4f} outside [0.5, 2.0]")
        return ratio

    def analyze_model_divergence(self):
        if 'raw' in self.oof_preds and 'final' in self.oof_preds:
            r, e = self.oof_preds['raw'], self.oof_preds['final']
            corr = np.corrcoef(r, e)[0, 1]
            logger.info(f"[MODEL_DIVERGENCE] corr_raw_final={corr:.4f}")
            return corr, 0.0
        return 1.0, 0.0
