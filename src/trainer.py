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
    DriftShieldScaler
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
            # Fit reconstructor on scaled train data
            tr_df_scaled_lite = scaler.transform(tr_df[Config.EMBED_BASE_COLS], Config.EMBED_BASE_COLS)
            reconstructor.fit(tr_df_scaled_lite.values)
            reconstructor.build_fold_cache(tr_df)
            del tr_df_scaled_lite; gc.collect()

            # [MEMORY_OPTIMIZATION] Zero-Copy Population (v18.0)
            # Step 1: Process Train for Pruning (Avoid Wide DataFrame)
            logger.info(f"[FOLD {fold}] Extracting features for pruning...")
            
            # 1a. Scale raw features
            tr_raw_scaled = scaler.transform(tr_df[FEATURE_SCHEMA['raw_features']], FEATURE_SCHEMA['raw_features'])
            X_raw_part = tr_raw_scaled[[f for f in initial_candidates if f in FEATURE_SCHEMA['raw_features']]].values.astype(np.float32)
            
            # 1b. Get latent stats directly
            latent_stats = reconstructor.calculate_graph_stats(tr_df)
            
            # 1c. Build temporary X for pruning without creating a full DF
            latent_cols = [f for f in initial_candidates if f in FEATURE_SCHEMA['embed_features']]
            X_latent_part = np.zeros((len(tr_df), len(latent_cols)), dtype=np.float32)
            
            for i, col in enumerate(latent_cols):
                # [CONTRACT_TRACE]
                logger.debug(f"[DIM_TRACE] processing latent column {col} | index {i}")
                
                # Check if it's a multi-dim feature (ends with _d<digits>)
                import re
                match = re.search(r'_d(\d+)$', col)
                
                if match:
                    base_name = col[:match.start()]
                    dim = int(match.group(1))
                    
                    # [CONTRACT_ENFORCEMENT] latent_stats[base_name] must be (N, Dim)
                    source_val = latent_stats[base_name]
                    if source_val.ndim != 2:
                        raise ValueError(f"[CONTRACT_FAIL] {base_name} expected 2D, got {source_val.shape} | source: data_loader.calculate_graph_stats")
                    
                    # [DIM_TRACE]
                    logger.debug(f"[DIM_TRACE] {col} source shape: {source_val[:, dim].shape}")
                    X_latent_part[:, i] = source_val[:, dim]
                else:
                    # [CONTRACT_ENFORCEMENT] latent_stats[col] must be (N,)
                    source_val = latent_stats[col]
                    
                    # [ROOT_CAUSE_PROBE] Trace why this might be (N, 1) or (N,)
                    logger.debug(f"[DIM_TRACE] {col} source shape: {source_val.shape}")
                    
                    # [CONTRACT_ENFORCEMENT] Strict (N,) check
                    if source_val.ndim != 1:
                        # [ROOT_CAUSE_IDENTIFIED] If this fails, calculate_graph_stats is returning (N, 1) instead of (N,)
                        raise ValueError(f"[CONTRACT_FAIL] {col} expected 1D (N,), got {source_val.shape}. NO AUTO-FIX ALLOWED.")
                        
                    X_latent_part[:, i] = source_val
            
            X_tr_temp = np.hstack([X_raw_part, X_latent_part])
            temp_feat_names = [f for f in initial_candidates if f in FEATURE_SCHEMA['raw_features']] + latent_cols
            
            del tr_raw_scaled, latent_stats, X_raw_part, X_latent_part; gc.collect()

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
            
            del X_tr_temp, temp_model, imp_df; gc.collect()
            memory_guard(f"Fold {fold} - Post Train Pruning", logger)
            
            # Step 2: Process Validation with ONLY selected features
            logger.info(f"[FOLD {fold}] Populating validation features (Selected Only)...")
            val_df_full = apply_latent_features(val_df, reconstructor, scaler=scaler, selected_features=fold_features)
            X_val = val_df_full[fold_features].values.astype(np.float32)
            del val_df_full; gc.collect()
            
            # Step 3: Process Test with ONLY selected features
            logger.info(f"[FOLD {fold}] Populating test features (Selected Only)...")
            test_df_full = apply_latent_features(test_df_fold, reconstructor, scaler=scaler, selected_features=fold_features)
            X_test_f = test_df_full[fold_features].values.astype(np.float32)
            del test_df_full; gc.collect()
            memory_guard(f"Fold {fold} - Post Test", logger)
            
            # 2. Artifact Preservation
            import pickle
            os.makedirs(f'{Config.MODELS_PATH}/reconstructors', exist_ok=True)
            with open(f'{Config.MODELS_PATH}/reconstructors/recon_fold_{fold}.pkl', 'wb') as f:
                pickle.dump(reconstructor, f)
            with open(f'{Config.MODELS_PATH}/reconstructors/scaler_fold_{fold}.pkl', 'wb') as f:
                pickle.dump(scaler, f)
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
            
            # 4. Weight Calculation (Reliability)
            weights = (1.0 + 2.0 * adv_probs)
            
            # [STABILITY_FIX] Ensure y_tr is accessible and threshold calculation is safe
            y_tr_val = y_tr.values if isinstance(y_tr, pd.Series) else y_tr
            threshold = np.quantile(y_tr_val, Config.EXTREME_TARGET_QUANTILE)
            weights[y_tr_val >= threshold] *= Config.EXTREME_SAMPLE_WEIGHT
            
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
        
        if Config.MODE in ['trace', 'debug'] or Config.SMOKE_TEST:
            logger.warning(f"[DIST_GUARD] Skipping failure in {Config.MODE} (Smoke={Config.SMOKE_TEST}) mode")
            return ratio
            
        if ratio < 0.7 or ratio > 1.5:
            raise RuntimeError(f"FAIL: std_ratio {ratio:.4f} outside [0.7, 1.5]")
        return ratio

    def analyze_model_divergence(self):
        if 'raw' in self.oof_preds and 'final' in self.oof_preds:
            r, e = self.oof_preds['raw'], self.oof_preds['final']
            corr = np.corrcoef(r, e)[0, 1]
            logger.info(f"[MODEL_DIVERGENCE] corr_raw_final={corr:.4f}")
            return corr, 0.0
        return 1.0, 0.0
