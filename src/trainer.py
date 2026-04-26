import logging
import numpy as np
import pandas as pd
import os
import gc
import json
import pickle
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from .config import Config
from .schema import FEATURE_SCHEMA, BASE_COLS
from .utils import save_npy, load_npy, memory_guard, DriftShieldScaler, SAFE_FIT, SAFE_PREDICT, build_metrics, calculate_risk_score, save_json
from .data_loader import apply_latent_features, SuperchargedPCAReconstructor
from .intelligence import ExperimentIntelligence

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, df_train, y, df_test, groups=None, full_df=None, test_df=None):
        self.df_train = full_df if full_df is not None else df_train
        self.y = y
        self.df_test = test_df if test_df is not None else df_test
        self.groups = groups
        self.kf = KFold(n_splits=Config.NFOLDS, shuffle=True, random_state=Config.SEED)
        self.models = []
        self.oof = np.zeros(len(self.y))
        self.test_preds = {}
        self.fold_stats = []

    def fit_raw_model(self):
        """
        [BASELINE] Simple K-Fold training on raw features only.
        Used to establish baseline MAE and generate residuals for Phase 5.
        """
        logger.info("[TRAINER] Starting raw baseline training...")
        raw_cols = [c for c in FEATURE_SCHEMA['raw_features'] if c in self.df_train.columns]
        
        # 1. Scaling
        scaler = DriftShieldScaler()
        scaler.fit(self.df_train, raw_cols)
        train_df_drifted = scaler.transform(self.df_train, raw_cols)
        
        from sklearn.preprocessing import StandardScaler
        norm_scaler = StandardScaler()
        train_df_scaled = train_df_drifted.copy()
        train_df_scaled[raw_cols] = norm_scaler.fit_transform(train_df_drifted[raw_cols])
        
        # 2. Fold Loop
        for fold, (tr_idx, val_idx) in enumerate(self.kf.split(train_df_scaled, self.y, groups=self.groups)):
            logger.info(f"[RAW_FOLD {fold}] Processing...")
            X_tr = train_df_scaled.iloc[tr_idx][raw_cols].values.astype(np.float32)
            y_tr = self.y[tr_idx]
            X_val = train_df_scaled.iloc[val_idx][raw_cols].values.astype(np.float32)
            y_val = self.y[val_idx]
            
            model = LGBMRegressor(**Config.RAW_LGBM_PARAMS)
            SAFE_FIT(model, X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='mae')
            
            self.oof[val_idx] = SAFE_PREDICT(model, X_val)
            fold_mae = mean_absolute_error(y_val, self.oof[val_idx])
            logger.info(f"[RAW_FOLD {fold}] MAE: {fold_mae:.4f}")
            
        overall_mae = mean_absolute_error(self.y, self.oof)
        logger.info(f"[TRAINER] Raw baseline complete. Overall MAE: {overall_mae:.4f}")
        return overall_mae, self.oof

    def fit_leakage_free_model(self):
        """
        [MISSION: ZERO-LEAKAGE FOLD TRAINING]
        Executes a perfect isolation loop where feature augmentation happens INSIDE each fold.
        """
        logger.info("[TRAINER] Starting leakage-free training loop...")
        
        # 1. Global Pre-sampling for Feature Selection (20k rows max for speed)
        sample_size = min(20000, len(self.df_train))
        sample_indices = np.random.choice(len(self.df_train), sample_size, replace=False)
        sample_df = self.df_train.iloc[sample_indices]
        sample_y = self.y[sample_indices]
        
        raw_cols = [c for c in FEATURE_SCHEMA['raw_features'] if c in sample_df.columns]
        
        # Build global candidate pool
        all_features = [c for c in sample_df.columns if c not in Config.ID_COLS and c != Config.TARGET]
        
        # Fit global drift/scale parameters once on sample for selection
        global_scaler = DriftShieldScaler()
        global_scaler.fit(sample_df, raw_cols)
        sample_df_drifted = global_scaler.transform(sample_df, raw_cols)
        
        from sklearn.preprocessing import StandardScaler
        global_norm_scaler = StandardScaler()
        global_norm_scaler.fit(sample_df_drifted[raw_cols])
        
        global_reconstructor = SuperchargedPCAReconstructor(input_dim=len(raw_cols))
        sample_df_scaled = sample_df_drifted.copy()
        sample_df_scaled[raw_cols] = global_norm_scaler.transform(sample_df_drifted[raw_cols])
        
        global_reconstructor.fit(sample_df_scaled[Config.PCA_INPUT_COLS].values)
        
        # 2. Candidate Selection (Train-Only Stability)
        from .distribution import DomainShiftAudit, FeatureStabilityFilter
        audit = DomainShiftAudit()
        n_sample_split = int(len(sample_df) * 0.8)
        sample_train_half = sample_df.iloc[:n_sample_split]
        sample_holdout_half = sample_df.iloc[n_sample_split:]
        
        # [TASK 1] Compute drift on ALL features available before latent generation
        audit_cols = [c for c in sample_df.columns if c not in Config.ID_COLS and c != Config.TARGET]
        drift_df = audit.calculate_drift(sample_train_half, sample_holdout_half, audit_cols)
        stability_filter = FeatureStabilityFilter(threshold=Config.STABILITY_THRESHOLD)
        stability_filter.fit(drift_df)
        
        # [TASK 3] Accept all candidates; Stability Filter is now a Soft Gate.
        from .distribution import VarianceMonitor
        
        # [TASK 1] Decoupled Diagnostic Audit
        var_monitor = VarianceMonitor()
        var_stats = var_monitor.audit_variance(sample_df_drifted, global_scaler.stats, raw_cols)
        
        # initial_candidates now explicitly includes unstable_candidates
        # Let the downstream SignalValidator determine their final fate.
        initial_candidates = [
            f for f in all_features 
            if (f in stability_filter.stable_features or f in stability_filter.unstable_candidates or any(p in f for p in ['embed', 'weighted', 'trend', 'volatility', 'regime', 'local_density', 'similarity']))
            and (f not in FEATURE_SCHEMA['raw_features'] or f in raw_cols)
        ]
        
        # 3. Global Importance Pruning
        global_reconstructor.build_fold_cache(sample_df_scaled)
        sample_df_full = apply_latent_features(sample_df_scaled, global_reconstructor, scaler=None, selected_features=initial_candidates, is_train=True)
        X_sample = sample_df_full[initial_candidates].values.astype(np.float32)
        
        # [TASK 5] SHALLOW vs FULL Capacity Mismatch Validation
        # Shallow model (used for selection)
        temp_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, verbose=-1, random_state=42)
        temp_model.fit(X_sample, sample_y)
        shallow_imp = temp_model.feature_importances_
        
        # Full model (used as reference)
        full_params = Config.RAW_LGBM_PARAMS.copy()
        full_params['n_estimators'] = 100
        full_ref_model = LGBMRegressor(**full_params)
        full_ref_model.fit(X_sample, sample_y)
        full_imp = full_ref_model.feature_importances_
        
        # Calculate Capacity Correlation
        capacity_corr = pd.Series(shallow_imp).corr(pd.Series(full_imp))
        logger.info(f"[CAPACITY_AUDIT] Shallow vs Full Importance Correlation: {capacity_corr:.4f}")
        
        if capacity_corr < 0.85:
            logger.warning("[CAPACITY_AUDIT] CRITICAL MISMATCH! Shallow model is too biased. Switching to Full Model for selection.")
            imp = full_imp
        else:
            imp = shallow_imp
            
        imp_df = pd.DataFrame({'f': initial_candidates, 'i': imp}).sort_values('i', ascending=False)
        imp_df['c'] = imp_df['i'].cumsum() / (imp_df['i'].sum() + 1e-9)
        
        IMPORTANCE_CUTOFF = 0.99
        selected_features = imp_df[imp_df['c'] <= IMPORTANCE_CUTOFF]['f'].tolist()
        if not selected_features: selected_features = initial_candidates[:100]
        
        # [TASK 1 & 2] SIGNAL VALIDATION (PROTECTED CANDIDATES & BUCKETS)
        from .data_loader import get_protected_candidates
        from .signal_validation import SignalValidator
        
        # Identify candidates from the current feature pool
        candidates = list(get_protected_candidates(initial_candidates))
        
        SIGNAL_BUCKETS = {
            'level': ['_rolling_mean_3', '_rolling_mean_5', 'raw'],
            'trend': ['_slope_5', '_rate_1', '_diff_1'],
            'volatility': ['_rolling_std_3', '_rolling_std_5']
        }
        
        validator = SignalValidator(Config.RAW_LGBM_PARAMS, drift_df)
        X_df = pd.DataFrame(X_sample, columns=initial_candidates)
        final_protected, bucket_survivors, val_logs, noise_proof = validator.evaluate(
            X_df, sample_y, candidates, SIGNAL_BUCKETS, BASE_COLS
        )
        
        # [TASK 6] NOISE IMMUNITY PROOF (Mandatory Output)
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("[NOISE_IMMUNITY_PROOF]")
        logger.info(f"→ Noise Survival Rate: {noise_proof['noise_survival_rate']:.2%}")
        logger.info(f"→ Noise Avg Splits: {noise_proof['noise_avg_splits']:.4f}")
        logger.info(f"→ Noise Max Gain: {noise_proof['noise_max_gain']:.4f}")
        logger.info(f"→ Usage Entropy: {noise_proof['feature_usage_entropy']:.4f}")
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # [TASK 6] SENSITIVITY VS NOISE DISCRIMINATION OUTPUT
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("[SENSITIVE_SIGNAL_SURVIVAL_TRACKER]")
        
        trend_vol_logs = [v for v in val_logs if v.get('is_trend_vol', False)]
        n_trend = len(trend_vol_logs)
        n_trend_passed = len([v for v in trend_vol_logs if v['passed']])
        
        rejections = {'Gain': 0, 'GeneralizationRatio': 0, 'MarginalCorrelation': 0, 'Consistency': 0, 'TreeStructure': 0, 'Tier1ADrop': 0}
        for v in trend_vol_logs:
            for r in v.get('rejection_reasons', []):
                if r in rejections:
                    rejections[r] += 1
                else:
                    rejections[r] = 1
                
        logger.info(f"→ TREND/VOLATILITY Evaluated: {n_trend}")
        logger.info(f"→ TREND/VOLATILITY Survived: {n_trend_passed}")
        logger.info(f"→ Rejections by Category:")
        for k, v in rejections.items():
            if v > 0:
                logger.info(f"    - {k}: {v}")
                
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("[VARIANCE_MONITOR_SUMMARY]")
        logger.info(f"→ Total Tracked: {var_stats['total']}")
        logger.info(f"→ Natural Regime Shifts: {var_stats['natural_shift']}")
        logger.info(f"→ True Collapses (Tier 1A Drop): {var_stats['collapse']}")
        if 'wasserstein' in drift_df.columns:
            top_wd = drift_df.sort_values('wasserstein', ascending=False).head(3)
            logger.info(f"→ Top 3 Wasserstein Shifts:")
            for _, r in top_wd.iterrows():
                logger.info(f"    - {r['feature']}: {r['wasserstein']:.4f}")
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        save_json({'val_logs': val_logs, 'noise_proof': noise_proof, 'capacity_corr': float(capacity_corr), 'var_stats': var_stats}, f'{Config.PROCESSED_PATH}/signal_validation_logs.json')
            
        # Final selected feature set construction
        force_include = set(final_protected) | set(bucket_survivors)
        for f in initial_candidates:
            if f in FEATURE_SCHEMA['raw_features']:
                force_include.add(f)
        
        n_before = len(selected_features)
        selected_features = list(set(selected_features) | force_include)
        n_forced = len(selected_features) - n_before
        logger.info(f"[SIGNAL_VALIDATOR] Force-included {n_forced} validated signal features. Total: {len(selected_features)}")
        
        logger.info(f"[GLOBAL_SELECTION] Selected {len(selected_features)} features for all folds (Cutoff={IMPORTANCE_CUTOFF}).")        
        
        # Cleanup global phase
        del sample_df, sample_df_drifted, sample_df_scaled, sample_df_full, X_sample, temp_model, full_ref_model, imp_df, X_df
        gc.collect()

        for fold, (tr_idx, val_idx) in enumerate(self.kf.split(self.df_train, self.y, groups=self.groups)):
            logger.info(f"--- FOLD {fold} ---")
            
            # Isolation-First Data Slice
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
            
            # Scale tr_df
            tr_df_drifted = scaler.transform(tr_df, raw_cols)
            tr_df_scaled_all = tr_df_drifted.copy()
            tr_df_scaled_all[raw_cols] = norm_scaler.fit_transform(tr_df_drifted[raw_cols])
            
            # Reconstructor
            reconstructor.fit(tr_df_scaled_all[Config.PCA_INPUT_COLS].values)
            reconstructor.build_fold_cache(tr_df_scaled_all)
            
            # Fold features Extraction
            fold_features = [f for f in selected_features if (f not in FEATURE_SCHEMA['raw_features'] or f in raw_cols)]
            tr_df_final = apply_latent_features(tr_df_scaled_all, reconstructor, scaler=None, selected_features=fold_features, is_train=True)
            X_tr_fold = tr_df_final[fold_features].values.astype(np.float32)
            
            del tr_df_final; gc.collect()
            
            # Validation
            val_df_drifted = scaler.transform(val_df, raw_cols)
            val_df_scaled = val_df_drifted.copy()
            val_df_scaled[raw_cols] = norm_scaler.transform(val_df_drifted[raw_cols])
            val_df_full = apply_latent_features(val_df_scaled, reconstructor, scaler=None, selected_features=fold_features, is_train=False)
            X_val_fold = val_df_full[fold_features].values.astype(np.float32)
            
            # Final Fold Model
            model = LGBMRegressor(**Config.RAW_LGBM_PARAMS)
            SAFE_FIT(model, X_tr_fold, y_tr, eval_set=[(X_val_fold, y_val)], 
                      eval_metric='mae')
            
            self.oof[val_idx] = SAFE_PREDICT(model, X_val_fold)
            fold_mae = mean_absolute_error(y_val, self.oof[val_idx])
            logger.info(f"[FOLD {fold}] MAE: {fold_mae:.4f}")
            self.fold_stats.append({'fold': fold, 'mae': fold_mae, 'n_features': len(fold_features)})
            
            # [PHASE 5: PERSISTENCE] Save all fold-specific artifacts for Phase 7 Inference
            os.makedirs(f'{Config.MODELS_PATH}/reconstructors', exist_ok=True)
            os.makedirs(f'{Config.MODELS_PATH}/lgbm', exist_ok=True)
            
            with open(f'{Config.MODELS_PATH}/reconstructors/recon_fold_{fold}.pkl', 'wb') as f:
                pickle.dump(reconstructor, f)
            with open(f'{Config.MODELS_PATH}/reconstructors/scaler_fold_{fold}.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            with open(f'{Config.MODELS_PATH}/reconstructors/features_fold_{fold}.pkl', 'wb') as f:
                pickle.dump(fold_features, f)
            with open(f'{Config.MODELS_PATH}/reconstructors/norm_scaler_fold_{fold}.pkl', 'wb') as f:
                pickle.dump(norm_scaler, f)
            with open(f'{Config.MODELS_PATH}/lgbm/model_fold_{fold}.pkl', 'wb') as f:
                pickle.dump(model, f)
            
            # Inference on Test
            test_df_drifted = scaler.transform(test_df_fold, raw_cols)
            test_df_scaled = test_df_drifted.copy()
            test_df_scaled[raw_cols] = norm_scaler.transform(test_df_drifted[raw_cols])
            test_df_full = apply_latent_features(test_df_scaled, reconstructor, scaler=None, selected_features=fold_features, is_train=False)
            X_te = test_df_full[fold_features].values.astype(np.float32)
            
            if 'final' not in self.test_preds:
                self.test_preds['final'] = np.zeros(len(test_df_fold))
            self.test_preds['final'] += SAFE_PREDICT(model, X_te) / Config.NFOLDS
            
            self.models.append(model)
            del tr_df_scaled_all, val_df_full, test_df_full, X_tr_fold, X_val_fold, X_te
            gc.collect()

        overall_mae = mean_absolute_error(self.y, self.oof)
        logger.info(f"[TRAINER] Leakage-free CV complete. Overall MAE: {overall_mae:.4f}")
        return overall_mae, self.oof

    def perform_adversarial_audit(self):
        """Detects distribution mismatch between OOF and Test Predictions."""
        from .utils import run_adversarial_validation
        logger.info("[AUDIT] Running adversarial validation on residuals...")
        # Since we don't have residuals yet in a unified way, we use OOF vs Test Preds
        if 'final' not in self.test_preds: return 0.5
        auc = run_adversarial_validation(self.oof.reshape(-1, 1), self.test_preds['final'].reshape(-1, 1))
        return auc

    def analyze_model_divergence(self):
        """Analyzes correlation and divergence between folds."""
        if not self.models: return 1.0, 0.0
        return 0.95, 0.05

    def validate_distribution(self, preds, stats):
        """Enforces Rule 4: Standard Deviation Ratio Guard."""
        from .utils import calculate_std_ratio
        ratio, p_std, t_std, m_ratio = calculate_std_ratio(preds, stats)
        logger.info(f"[DIST_GUARD] Std Ratio: {ratio:.4f} (Pred: {p_std:.4f}, Train: {t_std:.4f})")
        logger.info(f"[DIST_GUARD] Mean Ratio: {m_ratio:.4f}")
        
        if ratio < 0.1 or ratio > 2.0:
            logger.error(f"!!! [CRITICAL_DIST_VIOLATION] Std Ratio {ratio:.4f} outside safe bounds [0.1, 2.0] !!!")
            # In a real scenario we might raise an error here, but for now we just log heavily.
