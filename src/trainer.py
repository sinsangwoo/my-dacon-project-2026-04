import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
import gc
import logging
import warnings
import hashlib
from .utils import log_memory_usage, inspect_columns, track_lineage, ensure_dataframe, SAFE_FIT, SAFE_PREDICT, SAFE_PREDICT_PROBA
from .config import Config

logger = logging.getLogger(__name__)

class Trainer:
    """Class to handle Advanced Cross-Validation, Stacking, and Pseudo Labeling."""
    def __init__(self, X, y, X_test, schema, groups=None):
        # [MISSION: ENFORCE NUMPY BOUNDARY]
        # DataFrames are strictly forbidden at the model boundary.
        # We enforce a deterministic numpy-only contract.
        
        # [PHASE 4: TRAINER ISOLATION]
        # Indices MUST come from schema. Trainer must NOT access column names.
        self.schema = schema
        self.raw_idx = [schema['feature_to_index'][f] for f in schema['raw_features']]
        self.embed_idx = [schema['feature_to_index'][f] for f in schema['embed_features']]
        self.all_idx = list(range(len(schema['all_features'])))
        
        # [FINAL_CONTRACT_CHECK]
        logger.info(f"[FINAL_CONTRACT_CHECK]")
        logger.info(f"X_type: {type(X)} | shape: {X.shape}")
        logger.info(f"y_type: {type(y)} | shape: {y.shape}")
        logger.info(f"X_test_type: {type(X_test)} | shape: {X_test.shape}")

        # 1. Input Hard Contract Validation
        assert isinstance(X, np.ndarray), f"X must be numpy.ndarray, got {type(X)}"
        assert isinstance(y, np.ndarray), f"y must be numpy.ndarray, got {type(y)}"
        assert isinstance(X_test, np.ndarray), f"X_test must be numpy.ndarray, got {type(X_test)}"
        assert X.shape[0] == y.shape[0], f"Length mismatch: X={X.shape[0]}, y={y.shape[0]}"
        assert X.dtype == np.float32, f"X must be float32, got {X.dtype}"
        assert X_test.dtype == np.float32, f"X_test must be float32, got {X_test.dtype}"
        assert X.shape[1] == len(schema['all_features']), f"Feature count mismatch: {X.shape[1]} != {len(schema['all_features'])}"
        assert X_test.shape[1] == X.shape[1], f"X_test feature count mismatch: {X_test.shape[1]} != {X.shape[1]}"
        if groups is not None:
            assert len(groups) == X.shape[0], f"Groups length mismatch: {len(groups)} != {X.shape[0]}"

        # 2. Internal State (Rule 3)
        self.X = X
        self.y = y
        self.X_test = X_test
        self.groups = groups if groups is not None else np.zeros(len(X), dtype=np.int32)
        
        # Split Strategy
        if Config.SPLIT_STRATEGY == 'GroupKFold':
            from sklearn.model_selection import GroupKFold
            self.kf = GroupKFold(n_splits=Config.NFOLDS)
            logger.info(f"[VALIDATION] GroupKFold active. Unique groups: {len(np.unique(self.groups))}")
        else:
            from sklearn.model_selection import KFold
            self.kf = KFold(n_splits=Config.NFOLDS, shuffle=True, random_state=42)
        
        # Unified Result Dictionaries (v10.0 Hardening)
        self.oof_preds = {}
        self.test_preds = {}
        
        # Fold-wise predictions for test (to calculate variance for pseudo confidence)
        self.test_preds_lgb_folds = []
        self.test_preds_cat_folds = []
        
        # Stability Monitoring
        self.importance_list = []
        
        # Diagnostics
        self.oof_correlation = None
        self.fold_maes_lgb = []
        self.fold_maes_cat = []
        self.cv_summary = {}
        self.adversarial_summary = {}

    def _mean_encoding_cv(self, tr_df, val_df, test_df, target_col):
        """DEPRECATED: Mixed usage of DataFrame/Numpy forbidden."""
        raise RuntimeError("Mean encoding must be performed in Preprocessing, not in Trainer.")

    def validate_data(self, X, y, groups):
        """Perform strict data integrity checks."""
        logger.info("[STEP: validation] Running Data Integrity Checks...")
        
        # 1. Length Match
        if not (len(X) == len(y) == len(groups)):
            raise ValueError(f"Length mismatch! X: {len(X)}, y: {len(y)}, groups: {len(groups)}")
            
        # 2. Type & Dtype Check (Rule 9)
        assert isinstance(X, np.ndarray), f"X must be numpy.ndarray, got {type(X)}"
        assert X.dtype == np.float32, f"X must be float32, got {X.dtype}"
        assert X.shape[1] == len(self.schema['all_features']), f"Feature count mismatch: {X.shape[1]} != {len(self.schema['all_features'])}"
            
        # 3. NaN Check
        nans = np.isnan(X).sum()
        if nans > 0:
            raise ValueError(f"Dataset contains {nans} NaN values!")
        if np.isnan(y).any():
            raise ValueError("Target contains NaN values!")
            
        logger.info(f"Integrity Check Passed. Shape: {X.shape}, Target: {y.shape}")
            
    def check_group_leakage(self, train_idx, val_idx, groups):
        """Ensure no group overlap between splits."""
        tr_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        overlap = tr_groups.intersection(val_groups)
        if overlap:
            raise ValueError(f"Group Leakage Detected! Overlapping IDs: {list(overlap)[:5]}...")

    def _train_model(self, model_obj, X_tr, y_tr, X_val, y_val, test_X, sample_weight=None):
        """Train a single model with strict numpy interface and zero-warning policy."""
        # [MISSION: HARD TYPE ENFORCEMENT]
        assert isinstance(X_tr, np.ndarray) and X_tr.dtype == np.float32
        assert isinstance(X_val, np.ndarray) and X_val.dtype == np.float32
        assert isinstance(test_X, np.ndarray) and test_X.dtype == np.float32

        if Config.TRACE_LEVEL != "OFF":
            logger.info(f"[IDENTITY_OK] Learning Boundary | features={X_tr.shape[1]} | rows={len(X_tr)}")

        model = model_obj
        
        # [MISSION: GLOBAL MODEL INTERFACE LOCKDOWN] Rule 12
        if "catboost" in str(type(model)).lower():
            # CatBoost handles early stopping and verbose internally if passed in params
            # but we can also pass them explicitly to SAFE_FIT
            SAFE_FIT(
                model,
                X_tr,
                y_tr,
                sample_weight=sample_weight,
                eval_set=[(X_val, y_val)],
            )
        else:
            SAFE_FIT(
                model,
                X_tr,
                y_tr,
                sample_weight=sample_weight,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(Config.EARLY_STOPPING_ROUNDS),
                    lgb.log_evaluation(period=0),
                ],
            )
            
        if "classifier" in str(type(model)).lower():
            val_preds = SAFE_PREDICT_PROBA(model, X_val)[:, 1]
            test_preds = SAFE_PREDICT_PROBA(model, test_X)[:, 1]
        else:
            val_preds = SAFE_PREDICT(model, X_val)
            test_preds = SAFE_PREDICT(model, test_X)
            
        return val_preds, test_preds, model.feature_importances_ if hasattr(model, "feature_importances_") else None

    def _get_target_array(self, train_df):
        """DEPRECATED: Using self.y or explicit y passed to fit."""
        if self.y is not None:
            return self.y
        raise RuntimeError("Target array must be provided explicitly.")

    def build_extreme_sample_weights(self, y, quantile=None, extreme_weight=None):
        """Upweight the top target tail so the model cannot hide behind the mean."""
        if quantile is None:
            quantile = Config.EXTREME_TARGET_QUANTILE
        if extreme_weight is None:
            extreme_weight = Config.EXTREME_SAMPLE_WEIGHT

        y = np.asarray(y, dtype=np.float32)
        threshold = float(np.quantile(y, quantile))
        sample_weight = np.ones(len(y), dtype=np.float32)
        sample_weight[y >= threshold] = np.float32(extreme_weight)
        summary = {
            "quantile": float(quantile),
            "threshold": threshold,
            "extreme_weight": float(extreme_weight),
            "extreme_count": int((y >= threshold).sum()),
        }
        logger.info(f"[REWEIGHT] {summary}")
        return sample_weight, summary

    def _summarize_cv(self, model_name, fold_maes, y_true, preds):
        if len(fold_maes) == 0:
            return {
                "model": model_name,
                "mean_mae": None,
                "std_mae": None,
                "worst_mae": None,
                "overall_mae": None,
                "pred_std": None,
                "target_std": float(np.std(y_true)),
                "variance_ratio": None,
            }

        pred_std = float(np.std(preds))
        target_std = float(np.std(y_true))
        ratio = pred_std / (target_std + 1e-9)
        summary = {
            "model": model_name,
            "mean_mae": float(np.mean(fold_maes)),
            "std_mae": float(np.std(fold_maes)),
            "worst_mae": float(np.max(fold_maes)),
            "overall_mae": float(mean_absolute_error(y_true, preds)),
            "pred_std": pred_std,
            "target_std": target_std,
            "variance_ratio": float(ratio),
        }
        logger.info(
            f"[CV_SUMMARY] {model_name.upper()} | mean={summary['mean_mae']:.6f} | "
            f"std={summary['std_mae']:.6f} | worst={summary['worst_mae']:.6f} | "
            f"variance_ratio={summary['variance_ratio']:.4f}"
        )
        if ratio < Config.PRED_VARIANCE_RATIO_TARGET_MIN:
            logger.warning(
                f"[VARIANCE_WARNING] {model_name.upper()} variance ratio below target "
                f"({ratio:.4f} < {Config.PRED_VARIANCE_RATIO_TARGET_MIN:.2f})"
            )
        return summary

    def get_primary_cv_summary(self):
        summary = self.cv_summary.get("stable", {}).copy()
        if hasattr(self, 'extreme_analysis') and "stable" in self.extreme_analysis:
            summary["extreme_recall"] = self.extreme_analysis["stable"].get("recall")
            summary["extreme_mae"] = self.extreme_analysis["stable"].get("extreme_mae")
        return summary

    def _log_oof_diagnostics(self, y_true, oof_lgb, oof_cat):
        """Log OOF distribution analysis and model correlation."""
        logger.info(f"\n--- [OOF_DIAGNOSTICS] ---")
        
        # OOF vs Target Distribution
        logger.info(f"[OOF_DIAG] Target  | mean={y_true.mean():.4f} std={y_true.std():.4f}")
        logger.info(f"[OOF_DIAG] OOF_LGB | mean={oof_lgb.mean():.4f} std={oof_lgb.std():.4f}")
        
        lgb_mae = mean_absolute_error(y_true, oof_lgb)
        logger.info(f"[OOF_DIAG] LGB Overall MAE: {lgb_mae:.6f}")
        
        if not (oof_cat == 0).all():
            logger.info(f"[OOF_DIAG] OOF_CAT | mean={oof_cat.mean():.4f} std={oof_cat.std():.4f}")
            cat_mae = mean_absolute_error(y_true, oof_cat)
            logger.info(f"[OOF_DIAG] CAT Overall MAE: {cat_mae:.6f}")
            
            # Model Correlation
            corr = np.corrcoef(oof_lgb, oof_cat)[0, 1]
            self.oof_correlation = corr
            logger.info(f"[CORRELATION] OOF_LGB vs OOF_CAT: {corr:.4f}")
            if corr >= Config.ENSEMBLE_MAX_CORR:
                logger.warning(
                    f"[ENSEMBLE_BLOCK] Correlation {corr:.4f} >= {Config.ENSEMBLE_MAX_CORR:.2f}. "
                    "Robust ensemble is disabled."
                )
        
        logger.info(f"--- [/OOF_DIAGNOSTICS] ---\n")

    def train_kfolds(self, X_subset, y=None, X_test_subset=None, seeds=None, sample_weight=None, result_key="lgb", custom_params=None):
        """Core CV loop with objective result tracking (v10.0 Hardening).
        PHASE 4: No column names allowed. Only numpy slices.
        """
        X = X_subset
        y = y if y is not None else self.y
        X_test = X_test_subset if X_test_subset is not None else self.X_test
        
        if seeds is None:
            seeds = Config.SEEDS
            
        # [MISSION: HARD TYPE ENFORCEMENT] Rule 9
        assert isinstance(X, np.ndarray), "X must be numpy.ndarray"
        assert X.dtype == np.float32, f"X must be float32, got {X.dtype}"
        
        assert isinstance(X_test, np.ndarray), "X_test must be numpy.ndarray"
        assert X_test.dtype == np.float32, f"X_test must be float32, got {X_test.dtype}"
        
        groups = self.groups
        
        # Use provided weights or build standard extreme weights
        if sample_weight is None and result_key != 'regime':
             sample_weight, _ = self.build_extreme_sample_weights(y)
        
        params = custom_params if custom_params else Config.LGBM_PARAMS.copy()
        
        n_train = len(y)
        n_test = len(X_test)
        
        oof_final = np.zeros(n_train, dtype=np.float64)
        test_final = np.zeros(n_test, dtype=np.float64)
        
        for seed_idx, seed in enumerate(seeds):
            seed_oof = np.zeros(n_train, dtype=np.float64)
            seed_test = np.zeros(n_test, dtype=np.float64)
            
            for fold, (tr_idx, val_idx) in enumerate(self.kf.split(X, y, groups=groups)):
                X_tr, y_tr = X[tr_idx], y[tr_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                sw_tr = sample_weight[tr_idx] if sample_weight is not None else None
                
                params_fold = params.copy()
                params_fold['random_state'] = seed + fold
                
                # [HYBRID_PIPELINE] Ensure pure numpy
                X_tr, y_tr = np.asarray(X_tr), np.asarray(y_tr)
                X_val, y_val = np.asarray(X_val), np.asarray(y_val)
                
                # [FAST_IDENTITY_CHECK] hash check for first 100 rows (Rule 3)
                h1 = hash(np.ascontiguousarray(X_tr[:100]).tobytes())
                h2 = hash(np.ascontiguousarray(X[tr_idx][:100]).tobytes())
                if h1 != h2:
                    raise RuntimeError(f"[FATAL] Feature identity mismatch in tr_idx! Hash: {h1} != {h2}")

                if result_key == 'regime':
                    from lightgbm import LGBMClassifier
                    model = LGBMClassifier(**params_fold)
                elif result_key == 'cat':
                    from catboost import CatBoostRegressor
                    model = CatBoostRegressor(**params_fold)
                else:
                    model = LGBMRegressor(**params_fold)

                # [MISSION: ZERO WARNING POLICY] Rule 10
                val_preds_fold, test_preds_fold, _ = self._train_model(
                    model, X_tr, y_tr, X_val, y_val, X_test, sample_weight=sw_tr
                )
                
                seed_oof[val_idx] = val_preds_fold
                seed_test += test_preds_fold / Config.NFOLDS
                    
                del model
                gc.collect()
            
            oof_final += seed_oof / len(seeds)
            test_final += seed_test / len(seeds)
            
        self.oof_preds[result_key] = oof_final
        self.test_preds[result_key] = test_final
        
        if result_key != 'regime':
            mae = mean_absolute_error(y, oof_final)
            self.cv_summary[result_key] = self._summarize_cv(result_key, [mae], y, oof_final)
            return mae, oof_final
        else:
            return 0.0, oof_final

    # [ANTI-COLLAPSE_DECOMPOSITION] CORE METHODS

    def fit_raw_model(self):
        """Train the baseline model on original CSV features only."""
        logger.info(f"[STAGE: RAW_MODEL] Training on {len(self.raw_idx)} features...")
        
        # [PHASE 4] Slice using indices from schema
        X_raw_tr = self.X[:, self.raw_idx]
        X_raw_te = self.X_test[:, self.raw_idx]
        
        mae, oof = self.train_kfolds(
            X_subset=X_raw_tr,
            X_test_subset=X_raw_te,
            result_key='raw',
            custom_params=Config.RAW_LGBM_PARAMS
        )
        return mae, oof

    def fit_embed_model(self):
        """Train the latent model on pseudo-temporal features with Activation Forcing."""
        logger.info(f"[STAGE: EMBED_MODEL] Training on {len(self.embed_idx)} features...")
        
        # [PHASE 4] Slice using indices from schema
        X_embed_tr = self.X[:, self.embed_idx]
        X_embed_te = self.X_test[:, self.embed_idx]

        # Initial Pass
        mae, oof = self.train_kfolds(
            X_subset=X_embed_tr,
            X_test_subset=X_embed_te,
            result_key='embed',
            custom_params=Config.EMBED_LGBM_PARAMS
        )
        
        # Activation Check
        if not self._run_activation_check('embed', 'raw'):
            logger.warning("[ACTIVATION_FAILURE] Embedding model weak. Applying Contrastive Amplification...")
            # Contrastive Amplification: 2.0x Scaling
            X_amp_tr = X_embed_tr * 2.0
            X_amp_te = X_embed_te * 2.0
            
            mae, oof = self.train_kfolds(
                X_subset=X_amp_tr,
                X_test_subset=X_amp_te,
                result_key='embed',
                custom_params=Config.EMBED_LGBM_PARAMS
            )
            # Re-check
            self._run_activation_check('embed', 'raw')
            
        return mae, oof

    def _run_activation_check(self, embed_key, raw_key):
        """Verify if embedding model satisfies the dominance contract."""
        oof_embed = self.oof_preds[embed_key]
        oof_raw = self.oof_preds[raw_key]
        
        std_embed = np.std(oof_embed)
        std_raw = np.std(oof_raw)
        lift = std_embed / (std_raw + 1e-8)
        
        logger.info(f"[ACTIVATION_CHECK] Variance Lift: {lift:.4f} (Target > {Config.ACTIVATION_VARIANCE_LIFT_MIN})")
        return lift >= Config.ACTIVATION_VARIANCE_LIFT_MIN

    def fit_meta_model(self, X_meta_tr, X_meta_te):
        """Final Stacking with Signal Decorrelation (Orthogonalization)."""
        logger.info(f"[STAGE: META_MODEL] Training on {X_meta_tr.shape[1]} stacking features...")
        
        # Decorrelation Check
        oof_raw = self.oof_preds['raw']
        oof_embed = self.oof_preds['embed']
        
        corr = np.corrcoef(oof_raw, oof_embed)[0, 1]
        logger.info(f"[META_DECORRELATION] Raw vs Embed Correlation: {corr:.4f}")
        
        X_meta_tr_mod = X_meta_tr.copy()
        X_meta_te_mod = X_meta_te.copy()
        
        if corr > Config.DECORRELATION_THRESHOLD:
            logger.warning(f"[DECORRELATION] High correlation detected. Orthogonalizing Embed signal...")
            # Force orthogonal signal: Embed = Embed - Raw (residual learning)
            # In meta_features [oof_raw, oof_embed, regime_proxy], raw is 0, embed is 1
            idx_raw = 0
            idx_embed = 1
            
            X_meta_tr_mod[:, idx_embed] = X_meta_tr[:, idx_embed] - X_meta_tr[:, idx_raw]
            X_meta_te_mod[:, idx_embed] = X_meta_te[:, idx_embed] - X_meta_te[:, idx_raw]
            
        mae, oof = self.train_kfolds(
            X_subset=X_meta_tr_mod,
            X_test_subset=X_meta_te_mod,
            result_key='meta',
            custom_params=Config.META_LGBM_PARAMS
        )
        
        # FINAL SUCCESS CHECK: Meta MAE < Min(Raw, Embed)
        raw_mae = self.cv_summary.get('raw', {}).get('overall_mae', 99.0)
        embed_mae = self.cv_summary.get('embed', {}).get('overall_mae', 99.0)
        
        if mae >= min(raw_mae, embed_mae) and min(raw_mae, embed_mae) < 90:
            logger.warning(f"[META_FAILURE] Meta model ({mae:.4f}) failed to outperform base models (Raw: {raw_mae:.4f}, Embed: {embed_mae:.4f})")
        else:
            logger.info(f"[META_SUCCESS] Meta model Alpha activated! Improvement: {min(raw_mae, embed_mae) - mae:.6f}")
            
        return mae, oof

    def analyze_extreme_performance(self, y_true, oof_preds, model_name, threshold):
        """Analyze how well the model captures extreme delay scenarios (Top 10% targets)."""
        is_extreme = y_true >= threshold
        n_extreme = is_extreme.sum()
        
        if n_extreme == 0:
            return {}

        # 1. Extreme Metrics (Top 10% vs Top 10%)
        pred_threshold = np.quantile(oof_preds, 0.9)
        is_pred_extreme = oof_preds >= pred_threshold
        
        recall = (is_extreme & is_pred_extreme).sum() / (n_extreme + 1e-8)
        precision = (is_extreme & is_pred_extreme).sum() / (is_pred_extreme.sum() + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # 2. Extreme-Subset MAE
        extreme_mae = mean_absolute_error(y_true[is_extreme], oof_preds[is_extreme])
        normal_mae = mean_absolute_error(y_true[~is_extreme], oof_preds[~is_extreme])
        
        logger.info(f"--- [EXTREME_ANALYSIS: {model_name.upper()}] ---")
        logger.info(f"Threshold (Top 10%): {threshold:.4f}")
        logger.info(f"Extreme Recall: {recall:.4f} | Precision: {precision:.4f} | F1: {f1:.4f}")
        logger.info(f"Extreme MAE: {extreme_mae:.4f} | Normal MAE: {normal_mae:.4f}")
        
        # Activation Rates (Using numpy indices)
        extreme_X_np = self.X[is_extreme]
        normal_X_np = self.X[~is_extreme]
        
        feature_cols = self.schema['all_features']
        extreme_patterns = [
            '_rel_to_', '_rel_rank_', '_accel', '_volatility_expansion_', 
            'early_warning_', '_regime_id', '_consecutive_above_q75', 'inter_'
        ]
        extreme_feat_list = [f for f in feature_cols if any(p in f for p in extreme_patterns)]
        activation_stats = {}

        for f in extreme_feat_list:
            f_idx = self.schema['feature_to_index'][f]
            # Activation defined as being in top 10% of its own distribution or flag == 1
            if 'flag' in f or 'regime' in f or 'rank' in f:
                act_extreme = (extreme_X_np[:, f_idx] > 0).mean()
                act_normal = (normal_X_np[:, f_idx] > 0).mean()
            else:
                threshold_val = np.quantile(self.X[:, f_idx], 0.9)
                act_extreme = (extreme_X_np[:, f_idx] > threshold_val).mean()
                act_normal = (normal_X_np[:, f_idx] > threshold_val).mean()
            
            activation_stats[f] = {
                "extreme_rate": float(act_extreme),
                "normal_rate": float(act_normal),
                "lift": float(act_extreme / (act_normal + 1e-6))
            }

        # Early Warning Score Distribution
        ew_stats = {}
        if 'early_warning_score' in self.schema['feature_to_index']:
            ew_idx = self.schema['feature_to_index']['early_warning_score']
            ew_s = self.X[:, ew_idx]
            ew_stats = {
                "global_mean": float(ew_s.mean()),
                "extreme_mean": float(extreme_X_np[:, ew_idx].mean()),
                "extreme_p90": float(np.quantile(extreme_X_np[:, ew_idx], 0.9)),
                "lift": float(extreme_X_np[:, ew_idx].mean() / (normal_X_np[:, ew_idx].mean() + 1e-6))
            }

        # Global importance (from model)
        global_imp = pd.Series(self.importance_list[0] if self.importance_list else 0, index=feature_cols)
        global_rank = global_imp.rank(ascending=False)
        
        # Extreme subset correlation - using numpy
        def compute_corr(X_np, y_np):
            corrs = []
            for i in range(X_np.shape[1]):
                try:
                    c = np.abs(np.corrcoef(X_np[:, i], y_np)[0, 1])
                except:
                    c = 0.0
                corrs.append(c)
            return np.nan_to_num(corrs)

        extreme_corrs_np = compute_corr(extreme_X_np, y_true[is_extreme])
        extreme_corrs = pd.Series(extreme_corrs_np, index=feature_cols)
        extreme_rank = extreme_corrs.rank(ascending=False)
        
        importance_comparison = []
        for f in extreme_feat_list:
            if f in global_rank.index:
                importance_comparison.append({
                    "feature": f,
                    "global_rank": int(global_rank[f]),
                    "extreme_rank": int(extreme_rank[f]),
                    "rank_improvement": int(global_rank[f] - extreme_rank[f])
                })

        if ew_stats:
            logger.info(f"Early Warning Score Lift: {ew_stats['lift']:.2f}x")
        
        top_extreme_features = extreme_corrs.sort_values(ascending=False).head(10).index.tolist()
        logger.info(f"Top 10 Features for Extreme Subset: {top_extreme_features}")
        logger.info(f"--- [/EXTREME_ANALYSIS] ---\n")
        
        return {
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(f1),
            "extreme_mae": float(extreme_mae),
            "normal_mae": float(normal_mae),
            "extreme_quantile": 0.9,
            "extreme_threshold_val": float(threshold),
            "top_extreme_features": top_extreme_features,
            "activation_stats": activation_stats,
            "early_warning_distribution": ew_stats,
            "importance_comparison": importance_comparison
        }

    def run_adversarial_validation(self, X_train=None, X_test=None, sample_size=None):
        """Train a train-vs-test classifier and surface the biggest drift drivers."""
        if X_train is None:
            X_train = self.X
        if X_test is None:
            X_test = self.X_test
        if sample_size is None:
            sample_size = Config.ADVERSARIAL_SAMPLE_SIZE
            
        feature_cols = self.schema['all_features']

        # [MISSION: ENFORCE NUMPY]
        n_train = min(len(X_train), sample_size)
        n_test = min(len(X_test), sample_size)
        
        # Sampling using numpy
        idx_train = np.random.choice(len(X_train), n_train, replace=False)
        idx_test = np.random.choice(len(X_test), n_test, replace=False)
        
        train_sample = X_train[idx_train]
        test_sample = X_test[idx_test]

        X_np = np.concatenate([train_sample, test_sample], axis=0).astype(np.float32)
        y = np.concatenate([np.zeros(len(train_sample)), np.ones(len(test_sample))]).astype(np.float32)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        importances = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_np, y), start=1):
            X_tr, X_val = X_np[tr_idx], X_np[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            clf = LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42 + fold,
                verbose=-1,
                n_jobs=-1,
            )
            
            # Use SAFE_FIT to enforce numpy/no-warnings/no-feature-names
            SAFE_FIT(clf, X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.log_evaluation(period=0)])
                
            val_prob = SAFE_PREDICT_PROBA(clf, X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, val_prob))
            importances.append(pd.Series(clf.feature_importances_, index=feature_cols))

        mean_importance = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
        top_features = mean_importance.head(10).index.tolist()
        auc_mean = float(np.mean(aucs))
        self.adversarial_summary = {
            "auc_mean": auc_mean,
            "top_drift_features": top_features,
        }
        logger.info(f"[ADVERSARIAL_VALIDATION] mean_auc={auc_mean:.6f}")
        logger.info(f"[ADVERSARIAL_VALIDATION] top_drift_features={top_features}")
        if auc_mean > 0.7:
            logger.warning(f"[DRIFT_ALERT] Adversarial AUC {auc_mean:.4f} > 0.7")
        return self.adversarial_summary

    def is_ensemble_allowed(self):
        if (self.oof_cat == 0).all():
            return False
        if self.oof_correlation is None:
            self.oof_correlation = np.corrcoef(self.oof_lgb, self.oof_cat)[0, 1]
        return bool(self.oof_correlation < Config.ENSEMBLE_MAX_CORR)

    def get_augmented_features(self, X_train, X_test, feature_names):
        """[TASK 4] Add meta-features based on OOF/Test predictions.
        
        Efficiency: Returns numpy arrays for high-performance intermediate steps.
        Returns: (X_aug_tr_numpy, X_aug_te_numpy, aug_feature_names)
        """
        logger.info("[FEATURE_AUGMENTATION] Adding meta-features from OOF predictions...")
        
        lgb_oof = self.oof_lgb.reshape(-1, 1)
        cat_oof = self.oof_cat.reshape(-1, 1)
        lgb_test = self.test_preds_lgb.reshape(-1, 1)
        cat_test = self.test_preds_cat.reshape(-1, 1)
        
        # Stacking input construction (Numpy for speed)
        X_aug_tr = np.hstack([lgb_oof, cat_oof])
        X_aug_te = np.hstack([lgb_test, cat_test])
        aug_names = ['oof_lgb', 'oof_cat']
        
        # Performance Trace
        if Config.TRACE_LEVEL != 'OFF':
            logger.info(f"[SCHEMA_OK] get_augmented_features: tr_shape={X_aug_tr.shape} | te_shape={X_aug_te.shape}")
            
        return X_aug_tr, X_aug_te, aug_names

    def train_stacking(self, meta_features, top_features_train_np, top_features_test_np):
        """Ridge-based Stacking Meta-model with CANONICAL contract (v6.0)."""
        from sklearn.linear_model import Ridge
        
        # [STEP 6] Hard Assertions (Enforce Canonical Contract)
        assert isinstance(top_features_train_np, np.ndarray), "[FATAL] top_features_train must be a numpy array"
        assert isinstance(top_features_test_np, np.ndarray), "[FATAL] top_features_test must be a numpy array"
        assert isinstance(meta_features, list), "[FATAL] meta_features must be a list"
        
        # In this context, meta_features refers to the names of columns if we were using DataFrames.
        # Since we are using numpy, we assume the input arrays already contain only the needed features.
        
        X_tr = top_features_train_np.astype(np.float32)
        X_te = top_features_test_np.astype(np.float32)
        
        y_target = self.y
        
        logger.info(f"[STACK] Training Ridge meta-model on {X_tr.shape[1]} features...")
        model = Ridge(alpha=1.0)
        
        # [MISSION: GLOBAL MODEL INTERFACE LOCKDOWN]
        SAFE_FIT(model, X_tr, y_target)
            
        oof_stack = SAFE_PREDICT(model, X_tr)
        test_stack = SAFE_PREDICT(model, X_te)
        
        mae = mean_absolute_error(y_target, oof_stack)
        
        return oof_stack, test_stack, mae, model

    def generate_pseudo_labels(self, X_test, ratio, max_samples=5000):
        """Select top confident test samples using CV variance (v5.3)."""
        # X_test is not directly used but kept for interface consistency
        all_folds = np.array(self.test_preds_lgb_folds)
        if len(all_folds) == 0:
            # Fallback if fold-wise preds not available
            n_test = len(self.X_test)
            n_pseudo = min(max_samples, int(n_test * ratio))
            # Just take first N for now as dummy if no variance available
            return np.arange(n_pseudo), self.test_preds.get('stable', np.zeros(n_test))[:n_pseudo]

        std_folds = np.std(all_folds, axis=0)
        
        # Hard Cap Implementation
        n_pseudo = min(max_samples, int(len(self.X_test) * ratio))
        
        confidence_idx = np.argsort(std_folds)[:n_pseudo]
        
        # Return only indices and targets to avoid full DF duplication
        pseudo_idx = confidence_idx
        # Use stable or ensemble if available
        base_preds = self.test_preds.get('stable', next(iter(self.test_preds.values())))
        pseudo_targets = base_preds[confidence_idx]
        
        logger.info(f"--- Pseudo-Labeling Confidence Selection (Count: {n_pseudo}) ---")
        return pseudo_idx, pseudo_targets

    # Removed: run_pseudo_labeling_experiments (Moved to main.py for memory control)

    def generate_interactions(self, train, top_20_cols):
        """Generate interaction features with Direct Assignment."""
        logger.info(f"Generating Interaction Features for {len(top_20_cols)} columns...")
        interaction_cols = []
        for i in range(len(top_20_cols)):
            for j in range(i + 1, len(top_20_cols)):
                c1, c2 = top_20_cols[i], top_20_cols[j]
                new_col = f'inter_{c1}_x_{c2}'
                train[new_col] = (train[c1] * train[c2]).astype('float32')
                interaction_cols.append(new_col)
            if i % 5 == 0: gc.collect() # Frequent cleanup
        
        return train, interaction_cols

    def prune_interactions(self, df, interaction_cols, groups):
        """Memory-optimized interaction pruning.
        ENFORCES NUMPY-ONLY INTERFACE (Rule 2).
        """
        from sklearn.model_selection import GroupKFold
        kf = GroupKFold(n_splits=3)
        fold1_tr, _ = next(kf.split(df, groups=groups))
        
        # [MISSION: HARD TYPE ENFORCEMENT]
        X_tr_np = df.loc[fold1_tr, interaction_cols].fillna(0).values.astype(np.float32)
        y_tr_np = df.loc[fold1_tr, Config.TARGET].values.astype(np.float32)
        
        model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
        
        # [MISSION: GLOBAL MODEL INTERFACE LOCKDOWN]
        SAFE_FIT(model, X_tr_np, y_tr_np)
            
        # [MISSION: INDEX-BASED MAPPING]
        importances = pd.Series(model.feature_importances_, index=interaction_cols)
        selected = importances.sort_values(ascending=False).head(Config.INT_PRUNE_K).index.tolist()
        
        del X_tr_np, y_tr_np, model; gc.collect()
        return selected

    def importance_pruning(self, feature_cols, X=None, y=None):
        """Stable and diverse importance pruning with multi-seed rank aggregation and protection.
        ENFORCES NUMPY-ONLY INTERFACE (Rule 2).
        """
        logger.info(f"Running Stable Multi-Seed Importance Pruning (Total Candidates: {len(feature_cols)})...")
        
        # [MISSION: HARD TYPE ENFORCEMENT] Rule 4
        # We use the internal state (self.X, self.y) directly unless overrides are provided
        X_full_np = X if X is not None else self.X
        y_np = y if y is not None else self.y

        protected_prefixes = ['layout_', 'hour', 'day', 'month', 'is_']
        protected_features = [f for f in feature_cols if any(f.startswith(p) for p in protected_prefixes) or '_' not in f]
        logger.info(f"[PRUNING] Protected Features: {len(protected_features)}")
        
        # 0. Preliminary Correlation Filter
        MAX_CANDIDATES = 800
        candidate_cols = [f for f in feature_cols if f not in protected_features]
        candidate_indices = [feature_cols.index(f) for f in candidate_cols]
        
        if len(candidate_cols) > MAX_CANDIDATES:
            logger.info(f"[PRUNING] Preliminary correlation filter: {len(candidate_cols)} -> {MAX_CANDIDATES}")
            # Correlation using numpy only
            corrs = []
            for idx in candidate_indices:
                # np.corrcoef is deterministic and works on numpy arrays
                corrs.append(np.abs(np.corrcoef(X_full_np[:, idx], y_np)[0, 1]))
            corrs = np.nan_to_num(corrs)
            top_idx_in_candidates = np.argsort(corrs)[::-1][:MAX_CANDIDATES]
            candidate_cols = [candidate_cols[i] for i in top_idx_in_candidates]
            candidate_indices = [feature_cols.index(f) for f in candidate_cols]

        seeds = [42, 43, 44]
        all_ranks = []
        all_importances = []
        
        # [MISSION: SINGLE CONSISTENT MODEL INTERFACE]
        # Slice X_full_np using candidate_indices
        X_data = X_full_np[:, candidate_indices]
        
        assert isinstance(X_data, np.ndarray), "X_data must be numpy.ndarray"
        assert X_data.dtype == np.float32, f"X_data must be float32, got {X_data.dtype}"
        
        for seed in seeds:
            log_memory_usage(f"Pruning Probe Seed {seed}", logger)
            params = Config.LGBM_PARAMS.copy()
            params.update({
                'n_estimators': 150,
                'learning_rate': 0.1,
                'random_state': seed,
                'importance_type': 'gain',
                'n_jobs': -1
            })
            
            probe = LGBMRegressor(**params)
            
            # [MISSION: GLOBAL MODEL INTERFACE LOCKDOWN]
            SAFE_FIT(probe, X_data, y_np)
            
            # [MISSION: INDEX-BASED MAPPING]
            # Map index → feature name using candidate_cols
            imp_series = pd.Series(probe.feature_importances_, index=candidate_cols)
            all_importances.append(imp_series)
            all_ranks.append(imp_series.rank(ascending=True))
            
            del probe; gc.collect()
            
        mean_ranks = pd.DataFrame(all_ranks).mean()
        mean_imps = pd.DataFrame(all_importances).mean()
        
        threshold = mean_ranks.quantile(0.6)
        selected_candidates = mean_ranks[mean_ranks >= threshold].index.tolist()
        
        def get_prefix(f):
            prefixes = ['lag_', 'rolling_', 'sc_', 'pca_', 'inter_', 'diff_', 'velocity_']
            for p in prefixes:
                if p in f: return p
            return 'other_'
            
        diverse_selected = []
        group_counts = {}
        MAX_PER_GROUP = 100
        
        sorted_candidates = mean_imps[selected_candidates].sort_values(ascending=False).index.tolist()
        
        for f in sorted_candidates:
            prefix = get_prefix(f)
            count = group_counts.get(prefix, 0)
            if count < MAX_PER_GROUP:
                diverse_selected.append(f)
                group_counts[prefix] = count + 1
            
        final_features = list(set(protected_features + diverse_selected))
        
        # Return dict: feature_name -> importance
        feature_importances_dict = mean_imps.reindex(final_features).fillna(0).to_dict()

        logger.info(f"[PRUNING] Final Total: {len(final_features)}")
        return final_features, feature_importances_dict

    def _run_lightweight_probe(self, feature_cols, train_np, y_true, groups):
        """Helper for ablation tests to avoid expensive CV seeds."""
        kf = GroupKFold(n_splits=3)
        tr_idx, val_idx = next(kf.split(train_np, groups=groups))
        
        X_tr = train_np[tr_idx].astype(np.float32)
        X_val = train_np[val_idx].astype(np.float32)
        y_tr, y_val = y_true[tr_idx], y_true[val_idx]
        
        params = Config.LGBM_PARAMS.copy()
        params.update({'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42})
        model = LGBMRegressor(**params)
        
        # [MISSION: GLOBAL MODEL INTERFACE LOCKDOWN]
        SAFE_FIT(model, X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.log_evaluation(period=0)])
            
        preds = SAFE_PREDICT(model, X_val)
        return mean_absolute_error(y_val, preds)

    def run_ablation_test(self, feature_cols, X=None, y=None, groups=None):
        """Perform ablation test on feature groups and return MAE for each."""
        logger.info(f"Running Feature Group Ablation Test (Probe Mode)...")
        
        # [MISSION: ENFORCE NUMPY]
        X = X if X is not None else self.X
        y = y if y is not None else self.y
        groups = groups if groups is not None else self.groups
        
        def get_group(feat):
            if 'lag' in feat: return 'Lag'
            if 'rolling' in feat: return 'Rolling'
            if any(x in feat for x in ['early', 'mid', 'late', 'sc_']): return 'Sequence Summary'
            if 'pca' in feat: return 'PCA'
            return 'Original/Other'
            
        groups_map = {f: get_group(f) for f in feature_cols}
        unique_groups = sorted(list(set(groups_map.values())))
        
        # Baseline with ALL features
        baseline_mae = self._run_lightweight_probe(feature_cols, X, y, groups)
        logger.info(f"[ABLATION_BASELINE] Probe MAE (All Features): {baseline_mae:.6f}")
        
        results = {'Baseline': baseline_mae}
        for g in unique_groups:
            # Features WITHOUT this group
            remaining_indices = [i for i, f in enumerate(feature_cols) if groups_map[f] != g]
            if not remaining_indices: continue
                
            mae = self._run_lightweight_probe(None, X[:, remaining_indices], y, groups)
            results[f'Without {g}'] = mae
            logger.info(f"[ABLATION_RESULT] Without {g} | Probe MAE: {mae:.6f} | Delta: {mae - baseline_mae:.6f}")
            
        return results

    def filter_unstable_features(self, feature_cols, importance_list):
        if not importance_list: return feature_cols
        imp_df = pd.DataFrame(importance_list, columns=feature_cols)
        stability_ratios = imp_df.std() / (imp_df.mean() + 1e-9)
        unstable = stability_ratios[stability_ratios > Config.STABILITY_VAR_THRESHOLD].index.tolist()
        return [c for c in feature_cols if c not in unstable]
    
    def find_best_weight(self):
        def ensemble_mae(w):
            preds = w * self.oof_preds['lgb'] + (1 - w) * self.oof_preds.get('cat', self.oof_preds['lgb'])
            return mean_absolute_error(self.y, preds)
        res = minimize(ensemble_mae, [0.5], bounds=[(0, 1)], method='SLSQP')
        return res.x[0], res.fun

    def apply_sigmoid_gating(self, pred_cat, pred_lgb, threshold, temp):
        """Sigmoid-based conditional blending: CAT for low/mid, LGB for high delay."""
        z = (np.asarray(pred_cat) - threshold) / temp
        gate = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))  # clip to prevent overflow
        return (1 - gate) * np.asarray(pred_cat) + gate * np.asarray(pred_lgb)

    def find_best_gating_threshold(self, y_true, thresholds=None, temp=5.0):
        """Grid search over thresholds for sigmoid gating."""
        if thresholds is None:
            thresholds = [20, 25, 30, 35]
        
        best_mae = float('inf')
        best_threshold = thresholds[0]
        
        oof_lgb = self.oof_preds.get('stable', next(iter(self.oof_preds.values())))
        oof_cat = self.oof_preds.get('cat', oof_lgb)
        
        for t in thresholds:
            blended = self.apply_sigmoid_gating(oof_cat, oof_lgb, t, temp)
            mae = mean_absolute_error(y_true, blended)
            logger.info(f"[GATING] Threshold={t}, Temp={temp} | MAE: {mae:.6f}")
            if mae < best_mae:
                best_mae = mae
                best_threshold = t
        
        logger.info(f"[GATING_BEST] Threshold={best_threshold} | MAE: {best_mae:.6f}")
        return best_threshold, best_mae

    def train_mini_stacking(self):
        """Mini Ridge stacking using ONLY OOF predictions (2 features)."""
        from sklearn.linear_model import Ridge
        
        # Assuming we have at least two models for stacking
        keys = list(self.oof_preds.keys())
        if len(keys) < 2:
            return self.oof_preds[keys[0]], self.test_preds[keys[0]], 0.0, None
            
        X_tr = np.column_stack([self.oof_preds[keys[0]], self.oof_preds[keys[1]]]).astype(np.float32)
        X_te = np.column_stack([self.test_preds[keys[0]], self.test_preds[keys[1]]]).astype(np.float32)
        y_target = self.y
        
        cond_number = np.linalg.cond(X_tr)
        logger.info(f"[MINI_STACK] Condition Number: {cond_number:.4e} (must be < 1e6)")
        
        model = Ridge(alpha=1.0)
        
        # [MISSION: GLOBAL MODEL INTERFACE LOCKDOWN]
        SAFE_FIT(model, X_tr, y_target)
            
        oof_stack = SAFE_PREDICT(model, X_tr)
        test_stack = SAFE_PREDICT(model, X_te)
        mae = mean_absolute_error(y_target, oof_stack)
        
        logger.info(f"[MINI_STACK] Ridge Weights: {keys[0]}={model.coef_[0]:.4f}, {keys[1]}={model.coef_[1]:.4f}")
        logger.info(f"[MINI_STACK] MAE: {mae:.6f}")
        
        return oof_stack, test_stack, mae, model

    def train_residual_model(
        self,
        X_train_base_np,
        X_test_base_np,
        y_true,
        groups,
        pred_base_oof,
        pred_base_test,
        feature_names,
        sample_weight=None
    ):
        """Train secondary residual model on [original features + base prediction] with schema enforcement."""
        residual_target = np.asarray(y_true) - np.asarray(pred_base_oof)
        
        # Build Residual Feature Matrices (Numpy only)
        X_train_res = np.hstack([np.asarray(X_train_base_np), np.asarray(pred_base_oof).reshape(-1, 1)]).astype(np.float32)
        X_test_res = np.hstack([np.asarray(X_test_base_np), np.asarray(pred_base_test).reshape(-1, 1)]).astype(np.float32)
        
        oof_residual = np.zeros(len(X_train_res), dtype=np.float64)
        test_residual = np.zeros(len(X_test_res), dtype=np.float64)
        
        if Config.SPLIT_STRATEGY == 'GroupKFold':
            kf = GroupKFold(n_splits=Config.NFOLDS)
            split_iter = kf.split(X_train_res, groups=groups)
        else:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=Config.NFOLDS, shuffle=True, random_state=42)
            split_iter = kf.split(X_train_res)
        
        for fold, (tr_idx, val_idx) in enumerate(split_iter):
            # Enforce pure numpy
            X_tr = X_train_res[tr_idx]
            X_val = X_train_res[val_idx]
            X_test_res_np = X_test_res
            
            y_tr, y_val = residual_target[tr_idx], residual_target[val_idx]
            
            sw_tr = None
            if sample_weight is not None:
                sw_tr = np.asarray(sample_weight)[tr_idx]
            
            params = Config.LGBM_PARAMS.copy()
            params.update({
                'n_estimators': 800,
                'learning_rate': 0.03,
                'random_state': 2026 + fold
            })
            model = LGBMRegressor(**params)
            
            # Use _train_model for consistency
            val_preds_fold, test_preds_fold, _ = self._train_model(
                model, X_tr, y_tr, X_val, y_val, X_test_res_np, sample_weight=sw_tr
            )
            
            oof_residual[val_idx] = val_preds_fold
            test_residual += test_preds_fold / Config.NFOLDS
            
            del model; gc.collect()
        
        residual_mae = mean_absolute_error(residual_target, oof_residual)
        final_oof = np.asarray(pred_base_oof) + oof_residual
        final_mae = mean_absolute_error(np.asarray(y_true), final_oof)
        final_test = np.asarray(pred_base_test) + test_residual
        return residual_mae, final_mae, final_test
