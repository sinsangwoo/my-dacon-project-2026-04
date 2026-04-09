import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
import gc
import logging
from .utils import log_memory_usage
from .config import Config

logger = logging.getLogger(__name__)

class Trainer:
    """Class to handle Advanced Cross-Validation, Stacking, and Pseudo Labeling."""
    def __init__(self, train, test, feature_cols, groups):
        # Handle both DataFrame and Numpy arrays for memory efficiency
        self.train = train.reset_index(drop=True) if isinstance(train, pd.DataFrame) else train
        self.test = test.reset_index(drop=True) if isinstance(test, pd.DataFrame) else test
        self.feature_cols = feature_cols
        self.groups = groups if groups is not None else np.array([0] * len(train))
        
        # Split Strategy
        if Config.SPLIT_STRATEGY == 'GroupKFold':
            from sklearn.model_selection import GroupKFold
            self.kf = GroupKFold(n_splits=Config.NFOLDS)
            logger.info(f"[VALIDATION] GroupKFold active. Group column: scenario_id")
            logger.info(f"[VALIDATION] Unique groups: {len(np.unique(self.groups))}")
        else:
            from sklearn.model_selection import KFold
            self.kf = KFold(n_splits=Config.NFOLDS, shuffle=True, random_state=42)
        
        # Performance Tracking
        self.oof_lgb = np.zeros(len(train))
        self.oof_cat = np.zeros(len(train))
        self.test_preds_lgb = np.zeros(len(test))
        self.test_preds_cat = np.zeros(len(test))
        
        # Fold-wise predictions for test (to calculate variance for pseudo confidence)
        self.test_preds_lgb_folds = []
        self.test_preds_cat_folds = []
        
        # Stability Monitoring
        self.importance_list = []
        
        # Diagnostics
        self.oof_correlation = None
        self.fold_maes_lgb = []
        self.fold_maes_cat = []

    def _mean_encoding_cv(self, tr_df, val_df, test_df, target_col):
        """Perform CV-safe mean encoding for layout_id."""
        if 'layout_id' not in tr_df.columns:
            idx = test_df.index if isinstance(test_df, pd.DataFrame) else range(len(test_df))
            return tr_df, val_df, pd.Series([0] * len(test_df), index=idx)
            
        tr_df, val_df, test_df = tr_df.copy(), val_df.copy(), test_df.copy()
        
        means = tr_df.groupby('layout_id')[target_col].mean()
        global_mean = tr_df[target_col].mean()
        tr_df['layout_id_mean_target'] = tr_df['layout_id'].map(means).fillna(global_mean)
        val_df['layout_id_mean_target'] = val_df['layout_id'].map(means).fillna(global_mean)
        
        if isinstance(test_df, pd.DataFrame):
            test_enc = test_df['layout_id'].map(means).fillna(global_mean)
        else:
            # If test is numpy, mean encoding logic for layout_id must have been handled pre-conversion
            test_enc = pd.Series([global_mean] * len(test_df))
            
        return tr_df, val_df, test_enc

    def validate_data(self, X, y, groups):
        """Perform strict data integrity checks."""
        logger.info("[STEP: validation] Running Data Integrity Checks...")
        
        # 1. Length Match
        if not (len(X) == len(y) == len(groups)):
            raise ValueError(f"Length mismatch! X: {len(X)}, y: {len(y)}, groups: {len(groups)}")
            
        # 2. NaN Check
        if isinstance(X, pd.DataFrame):
            nans = X.isnull().sum().sum()
        else:
            nans = np.isnan(X).sum()
            
        if nans > 0:
            raise ValueError(f"Dataset contains {nans} NaN values!")
        if np.isnan(y).any():
            raise ValueError("Target contains NaN values!")
            
        # 3. Shape/Dtype Log (Ensure no Object Dtype)
        if isinstance(X, pd.DataFrame):
            if (X.dtypes == object).any():
                raise TypeError(f"Dataset contains object types: {X.select_dtypes('object').columns.tolist()}")

        logger.info(f"Integrity Check Passed. Shape: {X.shape}, Target: {y.shape}")
            
    def check_group_leakage(self, train_idx, val_idx, groups):
        """Ensure no group overlap between splits."""
        tr_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        overlap = tr_groups.intersection(val_groups)
        if overlap:
            raise ValueError(f"Group Leakage Detected! Overlapping IDs: {list(overlap)[:5]}...")

    def _train_model(self, model_type, X_tr, y_tr, X_val, y_val, test_X, seed, sample_weight_tr=None):
        """Encapsulate model training logic with optional sample weighting."""
        if model_type == 'lgb':
            params = Config.LGBM_PARAMS.copy()
            params['random_state'] = seed
            model = LGBMRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      sample_weight=sample_weight_tr,
                      callbacks=[lgb.early_stopping(Config.EARLY_STOPPING_ROUNDS), 
                                 lgb.log_evaluation(Config.LOG_EVALUATION_STEPS)])
        else: # CatBoost
            params = Config.CAT_PARAMS.copy()
            params['random_state'] = seed
            model = CatBoostRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                      sample_weight=sample_weight_tr,
                      use_best_model=True, verbose=False)
            
        # Hard Prediction Guard (v5.0)
        n_features = X_tr.shape[1]
        assert test_X.shape[1] == n_features, f"Prediction Guard: Feature count mismatch! Train: {n_features}, Test: {test_X.shape[1]}"
        
        val_preds = model.predict(X_val)
        test_preds = model.predict(test_X)
        return val_preds, test_preds, model.feature_importances_ if model_type=='lgb' else None

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
            if corr > 0.9:
                logger.warning(f"[CORRELATION_WARNING] High correlation ({corr:.4f}). Stacking benefit may be limited.")
        
        logger.info(f"--- [/OOF_DIAGNOSTICS] ---\n")

    def train_kfolds(self, feature_cols, train_df=None, seeds=None, sample_weight=None):
        """Core CV loop with Seed Averaging, Sample Weighting, and CV Diagnostics."""
        if train_df is None: train_df = self.train
        if seeds is None: seeds = Config.SEEDS
        
        oof_lgb_final = np.zeros(len(train_df))
        oof_cat_final = np.zeros(len(train_df))
        test_preds_lgb_final = np.zeros(len(self.test))
        test_preds_cat_final = np.zeros(len(self.test))
        
        self.test_preds_lgb_folds = []
        self.test_preds_cat_folds = []
        self.fold_maes_lgb = []
        self.fold_maes_cat = []
        
        for seed in seeds:
            logger.info(f"── Training with Seed: {seed} ──")
            logger.info(f"Seed {seed} Start")
            
            oof_lgb_seed = np.zeros(len(train_df))
            oof_cat_seed = np.zeros(len(train_df))
            
            # Memory Guard: Calculate split indices once per seed
            if Config.SPLIT_STRATEGY != 'GroupKFold':
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=Config.NFOLDS, shuffle=True, random_state=seed)
            else:
                kf = self.kf
                
            groups = train_df['scenario_id'].values if isinstance(train_df, pd.DataFrame) and 'scenario_id' in train_df.columns else self.groups
            
            # --- MEMORY_OPTIMIZATION: Pre-extract target to avoid repeated slicing ---
            y_all = train_df[Config.TARGET].values if isinstance(train_df, pd.DataFrame) else self.train[Config.TARGET].values if isinstance(self.train, pd.DataFrame) else self.train[:, -1]
            
            for fold, (tr_idx, val_idx) in enumerate(kf.split(train_df, groups=groups)):
                # === VALIDATION: GroupKFold Leakage Check ===
                self.check_group_leakage(tr_idx, val_idx, groups)
                
                # CV Stability Check
                tr_grps = len(np.unique(groups[tr_idx]))
                val_grps = len(np.unique(groups[val_idx]))
                logger.info(f"[STEP: cv_split] Fold {fold+1}: Train Groups={tr_grps}, Valid Groups={val_grps}")
                if val_grps < 2:
                    logger.warning(f"!!! WARNING: Fold {fold+1} has very few validation groups ({val_grps}) !!!")
                
                # FOLD_START SNAPSHOT
                logger.info(f"\n[FOLD_START] Fold {fold+1} | Seed {seed}")
                
                # --- MEMORY_OPTIMIZATION: Direct Slicing (Avoid df_tr/df_val if possible) ---
                if isinstance(train_df, pd.DataFrame) and 'layout_id' in train_df.columns:
                    df_tr = train_df.iloc[tr_idx].reset_index(drop=True)
                    df_val = train_df.iloc[val_idx].reset_index(drop=True)
                    df_tr, df_val, test_fold_enc = self._mean_encoding_cv(df_tr, df_val, self.test, Config.TARGET)
                    fold_features = feature_cols + ['layout_id_mean_target']
                    X_tr, y_tr = df_tr[fold_features].values, df_tr[Config.TARGET].values
                    X_val, y_val = df_val[fold_features].values, df_val[Config.TARGET].values
                    
                    if isinstance(self.test, pd.DataFrame):
                        test_X = self.test[feature_cols].copy()
                        test_X['layout_id_mean_target'] = test_fold_enc
                        test_X = test_X[fold_features].values
                    else:
                        test_X = self.test # Case handled elsewhere or simplified
                    del df_tr, df_val; gc.collect()
                else:
                    # Raw Numpy Path (Super Memory Efficient)
                    f_idx = len(feature_cols)
                    # Force all inputs to numpy for absolute indexing safety
                    X_np = train_df.values if isinstance(train_df, pd.DataFrame) else train_df
                    y_np = y_all if not isinstance(train_df, pd.DataFrame) else y_all # y_all is already numpy
                    
                    X_tr, X_val = X_np[tr_idx, :f_idx], X_np[val_idx, :f_idx]
                    y_tr, y_val = y_np[tr_idx], y_np[val_idx]
                    
                    test_X = self.test[feature_cols].values if isinstance(self.test, pd.DataFrame) else self.test
                    if not isinstance(self.test, pd.DataFrame) and test_X.shape[1] > f_idx:
                        test_X = test_X[:, :f_idx]
                
                X_tr, y_tr = np.asanyarray(X_tr), np.asanyarray(y_tr)
                X_val, y_val = np.asanyarray(X_val), np.asanyarray(y_val)
                test_X = np.asanyarray(test_X)
                
                # Sample Weight Slicing
                sw_tr = None
                if sample_weight is not None:
                    sw_tr = np.asanyarray(sample_weight)[tr_idx]
                
                # v5.2 PRE-FIT FORENSICS
                logger.info(f"[FOLD_SNAPSHOT] X_tr: {X_tr.shape} | X_val: {X_val.shape}")
                # assert not np.isnan(X_tr).any(), f"Fold {fold+1} X_tr contains NaNs!"
                
                # 1. LightGBM
                vp, tp, imp = self._train_model('lgb', X_tr, y_tr, X_val, y_val, test_X, seed, sample_weight_tr=sw_tr)
                logger.info(f"[PRED_TRACE] LGBM | Val: {vp.shape} | Test: {tp.shape}")
                
                # Fold-level MAE
                fold_mae_lgb = mean_absolute_error(y_val, vp)
                self.fold_maes_lgb.append(fold_mae_lgb)
                logger.info(f"[CV_FOLD] Fold {fold+1} | Seed {seed} | LGB MAE: {fold_mae_lgb:.6f}")
                
                oof_lgb_seed[val_idx] = vp
                test_preds_lgb_final += tp / (len(seeds) * Config.NFOLDS)
                self.test_preds_lgb_folds.append(tp)
                if seed == seeds[0]: self.importance_list.append(imp)
                
                # 2. CatBoost (Skip if DEBUG_PHASE3)
                if not Config.DEBUG_PHASE3:
                    vp_cat, tp_cat, _ = self._train_model('cat', X_tr, y_tr, X_val, y_val, test_X, seed, sample_weight_tr=sw_tr)
                    logger.info(f"[PRED_TRACE] CatBoost | Val: {vp_cat.shape} | Test: {tp_cat.shape}")
                    
                    # Fold-level MAE
                    fold_mae_cat = mean_absolute_error(y_val, vp_cat)
                    self.fold_maes_cat.append(fold_mae_cat)
                    logger.info(f"[CV_FOLD] Fold {fold+1} | Seed {seed} | CAT MAE: {fold_mae_cat:.6f}")
                    
                    oof_cat_seed[val_idx] = vp_cat
                    test_preds_cat_final += tp_cat / (len(seeds) * Config.NFOLDS)
                    self.test_preds_cat_folds.append(tp_cat)
                
                logger.info(f"[FOLD_END] Fold {fold+1} Finished.\n")
                # Mandatory Loop Cleanup
                del X_tr, X_val, y_tr, y_val, test_X; gc.collect()

            oof_lgb_final += oof_lgb_seed / len(seeds)
            if not Config.DEBUG_PHASE3:
                oof_cat_final += oof_cat_seed / len(seeds)
            
            logger.info(f"Seed {seed} End")
            
        self.oof_lgb, self.oof_cat = oof_lgb_final, oof_cat_final
        self.test_preds_lgb, self.test_preds_cat = test_preds_lgb_final, test_preds_cat_final
        
        lgb_mae = mean_absolute_error(y_all, oof_lgb_final)
        cat_mae = mean_absolute_error(y_all, oof_cat_final) if not Config.DEBUG_PHASE3 else 0.0
        
        logger.info(f"LGBM MAE: {lgb_mae:.6f}")
        
        # === CV SUMMARY ===
        if self.fold_maes_lgb:
            lgb_mean = np.mean(self.fold_maes_lgb)
            lgb_std = np.std(self.fold_maes_lgb)
            logger.info(f"[CV_SUMMARY] LGB Fold MAEs: mean={lgb_mean:.6f}, std={lgb_std:.6f}")
            logger.info(f"[CV_SUMMARY] LGB Fold MAEs detail: {[f'{m:.6f}' for m in self.fold_maes_lgb]}")
            if lgb_std > 0.2 * lgb_mean:
                logger.warning(f"[CV_WARNING] High LGB fold variance detected! std/mean = {lgb_std/lgb_mean:.3f}")
        
        if self.fold_maes_cat and not Config.DEBUG_PHASE3:
            cat_mean = np.mean(self.fold_maes_cat)
            cat_std = np.std(self.fold_maes_cat)
            logger.info(f"[CV_SUMMARY] CAT Fold MAEs: mean={cat_mean:.6f}, std={cat_std:.6f}")
            logger.info(f"[CV_SUMMARY] CAT Fold MAEs detail: {[f'{m:.6f}' for m in self.fold_maes_cat]}")
            if cat_std > 0.2 * cat_mean:
                logger.warning(f"[CV_WARNING] High CAT fold variance detected! std/mean = {cat_std/cat_mean:.3f}")
        
        # === OOF DIAGNOSTICS ===
        self._log_oof_diagnostics(y_all, oof_lgb_final, oof_cat_final)
        
        return lgb_mae, cat_mae

    def train_stacking(self, feature_cols, top_features_train=None, top_features_test=None):
        """Advanced meta-model with OOF predictions + top features + dual model experiment."""
        logger.info(f"Training Advanced Stacking Meta-model...")
        
        # 1. Build base meta features (4 OOF-derived)
        meta_train = pd.DataFrame({
            'oof_lgb': self.oof_lgb,
            'oof_cat': self.oof_cat,
            'oof_diff': np.abs(self.oof_lgb - self.oof_cat),
            'oof_mean': (self.oof_lgb + self.oof_cat) / 2.0
        })
        meta_test = pd.DataFrame({
            'oof_lgb': self.test_preds_lgb,
            'oof_cat': self.test_preds_cat,
            'oof_diff': np.abs(self.test_preds_lgb - self.test_preds_cat),
            'oof_mean': (self.test_preds_lgb + self.test_preds_cat) / 2.0
        })
        
        # 2. Add top features if provided (Advanced Stacking)
        if top_features_train is not None and top_features_test is not None:
            meta_train = pd.concat([meta_train, top_features_train.reset_index(drop=True)], axis=1)
            meta_test = pd.concat([meta_test, top_features_test.reset_index(drop=True)], axis=1)
            logger.info(f"[STACKING] Added {top_features_train.shape[1]} top features to meta-model")
        
        logger.info(f"[STACKING] Total meta features: {meta_train.shape[1]}")
        logger.info(f"[STACKING] Meta feature names: {list(meta_train.columns)[:10]}...")
        
        # 3. Get target
        if isinstance(self.train, pd.DataFrame):
            y_target = self.train[Config.TARGET].values
        else:
            raise ValueError("train_stacking requires DataFrame with target column")
        
        # 4. Feature Selection via LGBM probe (for high-dimensional meta)
        selected_features = list(meta_train.columns)
        if meta_train.shape[1] > 10:
            logger.info(f"[STACKING] Running feature selection on {meta_train.shape[1]} meta features...")
            probe = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
            probe.fit(meta_train.fillna(0), y_target)
            importances = pd.Series(probe.feature_importances_, index=meta_train.columns).sort_values(ascending=False)
            cum_imp = importances.cumsum() / importances.sum()
            selected_features = cum_imp[cum_imp <= 0.95].index.tolist()
            # Ensure at least 4 features (the OOF ones)
            if len(selected_features) < 4:
                selected_features = importances.head(max(4, meta_train.shape[1] // 2)).index.tolist()
            
            meta_train = meta_train[selected_features]
            meta_test = meta_test[selected_features]
            logger.info(f"[STACKING] After feature selection: {len(selected_features)} features retained")
            logger.info(f"[STACKING] Selected: {selected_features[:10]}")
        
        # 5. Base MAE (simple blending for comparison)
        base_preds = (self.oof_lgb + self.oof_cat) / 2.0
        base_mae = mean_absolute_error(y_target, base_preds)
        logger.info(f"[STACKING] Base Blending MAE: {base_mae:.6f}")
        
        # 6. Dual Meta Model Experiment (Ridge + LGBM)
        from sklearn.linear_model import Ridge
        
        results = {}
        kf = self.kf
        
        for model_name in ['ridge', 'lgbm']:
            oof_stack = np.zeros(len(self.train))
            test_preds_stack = np.zeros(len(self.test))
            
            for fold, (tr_idx, val_idx) in enumerate(kf.split(self.train, groups=self.groups)):
                X_tr = meta_train.iloc[tr_idx]
                X_val = meta_train.iloc[val_idx]
                y_tr = y_target[tr_idx]
                
                if model_name == 'ridge':
                    model = Ridge(alpha=1.0)
                    model.fit(X_tr, y_tr)
                else:
                    model = LGBMRegressor(
                        n_estimators=200, learning_rate=0.05, max_depth=3,
                        num_leaves=15, subsample=0.8, colsample_bytree=0.8,
                        verbose=-1, n_jobs=-1, random_state=42
                    )
                    model.fit(X_tr, y_tr)
                
                oof_stack[val_idx] = model.predict(X_val)
                test_preds_stack += model.predict(meta_test) / Config.NFOLDS
            
            stack_mae = mean_absolute_error(y_target, oof_stack)
            results[model_name] = (oof_stack.copy(), test_preds_stack.copy(), stack_mae)
            logger.info(f"[STACKING] {model_name.upper()} Meta MAE: {stack_mae:.6f}")
        
        # 7. Select best meta-model
        best_model = min(results, key=lambda k: results[k][2])
        best_oof, best_test, best_mae = results[best_model]
        logger.info(f"[STACKING] Best meta-model: {best_model.upper()} (MAE: {best_mae:.6f})")
        
        # 8. Auto-skip if worse than base
        if best_mae >= base_mae:
            logger.warning(f"[STACKING_SKIP] Stacking MAE ({best_mae:.6f}) >= Base MAE ({base_mae:.6f}). Using base blending.")
            base_test = (self.test_preds_lgb + self.test_preds_cat) / 2.0
            return base_preds, base_test, base_mae, True
        
        improvement = base_mae - best_mae
        logger.info(f"[STACKING] ✓ Improvement over base: {improvement:.6f} ({improvement/base_mae*100:.2f}%)")
        return best_oof, best_test, best_mae, False

    def generate_pseudo_labels(self, test_df, ratio, max_samples=5000):
        """Select top confident test samples using CV variance (v5.3)."""
        all_folds = np.array(self.test_preds_lgb_folds)
        std_folds = np.std(all_folds, axis=0)
        
        # Hard Cap Implementation
        n_pseudo = min(max_samples, int(len(test_df) * ratio))
        
        confidence_idx = np.argsort(std_folds)[:n_pseudo]
        
        # Return only indices and targets to avoid full DF duplication
        pseudo_idx = confidence_idx
        pseudo_targets = (self.test_preds_lgb[confidence_idx] + self.test_preds_cat[confidence_idx]) / 2.0
        
        logger.info(f"--- Pseudo-Labeling Confidence Selection (Count: {n_pseudo}) ---")
        return pseudo_idx, pseudo_targets

    # Removed: run_pseudo_labeling_experiments (Moved to main.py for memory control)

    def generate_interactions(self, train, top_20_cols):
        """Generate interaction features."""
        df = train.copy()
        new_interactions = {}
        interaction_cols = []
        for i in range(len(top_20_cols)):
            for j in range(i + 1, len(top_20_cols)):
                c1, c2 = top_20_cols[i], top_20_cols[j]
                new_col = f'inter_{c1}_x_{c2}'
                new_interactions[new_col] = df[c1] * df[c2]
                interaction_cols.append(new_col)
        
        df = pd.concat([df, pd.DataFrame(new_interactions)], axis=1)
        return df, interaction_cols

    def prune_interactions(self, df, interaction_cols, groups):
        from sklearn.model_selection import GroupKFold
        kf = GroupKFold(n_splits=3)
        fold1_tr, _ = next(kf.split(df, groups=groups))
        X_tr = df.loc[fold1_tr, interaction_cols]
        y_tr = df.loc[fold1_tr, Config.TARGET]
        model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        model.fit(X_tr, y_tr)
        importances = pd.Series(model.feature_importances_, index=interaction_cols)
        return importances.sort_values(ascending=False).head(Config.INT_PRUNE_K).index.tolist()

    def importance_pruning(self, train_df, feature_cols):
        """Identify features contributing to 95% of cumulative gain. Returns (features, importances)."""
        logger.info(f"Running Probe LGBM for Importance Pruning...")
        params = Config.LGBM_PARAMS.copy()
        params.update({'n_estimators': 200, 'learning_rate': 0.1, 'n_jobs': -1})
        probe_model = LGBMRegressor(**params)
        probe_model.fit(train_df[feature_cols], train_df[Config.TARGET])
        importances = pd.Series(probe_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        cum_importance = importances.cumsum() / importances.sum()
        selected_features = cum_importance[cum_importance <= Config.IMPORTANCE_THRESHOLD].index.tolist()
        if not selected_features:
            selected_features = importances.head(100).index.tolist()
        return selected_features, importances

    def filter_unstable_features(self, feature_cols, importance_list):
        if not importance_list: return feature_cols
        imp_df = pd.DataFrame(importance_list, columns=feature_cols)
        stability_ratios = imp_df.std() / (imp_df.mean() + 1e-9)
        unstable = stability_ratios[stability_ratios > Config.STABILITY_VAR_THRESHOLD].index.tolist()
        return [c for c in feature_cols if c not in unstable]
    
    def find_best_weight(self):
        def ensemble_mae(w):
            preds = w * self.oof_lgb + (1 - w) * self.oof_cat
            return mean_absolute_error(self.train[Config.TARGET], preds)
        res = minimize(ensemble_mae, [0.5], bounds=[(0, 1)], method='SLSQP')
        return res.x[0], res.fun
