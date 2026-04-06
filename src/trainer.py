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
        self.train = train.reset_index(drop=True)
        self.test = test.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.groups = groups if groups is not None else np.array([0] * len(train))
        
        # Split Strategy
        if Config.SPLIT_STRATEGY == 'GroupKFold':
            from sklearn.model_selection import GroupKFold
            self.kf = GroupKFold(n_splits=Config.NFOLDS)
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

    def _mean_encoding_cv(self, tr_df, val_df, test_df, target_col):
        """Perform CV-safe mean encoding for layout_id."""
        if 'layout_id' not in tr_df.columns:
            return tr_df, val_df, pd.Series([0] * len(test_df), index=test_df.index)
            
        tr_df = tr_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        
        means = tr_df.groupby('layout_id')[target_col].mean()
        global_mean = tr_df[target_col].mean()
        tr_df['layout_id_mean_target'] = tr_df['layout_id'].map(means).fillna(global_mean)
        val_df['layout_id_mean_target'] = val_df['layout_id'].map(means).fillna(global_mean)
        test_enc = test_df['layout_id'].map(means).fillna(global_mean)
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

    def _train_model(self, model_type, X_tr, y_tr, X_val, y_val, test_X, seed):
        """Encapsulate model training logic."""
        if model_type == 'lgb':
            params = Config.LGBM_PARAMS.copy()
            params['random_state'] = seed
            model = LGBMRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                      callbacks=[lgb.early_stopping(Config.EARLY_STOPPING_ROUNDS), 
                                 lgb.log_evaluation(Config.LOG_EVALUATION_STEPS)])
        else: # CatBoost
            params = Config.CAT_PARAMS.copy()
            params['random_state'] = seed
            model = CatBoostRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
            
        # Hard Prediction Guard (v5.0)
        n_features = X_tr.shape[1]
        assert test_X.shape[1] == n_features, f"Prediction Guard: Feature count mismatch! Train: {n_features}, Test: {test_X.shape[1]}"
        
        val_preds = model.predict(X_val)
        test_preds = model.predict(test_X)
        return val_preds, test_preds, model.feature_importances_ if model_type=='lgb' else None

    def train_kfolds(self, feature_cols, train_df=None, seeds=None):
        """Core CV loop with Seed Averaging support."""
        if train_df is None: train_df = self.train
        if seeds is None: seeds = Config.SEEDS
        
        oof_lgb_final = np.zeros(len(train_df))
        oof_cat_final = np.zeros(len(train_df))
        test_preds_lgb_final = np.zeros(len(self.test))
        test_preds_cat_final = np.zeros(len(self.test))
        
        self.test_preds_lgb_folds = []
        self.test_preds_cat_folds = []
        
        for seed in seeds:
            logger.info(f"── Training with Seed: {seed} ──")
            log_memory_usage(f"Seed {seed} Start")
            
            oof_lgb_seed = np.zeros(len(train_df))
            oof_cat_seed = np.zeros(len(train_df))
            
            if Config.SPLIT_STRATEGY != 'GroupKFold':
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=Config.NFOLDS, shuffle=True, random_state=seed)
            else:
                kf = self.kf
                
            groups = train_df['scenario_id'].values if 'scenario_id' in train_df.columns else self.groups
            
            for fold, (tr_idx, val_idx) in enumerate(kf.split(train_df, groups=groups)):
                # CV Stability Check
                tr_grps = len(np.unique(groups[tr_idx]))
                val_grps = len(np.unique(groups[val_idx]))
                logger.info(f"[STEP: cv_split] Fold {fold+1}: Train Groups={tr_grps}, Valid Groups={val_grps}")
                if val_grps < 2:
                    logger.warning(f"!!! WARNING: Fold {fold+1} has very few validation groups ({val_grps}) !!!")
                
                self.check_group_leakage(tr_idx, val_idx, groups)
                
                df_tr = train_df.iloc[tr_idx].reset_index(drop=True)
                df_val = train_df.iloc[val_idx].reset_index(drop=True)
                
                # CV-safe mean encoding (adds layout_id_mean_target if applicable)
                df_tr, df_val, test_fold_enc = self._mean_encoding_cv(df_tr, df_val, self.test, Config.TARGET)
                fold_features = feature_cols + (['layout_id_mean_target'] if 'layout_id' in train_df.columns else [])
                
                # v5.1 PURE CONSUMER: Direct column access, NO alignment
                X_tr, y_tr = df_tr[fold_features], df_tr[Config.TARGET]
                X_val, y_val = df_val[fold_features], df_val[Config.TARGET]
                
                # Build test_X: direct select from pre-canonicalized test data
                test_X = self.test[feature_cols].copy()
                if 'layout_id_mean_target' in fold_features:
                    test_X['layout_id_mean_target'] = test_fold_enc
                test_X = test_X[fold_features]
                
                # v5.1 ASSERT-ONLY: shapes must match (Phase 2 guarantees this)
                assert test_X.shape[1] == X_tr.shape[1], \
                    f"HARD FAIL Fold {fold+1}: Train features={X_tr.shape[1]}, Test features={test_X.shape[1]}"
                
                # 1. LightGBM
                vp, tp, imp = self._train_model('lgb', X_tr, y_tr, X_val, y_val, test_X, seed)
                oof_lgb_seed[val_idx] = vp
                test_preds_lgb_final += tp / (len(seeds) * Config.NFOLDS)
                self.test_preds_lgb_folds.append(tp)
                if seed == seeds[0]: self.importance_list.append(imp)
                
                # 2. CatBoost (Skip if DEBUG_PHASE3)
                if not Config.DEBUG_PHASE3:
                    vp_cat, tp_cat, _ = self._train_model('cat', X_tr, y_tr, X_val, y_val, test_X, seed)
                    oof_cat_seed[val_idx] = vp_cat
                    test_preds_cat_final += tp_cat / (len(seeds) * Config.NFOLDS)
                    self.test_preds_cat_folds.append(tp_cat)
                
                # Mandatory Loop Cleanup
                del df_tr, df_val, X_tr, X_val, y_tr, y_val, test_X; gc.collect()

            oof_lgb_final += oof_lgb_seed / len(seeds)
            if not Config.DEBUG_PHASE3:
                oof_cat_final += oof_cat_seed / len(seeds)
            
            log_memory_usage(f"Seed {seed} End")
            
        self.oof_lgb, self.oof_cat = oof_lgb_final, oof_cat_final
        self.test_preds_lgb, self.test_preds_cat = test_preds_lgb_final, test_preds_cat_final
        
        lgb_mae = mean_absolute_error(train_df[Config.TARGET], oof_lgb_final)
        cat_mae = mean_absolute_error(train_df[Config.TARGET], oof_cat_final) if not Config.DEBUG_PHASE3 else 0.0
        
        logger.info(f"LGBM MAE: {lgb_mae:.6f}")
        return lgb_mae, cat_mae

    def train_stacking(self, feature_cols):
        """Train meta-model (Ridge/LGBM) on OOF predictions."""
        logger.info(f"Training Stacking Meta-model ({Config.META_MODEL})...")
        meta_train = pd.DataFrame({'lgb': self.oof_lgb, 'cat': self.oof_cat})
        meta_test = pd.DataFrame({'lgb': self.test_preds_lgb, 'cat': self.test_preds_cat})
        
        from sklearn.linear_model import Ridge
        oof_stack = np.zeros(len(self.train))
        test_preds_stack = np.zeros(len(self.test))
        
        kf = self.kf
        for fold, (tr_idx, val_idx) in enumerate(kf.split(self.train, groups=self.groups)):
            X_tr, y_tr = meta_train.iloc[tr_idx], self.train.loc[tr_idx, Config.TARGET]
            X_val, y_val = meta_train.iloc[val_idx], self.train.loc[val_idx, Config.TARGET]
            
            if Config.META_MODEL == 'ridge':
                model = Ridge(alpha=1.0)
                model.fit(X_tr, y_tr)
            else:
                model = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, verbose=-1)
                model.fit(X_tr, y_tr)
                
            oof_stack[val_idx] = model.predict(X_val)
            test_preds_stack += model.predict(meta_test) / Config.NFOLDS
            
        stack_mae = mean_absolute_error(self.train[Config.TARGET], oof_stack)
        logger.info(f"Stacking MAE: {stack_mae:.4f}")
        return oof_stack, test_preds_stack, stack_mae

    def generate_pseudo_labels(self, test_df, ratio):
        """Select top confident test samples using CV variance."""
        all_folds = np.array(self.test_preds_lgb_folds)
        std_folds = np.std(all_folds, axis=0)
        
        n_pseudo = int(len(test_df) * ratio)
        max_allowed = int(len(self.train) * Config.MAX_PSEUDO_RATIO)
        n_pseudo = min(n_pseudo, max_allowed)
        
        confidence_idx = np.argsort(std_folds)[:n_pseudo]
        
        pseudo_df = test_df.iloc[confidence_idx].copy()
        pseudo_df[Config.TARGET] = (self.test_preds_lgb[confidence_idx] + self.test_preds_cat[confidence_idx]) / 2.0
        
        logger.info(f"--- Pseudo-Labeling Confidence Selection (Ratio: {ratio*100}%) ---")
        return pseudo_df

    def run_pseudo_labeling_experiments(self, feature_cols, test_df):
        """Experiment with pseudo-label ratios."""
        best_mae = float('inf')
        best_ratio = 0
        best_pseudo_preds = None
        
        for ratio in Config.PSEUDO_RATIOS:
            logger.info(f"\n--- Experimenting with Pseudo-Label Ratio: {ratio} ---")
            pseudo_df = self.generate_pseudo_labels(test_df, ratio)
            combined_train = pd.concat([self.train, pseudo_df], axis=0).reset_index(drop=True)
            lgb_mae, cat_mae = self.train_kfolds(feature_cols, train_df=combined_train, seeds=Config.SEEDS[:1])
            avg_mae = (lgb_mae + cat_mae) / 2.0
            
            if avg_mae < best_mae:
                best_mae = avg_mae
                best_ratio = ratio
                best_pseudo_preds = (self.test_preds_lgb + self.test_preds_cat) / 2.0
                
        logger.info(f"\n★ Best Pseudo-Label Ratio: {best_ratio} (MAE: {best_mae:.4f})")
        return best_ratio, best_mae, best_pseudo_preds

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
        """Identify features contributing to 95% of cumulative gain."""
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
        return selected_features

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
