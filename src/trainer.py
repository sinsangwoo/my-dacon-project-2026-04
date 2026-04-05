import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
import logging
from .config import Config

logger = logging.getLogger(__name__)

class Trainer:
    """Class to handle Advanced Cross-Validation, Stacking, and Pseudo Labeling."""
    def __init__(self, train, test, feature_cols, groups):
        self.train = train.reset_index(drop=True)
        self.test = test.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.groups = groups
        
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
        tr_df = tr_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        
        means = tr_df.groupby('layout_id')[target_col].mean()
        global_mean = tr_df[target_col].mean()
        tr_df['layout_id_mean_target'] = tr_df['layout_id'].map(means).fillna(global_mean)
        val_df['layout_id_mean_target'] = val_df['layout_id'].map(means).fillna(global_mean)
        test_enc = test_df['layout_id'].map(means).fillna(global_mean)
        return tr_df, val_df, test_enc

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
            oof_lgb_seed = np.zeros(len(train_df))
            oof_cat_seed = np.zeros(len(train_df))
            
            # Reset KF for each seed if not GroupKFold
            if Config.SPLIT_STRATEGY != 'GroupKFold':
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=Config.NFOLDS, shuffle=True, random_state=seed)
            else:
                kf = self.kf
                
            groups = train_df['scenario_id'] if 'scenario_id' in train_df.columns else None
            
            for fold, (tr_idx, val_idx) in enumerate(kf.split(train_df, groups=groups)):
                df_tr = train_df.iloc[tr_idx].reset_index(drop=True)
                df_val = train_df.iloc[val_idx].reset_index(drop=True)
                
                df_tr, df_val, test_fold_enc = self._mean_encoding_cv(df_tr, df_val, self.test, Config.TARGET)
                fold_features = feature_cols + ['layout_id_mean_target']
                
                X_tr, y_tr = df_tr[fold_features], df_tr[Config.TARGET]
                X_val, y_val = df_val[fold_features], df_val[Config.TARGET]
                test_X = self.test[feature_cols].copy()
                test_X['layout_id_mean_target'] = test_fold_enc
                
                # 1. LightGBM
                vp, tp, imp = self._train_model('lgb', X_tr, y_tr, X_val, y_val, test_X, seed)
                oof_lgb_seed[val_idx] = vp
                test_preds_lgb_final += tp / (len(seeds) * Config.NFOLDS)
                self.test_preds_lgb_folds.append(tp)
                if seed == seeds[0]: self.importance_list.append(imp)
                
                # 2. CatBoost
                vp_cat, tp_cat, _ = self._train_model('cat', X_tr, y_tr, X_val, y_val, test_X, seed)
                oof_cat_seed[val_idx] = vp_cat
                test_preds_cat_final += tp_cat / (len(seeds) * Config.NFOLDS)
                self.test_preds_cat_folds.append(tp_cat)

            oof_lgb_final += oof_lgb_seed / len(seeds)
            oof_cat_final += oof_cat_seed / len(seeds)
            
        self.oof_lgb, self.oof_cat = oof_lgb_final, oof_cat_final
        self.test_preds_lgb, self.test_preds_cat = test_preds_lgb_final, test_preds_cat_final
        
        lgb_mae = mean_absolute_error(train_df[Config.TARGET], oof_lgb_final)
        cat_mae = mean_absolute_error(train_df[Config.TARGET], oof_cat_final)
        return lgb_mae, cat_mae

    def train_stacking(self, feature_cols):
        """Train meta-model (Ridge/LGBM) on OOF predictions."""
        logger.info(f"Training Stacking Meta-model ({Config.META_MODEL})...")
        
        # Meta features: OOF + top original features
        # For simplicity, we just use OOF_LGB and OOF_CAT
        meta_train = pd.DataFrame({'lgb': self.oof_lgb, 'cat': self.oof_cat})
        meta_test = pd.DataFrame({'lgb': self.test_preds_lgb, 'cat': self.test_preds_cat})
        
        from sklearn.linear_model import Ridge
        oof_stack = np.zeros(len(self.train))
        test_preds_stack = np.zeros(len(self.test))
        
        kf = self.kf # Use same split for stacking
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
        # Confidence = low variance across folds
        all_folds = np.array(self.test_preds_lgb_folds) # [Folds*Seeds, N_test]
        std_folds = np.std(all_folds, axis=0)
        
        n_pseudo = int(len(test_df) * ratio)
        # Apply safety guard: pseudo data cannot exceed 30% of total training data
        max_allowed = int(len(self.train) * Config.MAX_PSEUDO_RATIO)
        n_pseudo = min(n_pseudo, max_allowed)
        
        # Higher confidence = lower standard deviation
        confidence_idx = np.argsort(std_folds)[:n_pseudo]
        
        pseudo_df = test_df.iloc[confidence_idx].copy()
        # Predictions are the averaged test predictions
        pseudo_df[Config.TARGET] = (self.test_preds_lgb[confidence_idx] + self.test_preds_cat[confidence_idx]) / 2.0
        
        # Distribution Check Logging
        logger.info(f"--- Pseudo-Labeling Confidence Selection (Ratio: {ratio*100}%) ---")
        logger.info(f"Target Distribution: Train Mean={self.train[Config.TARGET].mean():.3f}, Pseudo Mean={pseudo_df[Config.TARGET].mean():.3f}")
        logger.info(f"Target Distribution: Train Std={self.train[Config.TARGET].std():.3f}, Pseudo Std={pseudo_df[Config.TARGET].std():.3f}")
        
        return pseudo_df

    def run_pseudo_labeling_experiments(self, feature_cols, test_df):
        """Experiment with 10%, 20%, 30% pseudo-label ratios."""
        best_mae = float('inf')
        best_ratio = 0
        best_pseudo_preds = None
        
        for ratio in Config.PSEUDO_RATIOS:
            logger.info(f"\n--- Experimenting with Pseudo-Label Ratio: {ratio} ---")
            pseudo_df = self.generate_pseudo_labels(test_df, ratio)
            
            # Combined Train
            combined_train = pd.concat([self.train, pseudo_df], axis=0).reset_index(drop=True)
            
            # Retrain (Use fast config if needed, but here we use original seeds for accuracy)
            lgb_mae, cat_mae = self.train_kfolds(feature_cols, train_df=combined_train, seeds=Config.SEEDS[:1])
            avg_mae = (lgb_mae + cat_mae) / 2.0
            
            logger.info(f"Ratio {ratio} Result -> Avg MAE: {avg_mae:.4f}")
            
            if avg_mae < best_mae:
                best_mae = avg_mae
                best_ratio = ratio
                best_pseudo_preds = (self.test_preds_lgb + self.test_preds_cat) / 2.0
                
        logger.info(f"\n★ Best Pseudo-Label Ratio: {best_ratio} (MAE: {best_mae:.4f})")
        return best_ratio, best_mae, best_pseudo_preds

    def generate_interactions(self, train, top_20_cols):
        # Keep existing interaction generation
        df = train.copy()
        interaction_cols = []
        for i in range(len(top_20_cols)):
            for j in range(i + 1, len(top_20_cols)):
                c1, c2 = top_20_cols[i], top_20_cols[j]
                new_col = f'inter_{c1}_x_{c2}'
                df[new_col] = df[c1] * df[c2]
                interaction_cols.append(new_col)
        return df, interaction_cols

    def prune_interactions(self, df, interaction_cols, groups):
        # Keep existing pruning
        from sklearn.model_selection import KFold
        kf = GroupKFold(n_splits=3)
        fold1_tr, fold1_val = next(kf.split(df, groups=groups))
        X_tr = df.loc[fold1_tr, interaction_cols]
        y_tr = df.loc[fold1_tr, Config.TARGET]
        model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        model.fit(X_tr, y_tr)
        importances = pd.Series(model.feature_importances_, index=interaction_cols)
        return importances.sort_values(ascending=False).head(Config.INT_PRUNE_K).index.tolist()

    def filter_unstable_features(self, feature_cols, importance_list):
        # Keep existing stability check
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
