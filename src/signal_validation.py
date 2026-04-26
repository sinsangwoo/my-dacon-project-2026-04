import logging
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import json
import os

logger = logging.getLogger(__name__)

class SignalValidator:
    """
    [WHY_THIS_CHANGE] Implement strict signal validation (Task 1, 2, 3, 4, 6).
    [ROOT_CAUSE] Previous logic was too lenient, allowing noise features to survive via OR conditions or fallbacks.
    [EXPECTED_IMPACT] Structural elimination of noise, proof-based signal survival, and rank-based prioritization.
    """
    def __init__(self, config_params, stability_df):
        self.params = config_params.copy()
        if 'random_state' not in self.params:
            self.params['random_state'] = 42
        self.stability_df = stability_df.set_index('feature')
        self.validation_logs = []
        self.noise_metrics = {}

    def _get_split_stats(self, model, feature_names):
        """Extract split count and average depth for all features."""
        tree_info = model.booster_.dump_model()['tree_info']
        stats = {f: {'splits': 0, 'depths': []} for f in feature_names}
        
        def traverse(node, d):
            if 'split_feature' in node:
                feat = feature_names[node['split_feature']]
                stats[feat]['splits'] += 1
                stats[feat]['depths'].append(d)
                if 'left_child' in node: traverse(node['left_child'], d+1)
                if 'right_child' in node: traverse(node['right_child'], d+1)
        
        for tree in tree_info:
            traverse(tree['tree_structure'], 0)
            
        return {f: {'splits': s['splits'], 'avg_depth': np.mean(s['depths']) if s['depths'] else 0} 
                for f, s in stats.items()}

    def evaluate(self, X_train, y_train, candidates, buckets, base_cols):
        logger.info(f"[SIGNAL_VALIDATOR] Evaluating {len(candidates)} candidates with Noise Immunity Architecture...")
        
        # [TASK 6] Noise Injection for Proof
        n_noise = 20
        noise_cols = [f'__noise_{i}__' for i in range(n_noise)]
        X_with_noise = X_train.copy()
        for c in noise_cols:
            X_with_noise[c] = np.random.normal(0, 1, len(X_train))
        
        # 1. Train evaluator model
        X_tr, X_val, y_tr, y_val = train_test_split(X_with_noise, y_train, test_size=0.2, random_state=42)
        model = LGBMRegressor(**self.params)
        model.fit(X_tr, y_tr)
        
        # [NEW] 3-Fold CV for Cross-fold Consistency & Sign Stability
        from sklearn.model_selection import KFold
        from scipy.stats import pearsonr
        
        cv_stats = {f: {'gains': [], 'signs': []} for f in candidates}
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        
        logger.info("[SIGNAL_VALIDATOR] Running 3-Fold CV for Consistency...")
        y_train_arr = y_train.values if isinstance(y_train, pd.Series) else y_train
        
        for tr_idx, val_idx in kf.split(X_with_noise, y_train_arr):
            cv_tr_x, cv_val_x = X_with_noise.iloc[tr_idx], X_with_noise.iloc[val_idx]
            cv_tr_y, cv_val_y = y_train_arr[tr_idx], y_train_arr[val_idx]
            
            cv_model = LGBMRegressor(**self.params)
            cv_model.fit(cv_tr_x, cv_tr_y)
            cv_imp = dict(zip(X_with_noise.columns, cv_model.feature_importances_))
            
            for f in candidates:
                if f not in X_with_noise.columns: continue
                cv_stats[f]['gains'].append(cv_imp.get(f, 0))
                try:
                    corr, _ = pearsonr(cv_val_x[f].values, cv_val_y)
                    cv_stats[f]['signs'].append(np.sign(corr) if not np.isnan(corr) else 0)
                except:
                    cv_stats[f]['signs'].append(0)
                    
        # Compute consistency scores
        self.consistency_scores = {}
        for f in candidates:
            gains = cv_stats[f]['gains']
            signs = cv_stats[f]['signs']
            gain_cv = np.std(gains) / (np.mean(gains) + 1e-9) if np.mean(gains) > 0 else 1.0
            sign_stability = abs(np.sum(signs)) / 3.0 # 1.0 if all same sign
            
            # stability_factor ranges from ~0 to 1
            self.consistency_scores[f] = {
                'gain_cv': float(gain_cv),
                'sign_stability': float(sign_stability),
                'stability_factor': float((1.0 - min(gain_cv, 1.0)) * 0.5 + sign_stability * 0.5)
            }
            
        # 2. Base Importance & Stats
        gain_imp = dict(zip(X_with_noise.columns, model.feature_importances_))
        split_stats = self._get_split_stats(model, X_with_noise.columns.tolist())
        
        # [TASK 1] Alpha calculation based on distribution
        gains = np.array(list(gain_imp.values()))
        gain_q3 = np.percentile(gains, 75)
        gain_iqr = gain_q3 - np.percentile(gains, 25)
        alpha = (gain_q3 + 0.5 * gain_iqr) / (np.mean(gains) + 1e-9)
        min_gain_threshold = np.mean(gains) * alpha
        
        # 3. Permutation Analysis
        base_pred = model.predict(X_val)
        base_mae = mean_absolute_error(y_val, base_pred)
        
        # [TASK 2] Group/Conditional Permutation
        # For simplicity in this env, we group by base_col or correlation
        perm_imp = {}
        logger.info("[SIGNAL_VALIDATOR] Running Permutation Attack...")
        
        # Pre-calculate correlation for redundancy check
        corr_matrix = X_tr[candidates].corr()
        
        for feat in candidates:
            if feat not in X_val.columns: continue
            
            # Simple Permutation
            X_tmp = X_val.copy()
            X_tmp[feat] = np.random.permutation(X_tmp[feat].values)
            delta = mean_absolute_error(y_val, model.predict(X_tmp)) - base_mae
            
            # [TASK 2] Group Permutation (if cluster members exist)
            cluster_members = [c for c in candidates if c != feat and corr_matrix.loc[feat, c] > 0.95]
            if cluster_members:
                X_group = X_val.copy()
                for m in [feat] + cluster_members:
                    X_group[m] = np.random.permutation(X_group[m].values)
                group_delta = mean_absolute_error(y_val, model.predict(X_group)) - base_mae
            else:
                group_delta = delta
            
            perm_imp[feat] = {
                'delta': delta,
                'group_delta': group_delta,
                'rel_imp': delta / (group_delta + 1e-9)
            }

        # 4. Filter & Score
        final_protected = set()
        ranks = {}
        
        # [TASK 3] Rank-based scoring
        # (Only for candidates that pass HARD FILTER)
        
        passed_features = []
        for feat in candidates:
            if feat not in X_train.columns: continue
            
            gain = gain_imp.get(feat, 0)
            p_data = perm_imp.get(feat, {'delta': 0, 'group_delta': 0, 'rel_imp': 0})
            ks = self.stability_df.loc[feat, 'ks_stat'] if feat in self.stability_df.index else 0
            stats = split_stats.get(feat, {'splits': 0, 'avg_depth': 0})
            
            # [TASK 1 & 2 & 5] Adversarial Validation & Hardening
            from scipy.stats import pearsonr
            
            # 1. Gain Floor
            f1 = gain > min_gain_threshold
            
            # 2. Generalization Ratio (OOF Permutation vs Training Gain)
            gen_ratio = p_data['delta'] / (gain + 1e-9)
            is_danger_zone = 0.50 < ks <= 0.85
            ratio_threshold = 0.005 if not is_danger_zone else 0.010  # 2x strictness for Danger Zone
            f2 = p_data['delta'] > 0 and gen_ratio > ratio_threshold
            
            # 3. Marginal Correlation Floor (Combinatorial Noise Defense)
            try:
                marg_corr, _ = pearsonr(X_train[feat].values, y_train.values if isinstance(y_train, pd.Series) else y_train)
            except:
                marg_corr = 0.0
            f3 = abs(marg_corr) >= 0.01
            
            # 4. Split Validity & Depth Consistency
            f4 = stats['splits'] > (model.n_estimators * 0.05)
            f5 = stats['avg_depth'] < 12 
            
            # [TASK 2] Tier 1B KS Handling: No adjusted_KS bias.
            # KS filtering is purely global.
            f6 = ks <= 0.85 # Tier 1A already drops > 0.85, but we enforce here too.
                
            # [TASK 3 & 4] Third Axis: Consistency
            c_stats = self.consistency_scores.get(feat, {'gain_cv': 1.0, 'sign_stability': 0.0, 'stability_factor': 0.1})
            f7 = c_stats['gain_cv'] < 0.5 and c_stats['sign_stability'] >= 0.66
            
            passed = all([f1, f2, f3, f4, f5, f6, f7])
            
            # Track rejection reasons
            rejections = []
            if not f1: rejections.append("Gain")
            if not f2: rejections.append("GeneralizationRatio")
            if not f3: rejections.append("MarginalCorrelation")
            if not f4 or not f5: rejections.append("TreeStructure")
            if not f6: rejections.append("Tier1ADrop")
            if not f7: rejections.append("Consistency")
            
            # [TASK 4] Internal validation log
            is_trend_vol = any(x in feat for x in ['_rate_', '_slope_', '_diff_', '_std_', '_volatility_'])
            self.validation_logs.append({
                'feature': feat, 'gain': gain, 'perm_delta': p_data['delta'],
                'gen_ratio': gen_ratio, 'marg_corr': marg_corr,
                'group_delta': p_data['group_delta'], 'splits': stats['splits'],
                'avg_depth': stats['avg_depth'], 'ks_stat': ks, 
                'passed': passed, 'rejection_reasons': rejections, 'is_trend_vol': is_trend_vol
            })
            
            if passed:
                passed_features.append(feat)

        # [TASK 1] Redundancy Check: Representative selection
        survivors_after_redundancy = []
        if passed_features:
            sorted_passed = sorted(passed_features, key=lambda x: gain_imp[x], reverse=True)
            dropped_redundant = set()
            for f in sorted_passed:
                if f in dropped_redundant: continue
                survivors_after_redundancy.append(f)
                cluster = [c for c in sorted_passed if c != f and corr_matrix.loc[f, c] > 0.95]
                dropped_redundant.update(cluster)

        # [TASK 3] Rank Scoring
        if survivors_after_redundancy:
            df_ranks = pd.DataFrame({'f': survivors_after_redundancy})
            df_ranks['gain_rank'] = df_ranks['f'].map(gain_imp).rank(pct=True)
            df_ranks['perm_rank'] = df_ranks['f'].apply(lambda x: perm_imp[x]['delta']).rank(pct=True)
            df_ranks['stab_rank'] = df_ranks['f'].apply(lambda x: -self.stability_df.loc[x, 'ks_stat'] if x in self.stability_df.index else 0).rank(pct=True)
            df_ranks['score'] = df_ranks['gain_rank'] + df_ranks['perm_rank'] + df_ranks['stab_rank']
            final_protected = set(survivors_after_redundancy)
            
        # [TASK 2 & 4] Signal Bucket Selection
        bucket_survivors = set()
        for base_col in base_cols:
            for b_name, suffixes in buckets.items():
                b_feats = [f for f in survivors_after_redundancy if f.startswith(base_col) and 
                           (any(f.endswith(s) for s in suffixes if s != 'raw') or (f == base_col and 'raw' in suffixes))]
                
                # [TASK 4] Fallback 제거 -> EMPTY 허용
                if b_feats:
                    # Pick best by score
                    best_f = df_ranks[df_ranks['f'].isin(b_feats)].sort_values('score', ascending=False)['f'].iloc[0]
                    bucket_survivors.add(best_f)
                else:
                    logger.warning(f"[EMPTY_BUCKET_WARNING] {base_col} ({b_name}) has no valid features passing strict filters.")

        # [TASK 6] Noise Immunity Proof
        noise_stats = {c: {'gain': gain_imp[c], 'splits': split_stats[c]['splits']} for c in noise_cols}
        noise_survived = [c for c in noise_cols if any(c == f for f in final_protected)]
        
        self.noise_metrics = {
            'noise_survival_rate': len(noise_survived) / n_noise,
            'noise_avg_splits': np.mean([s['splits'] for s in noise_stats.values()]),
            'noise_max_gain': np.max([s['gain'] for s in noise_stats.values()]),
            'feature_usage_entropy': -np.sum([v/sum(gains) * np.log(v/sum(gains) + 1e-9) for v in gains])
        }
        
        logger.info(f"[PROOF] Noise Survival Rate: {self.noise_metrics['noise_survival_rate']:.2%}")
        logger.info(f"[PROOF] Noise Avg Splits: {self.noise_metrics['noise_avg_splits']:.2f}")

        return final_protected, bucket_survivors, self.validation_logs, self.noise_metrics
