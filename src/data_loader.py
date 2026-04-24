import gc
import logging
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import rankdata

from .config import Config
from .schema import FEATURE_SCHEMA
from .utils import downcast_df, inspect_columns, track_lineage, ensure_dataframe, GlobalStatStore

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# [PHASE 2: EMBEDDING RESTORATION]
# ─────────────────────────────────────────────────────────────────────────────

class TwoPassImputer:
    def __init__(self):
        self.global_imputer = SimpleImputer(strategy='median')
        self.layout_residuals = {}

    def fit(self, df, cols):
        self.global_imputer.fit(df[cols])
        global_medians = self.global_imputer.statistics_
        for lid, group in df.groupby('layout_id'):
            group_median = group[cols].median()
            residual = group_median.values - global_medians
            self.layout_residuals[lid] = np.nan_to_num(residual).astype('float32')

    def transform(self, df, cols):
        df = df.copy()
        df[cols] = self.global_imputer.transform(df[cols])
        for lid, res in self.layout_residuals.items():
            mask = df['layout_id'] == lid
            if mask.any():
                df.loc[mask, cols] += res
        return df

class SuperchargedPCAReconstructor:
    """Supervised, Multi-View PCA Reconstructor."""
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.pca_raw = PCA(n_components=8, svd_solver='randomized', random_state=42)
        self.pca_log = PCA(n_components=8, svd_solver='randomized', random_state=42)
        self.pca_rank = PCA(n_components=8, svd_solver='randomized', random_state=42)
        self.global_pca_local = PCA(n_components=8, svd_solver='randomized', random_state=42)
        self.kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto')
        # self.scaler = StandardScaler() # [PHASE 2: REMOVE DUPLICATED SCALING]
        self.imputer = SimpleImputer(strategy='median')
        self.embed_dim = 32 
        
        self.pool_embed = None
        self.pool_norm = None
        self.rank_reference = None # [PHASE 4: DETERMINISTIC RANK]

    def _get_multi_view_X(self, X):
        X_raw = X
        X_log = np.log1p(np.abs(X)) * np.sign(X)
        X_rank = np.zeros_like(X)
        
        # [PHASE 4: DETERMINISTIC RANK]
        # Use rank computed using global train reference instead of batch-size dependent rankdata
        if self.rank_reference is not None:
            for i in range(X.shape[1]):
                # Efficient percentile computation using searchsorted on reference
                ref = self.rank_reference[:, i]
                X_rank[:, i] = np.searchsorted(ref, X[:, i]) / (len(ref) + 1)
        else:
            # Fallback for fit time (will be stored after first fit)
            for i in range(X.shape[1]):
                X_rank[:, i] = rankdata(X[:, i]) / (len(X) + 1)
        return X_raw, X_log, X_rank

    def fit(self, X_train, residuals=None):
        # [DIM_TRACE] STEP 1: RAW INPUT
        X_np = np.asarray(X_train, dtype=np.float32)
        logger.debug(f"[DIM_TRACE] fit raw input: shape={X_np.shape}, dtype={X_np.dtype} | source=Reconstructor.fit")
        
        # [CONTRACT_ENFORCEMENT]
        if X_np.shape[1] == len(Config.EMBED_BASE_COLS):
            X_embed_base = X_np
        elif X_np.shape[1] == len(FEATURE_SCHEMA['raw_features']):
            base_cols_idx = [i for i, col in enumerate(FEATURE_SCHEMA['raw_features']) if col in Config.EMBED_BASE_COLS]
            X_embed_base = X_np[:, base_cols_idx]
        else:
            raise ValueError(f"[CONTRACT_FAIL] Reconstructor.fit received unexpected shape {X_np.shape}. Expected 30 or 698 features.")
        
        # [RELIABILITY_FIX] Handle Infs before imputation (v18.5)
        X_embed_base = np.where(np.isinf(X_embed_base), np.nan, X_embed_base)
        
        # [DIM_TRACE] STEP 2: PREPROCESSING
        X_embed_base = self.imputer.fit_transform(X_embed_base)
        # X_scaled = self.scaler.fit_transform(X_embed_base) # [PHASE 2: REMOVE DUPLICATED SCALING]
        X_scaled = X_embed_base # X_train MUST be already scaled by DriftShieldScaler
        logger.debug(f"[DIM_TRACE] after preprocessing: shape={X_scaled.shape} | source=imputer")
        
        # [PHASE 4: STORE RANK REFERENCE]
        if self.rank_reference is None:
            self.rank_reference = np.sort(X_scaled, axis=0)

        weights = 1.0 + np.abs(residuals) if residuals is not None else np.ones(len(X_np))
        X_weighted = X_scaled * weights[:, np.newaxis]
        
        X_w_raw, X_w_log, X_w_rank = self._get_multi_view_X(X_weighted)
        self.pca_raw.fit(X_w_raw)
        self.pca_log.fit(X_w_log)
        self.pca_rank.fit(X_w_rank)
        self.global_pca_local.fit(X_weighted)
        
        # [PHASE 8: PCA QUALITY CONTROL]
        for tag, pca in [('raw', self.pca_raw), ('log', self.pca_log), ('rank', self.pca_rank), ('local', self.global_pca_local)]:
            var_sum = np.sum(pca.explained_variance_ratio_)
            if var_sum < 0.8:
                err_msg = f"[PCA_QC_FAILURE] {tag} PCA explained variance: {var_sum:.4f} < 0.8. ABORTING."
                logger.error(err_msg)
                logger.info(f"[{tag}] Component variance ratios: {pca.explained_variance_ratio_}")
                raise RuntimeError(err_msg)
            logger.info(f"[PCA_QC_SUCCESS] {tag} PCA explained variance: {var_sum:.4f} >= 0.8")

        # [DIM_TRACE] STEP 3: EMBEDDING
        final_embed = self.get_embeddings(X_embed_base, already_scaled=True)
        logger.debug(f"[DIM_TRACE] after embedding: shape={final_embed.shape} | source=get_embeddings")
        
        self.kmeans.fit(final_embed.astype(np.float64))

    def get_embeddings(self, X, already_scaled=False):
        X_np = np.asarray(X, dtype=np.float32)
        
        if not already_scaled:
            # [CONTRACT_ENFORCEMENT]
            if X_np.shape[1] != len(Config.EMBED_BASE_COLS):
                 raise ValueError(f"[CONTRACT_FAIL] get_embeddings expected {len(Config.EMBED_BASE_COLS)} cols, got {X_np.shape[1]}")
            
            X_np = np.where(np.isinf(X_np), np.nan, X_np)
            X_np = self.imputer.transform(X_np)
            # X_scaled = self.scaler.transform(X_np) # [PHASE 2: REMOVE DUPLICATED SCALING]
            X_scaled = X_np # Assume already scaled by DriftShieldScaler if not already_scaled=True
        else:
            X_scaled = X_np
            
        X_raw, X_log, X_rank = self._get_multi_view_X(X_scaled)
        
        e_raw = self.pca_raw.transform(X_raw)
        e_log = self.pca_log.transform(X_log)
        e_rank = self.pca_rank.transform(X_rank)
        e_local = self.global_pca_local.transform(X_scaled)
        
        combined = np.hstack([e_raw, e_log, e_rank, e_local])
        # [PHASE 5: REMOVE FAKE STABILITY] Remove arbitrary 1.7 scaling
        return combined.astype(np.float32)

    def build_fold_cache(self, df_train_pool):
        """Build and cache pool embeddings once per fold."""
        base_cols = Config.EMBED_BASE_COLS
        # [PHASE 2: SSOT SCALING] Ensure we use already scaled data from df_train_pool if possible
        # Or assume get_embeddings handles it if we pass raw (but it shouldn't anymore)
        self.pool_embed = self.get_embeddings(df_train_pool[base_cols].values, already_scaled=True)
        self.pool_norm = self.pool_embed / (np.linalg.norm(self.pool_embed, axis=1, keepdims=True) + 1e-8)
        
    def clear_fold_cache(self):
        """Clear cache to ensure no leakage."""
        self.pool_embed = None
        self.pool_norm = None

    def calculate_graph_stats(self, df_target, is_train=False):
        """Compute multi-scale neighbor stats with high-performance indexing.
        
        [PHASE 3: KNN LEAKAGE ELIMINATION]
        If is_train=True, excludes the self-index from neighbors.
        """
        base_cols = Config.EMBED_BASE_COLS
        # [PHASE 2: SSOT SCALING] Assume df_target is already scaled if it comes from the main pipeline
        target_embed_all = self.get_embeddings(df_target[base_cols].values, already_scaled=True)
        
        if self.pool_embed is not None and self.pool_norm is not None:
            pool_embed = self.pool_embed
            pool_norm = self.pool_norm
        else:
            raise RuntimeError("[LEAKAGE GUARD] Cache missing. Reconstructor must be loaded from a pre-fitted train artifact.")
        
        target_norm_all = target_embed_all / (np.linalg.norm(target_embed_all, axis=1, keepdims=True) + 1e-8)
        
        # Chunking to prevent memory explosion (OOM Guard)
        chunk_size = getattr(Config, 'EMBED_CHUNK_SIZE', 2000)
        n_targets = len(target_norm_all)
        final_results = {}
        
        for i in range(0, n_targets, chunk_size):
            t_norm = target_norm_all[i:i+chunk_size]
            t_embed = target_embed_all[i:i+chunk_size]
            row_idx = np.arange(len(t_norm))[:, None]
            
            # 1. Distance matrix for this chunk (Single Pass)
            dist_matrix = 1 - np.dot(t_norm, pool_norm.T)
            
            # 2. Efficient Neighbor Search (O(N) Partitioning)
            # [PHASE 3: EXCLUDE SELF] If is_train, we need k+1 neighbors to exclude self
            max_k = max(Config.MULTI_K)
            search_k = max_k + 1 if is_train else max_k
            
            nn_indices_all = np.argpartition(dist_matrix, search_k, axis=1)[:, :search_k]
            
            # 3. Sort subset for exact sim-order (Required for weighted stats)
            top_dists = dist_matrix[row_idx, nn_indices_all]
            sort_idx = np.argsort(top_dists, axis=1)
            nn_indices_all = nn_indices_all[row_idx, sort_idx]
            
            # [RELIABILITY_FIX] Define candidates dists for leakage check
            candidate_dists = dist_matrix[row_idx, nn_indices_all]
            
            # [PHASE 3: EXCLUDE SELF]
            if is_train:
                # [RELIABILITY_FIX] Enforce BOTH index exclusion AND distance floor (1e-6)
                # We assume pool and target are aligned for self-exclusion in training
                
                # Identify self-neighbors by index (most reliable)
                current_target_indices = np.arange(i, min(i + chunk_size, n_targets))
                
                # Check if first neighbor is self by index OR distance
                first_neighbor_idx = nn_indices_all[:, 0]
                first_neighbor_dist = candidate_dists[:, 0]
                
                # Leakage if (index match) OR (distance < 1e-12)
                is_leakage = (first_neighbor_idx == current_target_indices) | (first_neighbor_dist < 1e-12)
                
                if is_leakage.any():
                    # For leakage rows, shift neighbors by 1
                    nn_indices_all = np.where(is_leakage[:, None], nn_indices_all[:, 1:max_k+1], nn_indices_all[:, :max_k])
                    final_dists = np.where(is_leakage[:, None], candidate_dists[:, 1:max_k+1], candidate_dists[:, :max_k])
                else:
                    nn_indices_all = nn_indices_all[:, :max_k]
                    final_dists = candidate_dists[:, :max_k]
                
                # [PART 2: FORCE DISTANCE] Enforce distance > 1e-6 for all selected neighbors
                if (final_dists < 1e-6).any():
                    logger.error(f"[KNN_LEAKAGE_CRITICAL] Distance < 1e-6 detected after exclusion! Min dist: {final_dists.min():.2e}")
                    raise RuntimeError("[KNN_LEAKAGE_CRITICAL] Self-neighbor inclusion detected.")
            else:
                nn_indices_all = nn_indices_all[:, :max_k]
                final_dists = candidate_dists[:, :max_k]

            
            # 4. Multi-Scale Feature Generation
            for k in Config.MULTI_K:
                nn_indices = nn_indices_all[:, :k]
                neighbor_embeds = pool_embed[nn_indices]
                k_dists = final_dists[:, :k]
                
                # [PHASE 10: REMOVE DUPLICATED LOGIC]
                # Vectorized Aggregations
                # [PHASE 1: CANONICAL NAMING] embed_mean_{k}_d{dim}
                # We store them as (N, 32) arrays, apply_latent_features will handle _d{dim} suffix
                
                mean_feat = neighbor_embeds.mean(axis=1).astype(np.float32)
                std_feat = neighbor_embeds.std(axis=1).astype(np.float32)
                
                sim_scores = 1.0 - k_dists
                weights = np.exp(sim_scores) / np.sum(np.exp(sim_scores), axis=1, keepdims=True)
                weighted_mean_feat = np.sum(weights[:, :, None] * neighbor_embeds, axis=1).astype(np.float32)
                
                # Store in final_results (concatenate later)
                if f'embed_mean_{k}' not in final_results: final_results[f'embed_mean_{k}'] = []
                if f'embed_std_{k}' not in final_results: final_results[f'embed_std_{k}'] = []
                if f'weighted_mean_{k}' not in final_results: final_results[f'weighted_mean_{k}'] = []
                if f'trend_proxy_{k}' not in final_results: final_results[f'trend_proxy_{k}'] = []
                if f'volatility_proxy_{k}' not in final_results: final_results[f'volatility_proxy_{k}'] = []
                
                final_results[f'embed_mean_{k}'].append(mean_feat)
                final_results[f'embed_std_{k}'].append(std_feat)
                final_results[f'weighted_mean_{k}'].append(weighted_mean_feat)
                final_results[f'trend_proxy_{k}'].append((mean_feat - t_embed).astype(np.float32))
                final_results[f'volatility_proxy_{k}'].append(std_feat)

            # Global features (not k-dependent or fixed k)
            if 'regime_proxy' not in final_results: final_results['regime_proxy'] = []
            if 'local_density' not in final_results: final_results['local_density'] = []
            if 'similarity_entropy' not in final_results: final_results['similarity_entropy'] = []
            
            final_results['regime_proxy'].append(self.kmeans.predict(t_embed.astype(np.float64)).astype(np.float32))
            
            # Use k=20 for local density as a stable reference
            avg_dist_20 = final_dists[:, :20].mean(axis=1) if search_k >= 20 else final_dists.mean(axis=1)
            final_results['local_density'].append((1.0 / (avg_dist_20 + 1e-6)).astype(np.float32))
            
            # Entropy for max k
            sim_scores_max = 1.0 - final_dists
            weights_max = np.exp(sim_scores_max) / np.sum(np.exp(sim_scores_max), axis=1, keepdims=True)
            final_results['similarity_entropy'].append((-np.sum(weights_max * np.log(weights_max + 1e-8), axis=1)).astype(np.float32))
            
            # [MEMORY_OPTIMIZATION] Explicit deletion and GC
            del dist_matrix, nn_indices_all, top_dists, sort_idx, weights_max
            if i % (chunk_size * 5) == 0:
                gc.collect()

        # 5. Final Reconstruction (Concatenate Chunks)
        concatenated_results = {}
        for key in final_results.keys():
            concatenated_results[key] = np.concatenate(final_results[key], axis=0)
            
        return concatenated_results

# ─────────────────────────────────────────────────────────────────────────────
# [PHASE 3: DATA FLOW ALIGNMENT]
# ─────────────────────────────────────────────────────────────────────────────

def build_base_features(df):
    """
    [PHASE 1: ISOLATED BASE FEATURES]
    Builds temporal and row-wise features that are safe to compute globally.
    """
    logger.info(f"[BUILD_BASE] Input shape: {df.shape}")
    df = df.copy()
    original_ids = df['ID'].values.copy()
    
    # 1. Temporal Ordering
    df = df.sort_values(by=["scenario_id", "ID"]).reset_index(drop=True)
    
    # 2. Base Feature Engineering
    df = add_time_series_features(df)
    df = add_extreme_detection_features(df)
    
    # 3. Order Restoration
    df = df.set_index('ID').loc[original_ids].reset_index()
    
    logger.info(f"[BUILD_BASE] Output shape: {df.shape}")
    return df

def apply_latent_features(df, reconstructor, scaler=None, selected_features=None, is_train=False):
    """
    [PHASE 2: FOLD-AWARE LATENT FEATURES]
    Applies fitted reconstructor and scaler to the dataframe.
    If selected_features is provided, ONLY those features are populated to save memory.
    """
    df = df.copy()
    
    # 0. Initialize expected features (Contract Enforcement)
    all_schema_features = selected_features if selected_features is not None else FEATURE_SCHEMA['all_features']
    existing_cols = set(df.columns)
    
    # [PHASE 1: HARD CHECK] RAISE ERROR IF EXPECTED FEATURES MISSING
    missing_base = [c for c in Config.EMBED_BASE_COLS if c not in existing_cols]
    if missing_base:
        raise RuntimeError(f"[SCHEMA_VIOLATION] Missing {len(missing_base)} base columns for embeddings: {missing_base}. NO SILENT FALLBACK ALLOWED.")
    
    # 1. Apply Drift Shield (if provided)
    if scaler is not None:
        df = scaler.transform(df, FEATURE_SCHEMA['raw_features'])

    # 2. Populate Latent Features
    logger.info("[LATENT_FEATURES] Populating latent features...")
    
    if reconstructor.pool_embed is None or reconstructor.pool_norm is None:
        raise RuntimeError("[LEAKAGE GUARD] Cache missing. Reconstructor must be loaded from a pre-fitted train artifact.")
    
    # [OPTIMIZATION] Process entire dataframe at once (Internal chunking handles memory)
    # [PHASE 3: EXCLUDE SELF] Pass is_train flag
    latent_stats = reconstructor.calculate_graph_stats(df, is_train=is_train)
    
    new_features_df_dict = {}
    for feat_name, values in latent_stats.items():
        if isinstance(values, np.ndarray) and values.ndim > 1:
            # Check if it's supposed to have _d suffixes according to SSOT
            # [SSOT_RULE] MULTI_K based features always have _d suffixes (32 dims)
            if values.shape[1] > 1:
                for d in range(values.shape[1]):
                    col_name = f"{feat_name}_d{d}"
                    if col_name in all_schema_features:
                        new_features_df_dict[col_name] = values[:, d].astype('float32')
            else:
                # (N, 1) case - regime_proxy, local_density, etc.
                col_name_d0 = f"{feat_name}_d0"
                if col_name_d0 in all_schema_features:
                    new_features_df_dict[col_name_d0] = values.ravel().astype('float32')
                elif feat_name in all_schema_features:
                    new_features_df_dict[feat_name] = values.ravel().astype('float32')
        else:
            col_name = feat_name
            if col_name in all_schema_features:
                val_flat = values.ravel() if isinstance(values, np.ndarray) else values
                new_features_df_dict[col_name] = val_flat.astype('float32')
    
    # [PHASE 1: HARD CHECK] Verify all expected features are present in new_features_df_dict or df
    # [MISSION: ZERO SILENT FAILURE]
    calculated_keys = set(new_features_df_dict.keys())
    for f in all_schema_features:
        if f in FEATURE_SCHEMA['embed_features'] and f not in calculated_keys:
             # Check if it was already in df (unlikely for latent)
             if f not in existing_cols:
                raise RuntimeError(f"[SCHEMA_VIOLATION] Feature {f} expected but NOT generated by reconstructor. ABORTING.")

    # Efficiently update the dataframe
    if new_features_df_dict:
        new_feats_df = pd.DataFrame(new_features_df_dict, index=df.index)
        # We don't use .update() because it might be slow or silent. 
        # Since we initialized new_cols_to_add in previous version, now we just concat the new features.
        # But wait, some might already exist. Let's be safe.
        for col in new_feats_df.columns:
            df[col] = new_feats_df[col]
        del new_feats_df; gc.collect()
                
    return df

def build_features(df, mode='raw', reconstructor=None, scaler=None, residuals=None, raw_preds=None):
    """Legacy entry point for compatibility.
    
    [PHASE 2: SSOT SCALING] Added scaler argument to ensure consistent scaling.
    """
    df_base = build_base_features(df)
    
    # In legacy mode, we still use the global stats if available for 'raw'
    if mode == 'raw':
        from .utils import assert_artifact_exists
        assert_artifact_exists(Config.GLOBAL_STATS_PATH, "Global Stats Cache")
        stats = GlobalStatStore.load(Config.GLOBAL_STATS_PATH)
        df_base = GlobalStatStore.apply_drift_shield(df_base, stats, FEATURE_SCHEMA['raw_features'])
    
    if mode == 'full' and reconstructor is not None:
        # Scale if scaler is provided
        if scaler is not None:
            df_base = scaler.transform(df_base, FEATURE_SCHEMA['raw_features'])
        df_base = apply_latent_features(df_base, reconstructor)

    all_schema_features = FEATURE_SCHEMA['all_features']
    X_df = df_base[all_schema_features].astype('float32').fillna(0.0)
    return X_df, df_base

def add_time_series_features(df):
    logger.info(f"[TS_FEATURES] Adding features to df shape {df.shape}")
    if "timestep_index" not in df.columns:
        df["timestep_index"] = df.groupby("scenario_id").cumcount().astype("int16")
    df["normalized_time"] = (df["timestep_index"] / 24.0).astype("float32")
    df["cold_start_flag"] = 0
    
    new_features = {}
    for col in Config.EMBED_BASE_COLS:
        # logger.debug(f"[TS_FEATURES] Processing {col}")
        series = df.groupby("scenario_id")[col]
        new_features[f"{col}_rolling_mean_3"] = series.rolling(3, min_periods=1).mean().values
        new_features[f"{col}_rolling_mean_5"] = series.rolling(5, min_periods=1).mean().values
        new_features[f"{col}_rolling_std_3"] = series.rolling(3, min_periods=1).std().values
        new_features[f"{col}_rolling_std_5"] = series.rolling(5, min_periods=1).std().values
        
        shift1 = series.shift(1).values
        shift3 = series.shift(3).values
        new_features[f"{col}_diff_1"] = df[col] - shift1
        new_features[f"{col}_diff_3"] = df[col] - shift3
        new_features[f"{col}_rate_1"] = new_features[f"{col}_diff_1"] / (np.abs(shift1) + 1e-6)
        
        # Optimized slope (approximate via diff for speed in smoke test)
        new_features[f"{col}_slope_5"] = series.rolling(5, min_periods=1).mean().diff().fillna(0).values
        
        new_features[f"{col}_recent_max_5"] = series.rolling(5, min_periods=1).max().values
        new_features[f"{col}_recent_min_5"] = series.rolling(5, min_periods=1).min().values
        new_features[f"{col}_range_5"] = new_features[f"{col}_recent_max_5"] - new_features[f"{col}_recent_min_5"]
        new_features[f"{col}_expanding_mean"] = series.expanding().mean().values
        new_features[f"{col}_expanding_sum"] = series.expanding().sum().values
        new_features[f"{col}_expanding_std"] = series.expanding().std().values

    logger.info(f"[TS_FEATURES] Concat-ing {len(new_features)} new features")
    ts_df = pd.DataFrame(new_features, index=df.index)
    
    # [MISSION: PCA FEATURE SANITIZATION] Step 2: Noise Detection
    # Identify features: rate_*, slope_*, diff_*, accel_*
    # Keep ONLY if: variance sufficient (Step 1 rule: > 1e-6)
    bad_derivatives = []
    for col in ts_df.columns:
        var = ts_df[col].var()
        if var < 1e-6 or np.isnan(var):
            bad_derivatives.append(col)
    
    if bad_derivatives:
        logger.info(f"[TS_SANITIZATION] Removing {len(bad_derivatives)} noisy derivatives (low variance)")
        ts_df = ts_df.drop(columns=bad_derivatives)
        
    return pd.concat([df, ts_df], axis=1)

def add_extreme_detection_features(df):
    new_features = {}
    for col in Config.EMBED_BASE_COLS:
        series = df.groupby("scenario_id")[col]
        rm5 = series.rolling(5, min_periods=1).mean().values
        rx5 = series.rolling(5, min_periods=1).max().values
        new_features[f"{col}_rel_to_mean_5"] = df[col] / (rm5 + 1e-6)
        new_features[f"{col}_rel_to_max_5"] = df[col] / (rx5 + 1e-6)
        new_features[f"{col}_rel_rank_5"] = series.rolling(5, min_periods=1).rank().values
        new_features[f"{col}_accel"] = (df[col] - series.shift(1).values) - (series.shift(1).values - series.shift(2).values)
        new_features[f"{col}_volatility_expansion_std"] = series.rolling(3, min_periods=1).std().values / (series.rolling(10, min_periods=1).std().values + 1e-6)
        new_features[f"{col}_volatility_expansion_range"] = (series.rolling(3, min_periods=1).max().values - series.rolling(3, min_periods=1).min().values) / (series.rolling(10, min_periods=1).max().values - series.rolling(10, min_periods=1).min().values + 1e-6)
        new_features[f"{col}_regime_id"] = pd.qcut(df[col], 5, labels=False, duplicates='drop')
        new_features[f"{col}_consecutive_above_q75"] = series.rolling(5, min_periods=1).apply(lambda x: (x > np.quantile(x, 0.75)).sum()).values

    # Interactions
    new_features['inter_order_inflow_15m_x_robot_utilization'] = df['order_inflow_15m'] * df['robot_utilization']
    new_features['inter_heavy_item_ratio_x_order_inflow_15m'] = df['heavy_item_ratio'] * df['order_inflow_15m']
    new_features['inter_heavy_item_ratio_x_robot_utilization'] = df['heavy_item_ratio'] * df['robot_utilization']
    
    # Early Warning
    new_features['early_warning_flag'] = ((new_features['order_inflow_15m_accel'] > 0) & (new_features['robot_utilization_rel_to_mean_5'] > 1.2)).astype(int)
    new_features['early_warning_score'] = new_features['order_inflow_15m_rel_to_mean_5'] + new_features['robot_utilization_rel_to_mean_5']

    # [PART 4: EXTREME VALUE SEMANTIC RESTORATION]
    # Detect extreme threshold: use 95th percentile of order_inflow_15m
    threshold = df['order_inflow_15m'].quantile(0.95)
    df['is_extreme'] = (df['order_inflow_15m'] >= threshold).astype(np.int8)
    
    # [MISSION: FINAL EDGE BOOST] Multi-feature extreme detection
    key_extreme_cols = ['order_inflow_15m', 'robot_utilization', 'congestion_score', 'near_collision_15m']
    extreme_masks = []
    for c in key_extreme_cols:
        if c in df.columns:
            q95 = df[c].quantile(0.95)
            extreme_masks.append(df[c] >= q95)
    
    if extreme_masks:
        df['is_extreme_multi'] = np.logical_or.reduce(extreme_masks).astype(np.int8)
    else:
        df['is_extreme_multi'] = df['is_extreme']
        
    coverage = df['is_extreme_multi'].mean()
    logger.info(f"[EXTREME_INTELLIGENCE] Threshold (P95): {threshold:.4f} | Coverage: {coverage:.2%}")

    return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)

def load_data():
    train = pd.read_csv(f"{Config.DATA_PATH}train.csv")
    test = pd.read_csv(f"{Config.DATA_PATH}test.csv")
    layout = pd.read_csv(Config.LAYOUT_PATH)
    train = train.merge(layout, on="layout_id", how="left")
    test = test.merge(layout, on="layout_id", how="left")
    return downcast_df(train), downcast_df(test)
