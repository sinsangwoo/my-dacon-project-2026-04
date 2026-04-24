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
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.embed_dim = 32 
        
        self.pool_embed = None
        self.pool_norm = None

    def _get_multi_view_X(self, X):
        X_raw = X
        X_log = np.log1p(np.abs(X)) * np.sign(X)
        X_rank = np.zeros_like(X)
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
        X_scaled = self.scaler.fit_transform(X_embed_base)
        logger.debug(f"[DIM_TRACE] after preprocessing: shape={X_scaled.shape} | source=imputer/scaler")
        
        weights = 1.0 + np.abs(residuals) if residuals is not None else np.ones(len(X_np))
        X_weighted = X_scaled * weights[:, np.newaxis]
        
        X_w_raw, X_w_log, X_w_rank = self._get_multi_view_X(X_weighted)
        self.pca_raw.fit(X_w_raw)
        self.pca_log.fit(X_w_log)
        self.pca_rank.fit(X_w_rank)
        self.global_pca_local.fit(X_weighted)
        
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
            X_scaled = self.scaler.transform(X_np)
        else:
            X_scaled = X_np
            
        X_raw, X_log, X_rank = self._get_multi_view_X(X_scaled)
        
        e_raw = self.pca_raw.transform(X_raw)
        e_log = self.pca_log.transform(X_log)
        e_rank = self.pca_rank.transform(X_rank)
        e_local = self.global_pca_local.transform(X_scaled)
        
        combined = np.hstack([e_raw, e_log, e_rank, e_local])
        return (combined * 1.7).astype(np.float32)

    def build_fold_cache(self, df_train_pool):
        """Build and cache pool embeddings once per fold."""
        base_cols = Config.EMBED_BASE_COLS
        self.pool_embed = self.get_embeddings(df_train_pool[base_cols].values)
        self.pool_norm = self.pool_embed / (np.linalg.norm(self.pool_embed, axis=1, keepdims=True) + 1e-8)
        
    def clear_fold_cache(self):
        """Clear cache to ensure no leakage."""
        self.pool_embed = None
        self.pool_norm = None

    def calculate_graph_stats(self, df_target, df_train_pool=None):
        """Compute multi-scale neighbor stats with high-performance indexing."""
        base_cols = Config.EMBED_BASE_COLS
        target_embed_all = self.get_embeddings(df_target[base_cols].values)
        
        if self.pool_embed is not None and self.pool_norm is not None:
            pool_embed = self.pool_embed
            pool_norm = self.pool_norm
        else:
            raise RuntimeError("[LEAKAGE GUARD] Cache missing. Reconstructor must be loaded from a pre-fitted train artifact.")
        
        target_norm_all = target_embed_all / (np.linalg.norm(target_embed_all, axis=1, keepdims=True) + 1e-8)
        
        # Chunking to prevent memory explosion (OOM Guard)
        chunk_size = getattr(Config, 'EMBED_CHUNK_SIZE', 2000)
        n_targets = len(target_norm_all)
        results_list = []
        
        for i in range(0, n_targets, chunk_size):
            t_norm = target_norm_all[i:i+chunk_size]
            t_embed = target_embed_all[i:i+chunk_size]
            row_idx = np.arange(len(t_norm))[:, None]
            
            chunk_res = {}
            # 1. Distance matrix for this chunk (Single Pass)
            dist_matrix = 1 - np.dot(t_norm, pool_norm.T)
            
            # 2. Efficient Neighbor Search (O(N) Partitioning)
            max_k = max(Config.MULTI_K)
            nn_indices_all = np.argpartition(dist_matrix, max_k, axis=1)[:, :max_k]
            
            # 3. Sort subset for exact sim-order (Required for weighted stats)
            top_dists = dist_matrix[row_idx, nn_indices_all]
            sort_idx = np.argsort(top_dists, axis=1)
            nn_indices_all = nn_indices_all[row_idx, sort_idx]
            
            weights = None # Keep track for similarity_entropy
            for k in Config.MULTI_K:
                nn_indices = nn_indices_all[:, :k]
                neighbor_embeds = pool_embed[nn_indices]
                
                # Vectorized Aggregations
                chunk_res[f'embed_mean_{k}'] = neighbor_embeds.mean(axis=1).astype(np.float32)
                chunk_res[f'embed_std_{k}'] = neighbor_embeds.std(axis=1).astype(np.float32)
                
                k_dists = dist_matrix[row_idx, nn_indices]
                sim_scores = 1.0 - k_dists
                weights = np.exp(sim_scores) / np.sum(np.exp(sim_scores), axis=1, keepdims=True)
                chunk_res[f'weighted_mean_{k}'] = np.sum(weights[:, :, None] * neighbor_embeds, axis=1).astype(np.float32)
                
                # [FIX: KEYERROR] Ensure all patterns from LATENT_PATTERNS are generated for each k
                # Dimension: (N, Dim) where Dim=32. SSOT expects f"{pattern}_{k}_d{d}"
                # We store them in chunk_res with the base pattern name
                chunk_res[f'embed_mean_{k}'] = neighbor_embeds.mean(axis=1).astype(np.float32)
                chunk_res[f'embed_std_{k}'] = neighbor_embeds.std(axis=1).astype(np.float32)
                
                # [CONTRACT_TRACE]
                logger.debug(f"[DIM_TRACE] neighbor_embeds: {neighbor_embeds.shape} | k: {k}")
                
                k_dists = dist_matrix[row_idx, nn_indices]
                sim_scores = 1.0 - k_dists
                weights = np.exp(sim_scores) / np.sum(np.exp(sim_scores), axis=1, keepdims=True)
                chunk_res[f'weighted_mean_{k}'] = np.sum(weights[:, :, None] * neighbor_embeds, axis=1).astype(np.float32)
                
                chunk_res[f'trend_proxy_{k}'] = (chunk_res[f'embed_mean_{k}'] - t_embed).astype(np.float32)
                chunk_res[f'volatility_proxy_{k}'] = chunk_res[f'embed_std_{k}'].copy() # Copy to avoid shared reference

            # [ROOT_CAUSE_REMOVAL] Disable explicit np.newaxis or dimension fixing
            chunk_res['regime_proxy'] = self.kmeans.predict(t_embed.astype(np.float64)).astype(np.float32)
            avg_dist_20 = dist_matrix[row_idx, nn_indices_all[:, :20]].mean(axis=1)
            chunk_res['local_density'] = (1.0 / (avg_dist_20 + 1e-6)).astype(np.float32)
            chunk_res['similarity_entropy'] = (-np.sum(weights * np.log(weights + 1e-8), axis=1)).astype(np.float32)
            
            results_list.append(chunk_res)
            
            # [MEMORY_OPTIMIZATION] Explicit deletion and GC
            del dist_matrix, nn_indices_all, top_dists, sort_idx, weights
            if i % (chunk_size * 5) == 0:
                gc.collect()

        # 4. Final Reconstruction (Concatenate Chunks)
        final_results = {}
        if results_list:
            for key in results_list[0].keys():
                final_results[key] = np.concatenate([c[key] for c in results_list], axis=0)
            
        return final_results

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

def apply_latent_features(df, reconstructor, scaler=None, selected_features=None):
    """
    [PHASE 2: FOLD-AWARE LATENT FEATURES]
    Applies fitted reconstructor and scaler to the dataframe.
    If selected_features is provided, ONLY those features are populated to save memory.
    """
    df = df.copy()
    
    # 0. Initialize expected features to zero (Contract Enforcement)
    # [MEMORY_OPTIMIZATION] Only initialize what's needed
    all_schema_features = selected_features if selected_features is not None else FEATURE_SCHEMA['all_features']
    existing_cols = set(df.columns)
    new_cols_to_add = [c for c in all_schema_features if c not in existing_cols]
    
    # [STABILITY] Explicit check for BASE columns required for embeddings (v18.5)
    missing_base = [c for c in Config.EMBED_BASE_COLS if c not in existing_cols]
    if missing_base:
        logger.warning(f"[RELIABILITY_ISSUE] Missing {len(missing_base)} base columns for embeddings: {missing_base}. Filling with 0.0.")
    
    if new_cols_to_add:
        df = pd.concat([df, pd.DataFrame(0.0, index=df.index, columns=new_cols_to_add, dtype='float32')], axis=1)
    
    # 1. Apply Drift Shield (if provided)
    if scaler is not None:
        df = scaler.transform(df, FEATURE_SCHEMA['raw_features'])

    # 2. Populate Latent Features
    logger.info("[LATENT_FEATURES] Populating latent features...")
    
    if reconstructor.pool_embed is None or reconstructor.pool_norm is None:
        raise RuntimeError("[LEAKAGE GUARD] Cache missing. Reconstructor must be loaded from a pre-fitted train artifact.")
    
    # [OPTIMIZATION] Process entire dataframe at once (Internal chunking handles memory)
    # This avoids redundant layout-based looping and slow .loc assignments
    latent_stats = reconstructor.calculate_graph_stats(df)
    
    new_features_df_dict = {}
    for feat_name, values in latent_stats.items():
        if isinstance(values, np.ndarray) and values.ndim > 1:
            # Check if it's supposed to have _d suffixes according to SSOT
            # [SSOT_RULE] MULTI_K based features always have _d suffixes (32 dims)
            if values.shape[1] > 1:
                for d in range(values.shape[1]):
                    col_name = f"{feat_name}_d{d}"
                    if col_name in existing_cols or col_name in new_cols_to_add:
                        new_features_df_dict[col_name] = values[:, d].astype('float32')
            else:
                # (N, 1) case - regime_proxy, local_density, etc.
                # [EXCEPTION] If SSOT expects _d suffix even for (N, 1) latent features
                col_name_d0 = f"{feat_name}_d0"
                if col_name_d0 in existing_cols or col_name_d0 in new_cols_to_add:
                    new_features_df_dict[col_name_d0] = values.ravel().astype('float32')
                elif feat_name in existing_cols or feat_name in new_cols_to_add:
                    new_features_df_dict[feat_name] = values.ravel().astype('float32')
        else:
            col_name = feat_name
            if col_name in existing_cols or col_name in new_cols_to_add:
                val_flat = values.ravel() if isinstance(values, np.ndarray) else values
                new_features_df_dict[col_name] = val_flat.astype('float32')
    
    # Efficiently update the dataframe
    if new_features_df_dict:
        new_feats_df = pd.DataFrame(new_features_df_dict, index=df.index)
        # Update only the columns that were actually calculated
        df.update(new_feats_df)
        del new_feats_df; gc.collect()
                
    return df

def build_features(df, mode='raw', reconstructor=None, residuals=None, raw_preds=None):
    """Legacy entry point for compatibility."""
    df_base = build_base_features(df)
    
    # In legacy mode, we still use the global stats if available for 'raw'
    if mode == 'raw':
        from .utils import assert_artifact_exists
        assert_artifact_exists(Config.GLOBAL_STATS_PATH, "Global Stats Cache")
        stats = GlobalStatStore.load(Config.GLOBAL_STATS_PATH)
        df_base = GlobalStatStore.apply_drift_shield(df_base, stats, FEATURE_SCHEMA['raw_features'])
    
    if mode == 'full' and reconstructor is not None:
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
    return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)

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

    return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)

def load_data():
    train = pd.read_csv(f"{Config.DATA_PATH}train.csv")
    test = pd.read_csv(f"{Config.DATA_PATH}test.csv")
    layout = pd.read_csv(Config.LAYOUT_PATH)
    train = train.merge(layout, on="layout_id", how="left")
    test = test.merge(layout, on="layout_id", how="left")
    return downcast_df(train), downcast_df(test)
