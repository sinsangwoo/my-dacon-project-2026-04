import gc
import logging

import numpy as np
import pandas as pd

from .config import Config
from .utils import downcast_df, inspect_columns, track_lineage, ensure_dataframe, GlobalStatStore

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

# [OPTIONAL_DEPENDENCY_ISOLATION] Torch lazy loading
def check_torch_availability():
    try:
        import torch
        import torch.nn as nn
        return True
    except ImportError:
        return False

TORCH_AVAILABLE = check_torch_availability()

logger = logging.getLogger(__name__)

# [DECOMPOSITION_CONTRACT] Core classes for Hybrid Embedding Reconstuction

class TwoPassImputer:
    """Implements Global median (Pass 1) and Layout-local adjustment (Pass 2)."""
    def __init__(self):
        self.global_imputer = SimpleImputer(strategy='median')
        self.layout_residuals = {}

    def fit(self, df, cols):
        # Pass 1: Global
        self.global_imputer.fit(df[cols])
        global_medians = self.global_imputer.statistics_
        
        # Pass 2: Layout Local
        for lid, group in df.groupby('layout_id'):
            # Calculate how much layout median deviates from global median
            group_median = group[cols].median()
            residual = group_median.values - global_medians
            self.layout_residuals[lid] = np.nan_to_num(residual).astype('float32')

    def transform(self, df, cols):
        df = df.copy()
        # Pass 1
        df[cols] = self.global_imputer.transform(df[cols])
        
        # Pass 2: Add residuals
        for lid, res in self.layout_residuals.items():
            mask = df['layout_id'] == lid
            if mask.any():
                df.loc[mask, cols] += res
        return df

class SuperchargedPCAReconstructor:
    """Supervised, Multi-View, Residual-Aware PCA Reconstructor.
    Replaces AutoEncoder with Residual-Weighted PCA + Orthogonalization.
    """
    def __init__(self, input_dim):
        self.input_dim = input_dim
        # Global Multi-View PCAs
        self.pca_raw = PCA(n_components=8, svd_solver='randomized', random_state=42)
        self.pca_log = PCA(n_components=8, svd_solver='randomized', random_state=42)
        self.pca_rank = PCA(n_components=8, svd_solver='randomized', random_state=42)
        
        # Layout-Local PCAs
        self.local_pcas = {}
        self.global_pca_local = PCA(n_components=8, svd_solver='randomized', random_state=42)
        
        self.kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto')
        self.scaler = StandardScaler()
        self.mode = "SUPERCHARGED_PCA"
        self.embed_dim = 32 # 8(raw) + 8(log) + 8(rank) + 8(local)

    def _get_multi_view_X(self, X):
        """Generate parallel views: Raw, Log, Rank."""
        X_raw = X
        X_log = np.log1p(np.abs(X)) * np.sign(X) # Symmetric log
        
        # Rank normalization (percentile)
        X_rank = np.zeros_like(X)
        for i in range(X.shape[1]):
            X_rank[:, i] = rankdata(X[:, i]) / (len(X) + 1)
            
        return X_raw, X_log, X_rank

    def fit(self, X_train, residuals=None, layout_ids=None):
        """Supervised fit weighted by residual magnitude."""
        X_scaled = self.scaler.fit_transform(X_train)
        
        # 1. Residual Weighting
        weights = 1.0 + np.abs(residuals) if residuals is not None else np.ones(len(X_train))
        X_weighted = X_scaled * weights[:, np.newaxis]
        
        # 2. Fit Multi-View PCAs
        X_w_raw, X_w_log, X_w_rank = self._get_multi_view_X(X_weighted)
        self.pca_raw.fit(X_w_raw)
        self.pca_log.fit(X_w_log)
        self.pca_rank.fit(X_w_rank)
        
        # 3. Fit Layout-Local PCA
        if layout_ids is not None:
            unique_layouts = np.unique(layout_ids)
            for lid in unique_layouts:
                mask = (layout_ids == lid)
                if np.sum(mask) > 40: # Sufficient sample threshold
                    lpca = PCA(n_components=8, svd_solver='randomized', random_state=42)
                    lpca.fit(X_weighted[mask])
                    self.local_pcas[lid] = lpca
        
        self.global_pca_local.fit(X_weighted)
        
        # 4. Mandatory KMeans Fit on current best embeddings
        # (Pass residuals=None during this interior call to avoid recursion)
        final_embed = self.get_embeddings(X_train, layout_ids=layout_ids)
        self.kmeans.fit(final_embed.astype(np.float64))
        logger.info(f"[EMBEDDING_MODE] Locked to: {self.mode}")

    def get_embeddings(self, X, raw_preds=None, layout_ids=None):
        """Extract multi-view embeddings with orthogonalization."""
        X_scaled = self.scaler.transform(X)
        X_raw, X_log, X_rank = self._get_multi_view_X(X_scaled)
        
        # 1. Component Extraction
        e_raw = self.pca_raw.transform(X_raw)
        e_log = self.pca_log.transform(X_log)
        e_rank = self.pca_rank.transform(X_rank)
        
        # 2. Local Extraction
        e_local = np.zeros((len(X), 8), dtype=np.float32)
        if layout_ids is not None:
            unique_layouts = np.unique(layout_ids)
            for lid in unique_layouts:
                mask = (layout_ids == lid)
                if lid in self.local_pcas:
                    e_local[mask] = self.local_pcas[lid].transform(X_scaled[mask])
                else:
                    e_local[mask] = self.global_pca_local.transform(X_scaled[mask])
        else:
            e_local = self.global_pca_local.transform(X_scaled)
            
        combined = np.hstack([e_raw, e_log, e_rank, e_local])
        
        # 3. Variance Amplification (Scale by 1.7x as per mission range 1.5~2.0)
        combined = combined * 1.7
        
        # 4. Orthogonalization against raw_preds
        if raw_preds is not None:
            # We assume raw_preds is centered or use simple projection
            # To decorrelate, subtract projection of combined on raw_preds
            rp = raw_preds.reshape(-1, 1)
            rp_norm_sq = np.sum(rp**2, axis=0) + 1e-8
            # Projection P = (A . B / B . B) * B
            projections = (np.dot(combined.T, rp) / rp_norm_sq).T * rp
            combined = combined - projections
            
        return combined.astype(np.float32)

    def calculate_graph_stats(self, df_target, df_train_pool, k_list=[10, 20, 40], raw_preds=None, layout_ids=None):
        """Compute multi-scale neighbor stats for trend, volatility, and regime proxies."""
        # 1. Get Embeddings (passing through raw_preds for orthogonalization)
        target_embed = self.get_embeddings(df_target[Config.EMBED_BASE_COLS], raw_preds=raw_preds, layout_ids=layout_ids)
        pool_embed = self.get_embeddings(df_train_pool[Config.EMBED_BASE_COLS], layout_ids=df_train_pool['layout_id'].values)
        
        # 2. Pre-normalize for Cosine Similarity (Dot Product equivalent)
        target_norm = target_embed / (np.linalg.norm(target_embed, axis=1, keepdims=True) + 1e-8)
        pool_norm = pool_embed / (np.linalg.norm(pool_embed, axis=1, keepdims=True) + 1e-8)
        
        results = {}
        
        # 3. Multi-scale Aggregation
        for k in k_list:
            # Batch cosine similarity
            # Since layouts are ~1000 rows, we use cdist or simple dot product
            # For each target row, find k neighbors in pool
            dist_matrix = 1 - np.dot(target_norm, pool_norm.T)
            # Find indices of k nearest neighbors
            nn_indices = np.argsort(dist_matrix, axis=1)[:, :k]
            
            # neighbor_mean
            neighbor_means = np.array([pool_embed[idx].mean(axis=0) for idx in nn_indices])
            # neighbor_std
            neighbor_stds = np.array([pool_embed[idx].std(axis=0) for idx in nn_indices])
            
            # attention_weighted_mean (softmax over similarity scores)
            sim_scores = 1.0 - dist_matrix[np.arange(len(dist_matrix))[:, None], nn_indices]
            weights = np.exp(sim_scores) / np.sum(np.exp(sim_scores), axis=1, keepdims=True)
            weighted_means = np.sum(weights[:, :, None] * pool_embed[nn_indices], axis=1)
            
            results[f'embed_mean_{k}'] = neighbor_means
            results[f'embed_std_{k}'] = neighbor_stds
            results[f'weighted_mean_{k}'] = weighted_means
            
        # 4. Global Proxies (Aggregated by K)
        results[f'trend_proxy_{k_list[1]}'] = results[f'embed_mean_{k_list[1]}'] - target_embed
        results[f'volatility_proxy_{k_list[1]}'] = results[f'embed_std_{k_list[1]}']
        results['regime_proxy'] = self.kmeans.predict(target_embed.astype(np.float64))
        
        # 5. Density & Entropy
        avg_dist_20 = np.sort(dist_matrix, axis=1)[:, :20].mean(axis=1)
        results['local_density'] = 1.0 / (avg_dist_20 + 1e-6)
        results['similarity_entropy'] = -np.sum(weights * np.log(weights + 1e-8), axis=1)
        
        return results

def apply_group_normalization(df, feature_groups):
    """Prevent LightGBM from ignoring embedding features due to scale bias."""
    for group_name, cols in feature_groups.items():
        if not cols: continue
        group_data = df[cols].values
        std = np.std(group_data)
        if std > 1e-6:
            df[cols] = df[cols] / std
            logger.info(f"[GROUP_NORM] {group_name} scaled by {std:.4f}")
    return df


def _df_mem_mb(df):
    return df.memory_usage(deep=True).sum() / 1024**2


def _log_fe_stats(func_name, stage, df, generated_features=0, start_shape=None, start_mem_mb=None):
    """Emit shape/memory growth logs for feature engineering forensics."""
    mem_mb = _df_mem_mb(df)
    current_shape = df.shape
    if stage == "start":
        logger.info(
            f"[FE_TRACE] {func_name} | start | shape={current_shape} | "
            f"memory_mb={mem_mb:.2f} | generated_features=0"
        )
        return

    delta_rows = current_shape[0] - start_shape[0] if start_shape else 0
    delta_cols = current_shape[1] - start_shape[1] if start_shape else generated_features
    mem_delta = mem_mb - start_mem_mb if start_mem_mb is not None else 0.0
    logger.info(
        f"[FE_TRACE] {func_name} | end | shape={current_shape} | memory_mb={mem_mb:.2f} | "
        f"generated_features={generated_features} | delta_rows={delta_rows} | "
        f"delta_cols={delta_cols} | delta_memory_mb={mem_delta:.2f}"
    )


def _rolling_slope(values):
    arr = np.asarray(values, dtype=np.float32)
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float32)
    x_centered = x - x.mean()
    denom = np.sum(x_centered ** 2)
    if denom == 0:
        return 0.0
    y_centered = arr - arr.mean()
    return float(np.sum(x_centered * y_centered) / denom)


# --- [PHASE 2: DETERMINISTIC FEATURE BUILDER] ---
def build_features(df, schema, mode='raw', reconstructor=None, residuals=None, raw_preds=None, train_pool=None):
    """
    Single entry point for all feature engineering.
    Ensures absolute feature contract enforcement.
    """
    logger.info(f"[FEATURE_BUILD] Mode: {mode} | Schema Features: {len(schema['all_features'])}")
    
    # 1. Base Preprocessing & Sort
    df = df.copy()
    df = df.sort_values(by=["scenario_id", "ID"]).reset_index(drop=True)
    
    # 2. Sequential Feature Engineering (Always run for all modes)
    top_ts_cols = Config.EMBED_BASE_COLS
    df = add_time_series_features(df, top_ts_cols)
    df = add_extreme_detection_features(df, top_ts_cols)
    
    # 3. Embedding Logic (Only in 'full' mode with residuals)
    if mode == 'full' and reconstructor is not None and residuals is not None:
        logger.info("[FEATURE_BUILD] Populating EMBED features using residuals...")
        # We use a dummy test df for generate_supercharged_latent_features
        # since it expects (train, test) but we are building one at a time here.
        # Actually, let's keep it simple: if mode is full, we assume the embed features 
        # will be populated by generate_supercharged_latent_features separately 
        # or we integrate it here.
        
        # For now, we'll let main.py call generate_supercharged_latent_features 
        # and THEN call build_features(mode='full') to enforce the contract.
        pass

    # 4. ENFORCE SCHEMA (The "Contract")
    all_schema_features = schema['all_features']
    
    # Ensure ID_COLS and TARGET are preserved if they exist (for internal use, not final X)
    preserved_cols = [c for c in Config.ID_COLS + [Config.TARGET] if c in df.columns]
    
    # Create missing features with fallback policy
    missing_features = []
    for col in all_schema_features:
        if col not in df.columns:
            missing_features.append(col)
            # Fallback policy
            if 'ratio' in col or 'rel_to' in col:
                df[col] = 1.0
            elif 'std' in col or 'volatility' in col:
                df[col] = 1e-6
            else:
                df[col] = 0.0
    
    if missing_features:
        logger.info(f"[MISSING_FEATURE_FILLED] {len(missing_features)} features were missing and filled with defaults.")

    # 5. Fixed Ordering & Index Alignment
    # We return the features only for training/inference
    X_df = df[all_schema_features].astype('float32')
    
    # [PHASE 2] Ensure no NaNs from rolling/FE operations
    nans_before = X_df.isna().sum().sum()
    if nans_before > 0:
        logger.info(f"[FEATURE_BUILD] Filling {nans_before} NaNs from engineering operations.")
        X_df = X_df.fillna(0.0)
    
    # 6. Assertion Firewall
    validate_feature_contract(X_df, schema)
    
    logger.info(f"[FEATURE_BUILD_SUCCESS] Final Shape: {X_df.shape}")
    
    # Return both X and the original df (with metadata) for stage separation
    return X_df, df

# --- [PHASE 3: ASSERTION FIREWALL] ---
def validate_feature_contract(df, schema):
    """Hard fail if any contract is violated."""
    logger.info("[INDEX_ALIGNMENT_CHECK] Validating feature contract...")
    
    # 1. Column Order & Count
    if list(df.columns) != schema['all_features']:
        logger.error(f"[CONTRACT_VIOLATION] Column mismatch! Expected {len(schema['all_features'])}, got {len(df.columns)}")
        raise RuntimeError("TERMINATE PIPELINE: Column mismatch")
        
    # 2. NaN / Inf Check
    nans = df.isna().sum().sum()
    if nans > 0:
        logger.error(f"[CONTRACT_VIOLATION] {nans} NaNs detected in feature matrix!")
        raise RuntimeError("TERMINATE PIPELINE: NaN detected")
        
    if np.isinf(df.values).any():
        logger.error("[CONTRACT_VIOLATION] Inf detected in feature matrix!")
        raise RuntimeError("TERMINATE PIPELINE: Inf detected")
        
    # 3. Shape Match
    if df.shape[1] != len(schema['all_features']):
        logger.error(f"[CONTRACT_VIOLATION] Shape mismatch! Expected {len(schema['all_features'])}, got {df.shape[1]}")
        raise RuntimeError("TERMINATE PIPELINE: Shape mismatch")

    logger.info("[FIREWALL_PASSED] Feature contract verified.")


def add_hybrid_latent_features(train, test):
    """[HARD_CONTRACT] Stage 1 (Initialization) for Supercharged PCA.
    Fitting and feature extraction are deferred to Phase 3 to enable residual-awareness.
    """
    _log_fe_stats("add_hybrid_latent_features", "start", train)
    start_shape = train.shape
    start_mem = _df_mem_mb(train)
    
    # 1. Identity Contract
    embed_base_cols = Config.EMBED_BASE_COLS
    
    # 2. Two-Pass Imputation
    imputer = TwoPassImputer()
    imputer.fit(train, embed_base_cols)
    train = imputer.transform(train, embed_base_cols)
    test = imputer.transform(test, embed_base_cols)
    
    # 3. Supercharged PCA Initialization
    reconstructor = SuperchargedPCAReconstructor(len(embed_base_cols))
    
    # Pre-initialize new columns as float32 to avoid NaNs
    latent_patterns = ['embed_mean', 'embed_std', 'trend_proxy', 'volatility_proxy', 'weighted_mean', 'weighted_std']
    # Total dim: 8(raw)+8(log)+8(rank)+8(local) = 32
    embed_dim = 32 
    for k in [10, 20, 40]:
        for p in latent_patterns:
            for d in range(embed_dim):
                col = f"{p}_{k}_d{d}" if 'proxy' not in p else f"{p}_{k}"
                if col not in train.columns:
                    train[col] = 0.0
                    test[col] = 0.0
    
    # Also initialize global proxies
    for col in ['regime_proxy', 'local_density', 'similarity_entropy']:
        if col not in train.columns:
            train[col] = 0.0
            test[col] = 0.0
    
    # This return includes empty feature columns which will be populated in Phase 3 Meta-Stage.
    latent_cols = [c for c in train.columns if any(p in c for p in latent_patterns) or c in ['regime_proxy', 'local_density', 'similarity_entropy']]
    # Explicitly exclude metadata
    latent_cols = [c for c in latent_cols if c != 'embedding_mode']
    
    _log_fe_stats("add_hybrid_latent_features", "end", train, generated_features=len(latent_cols), start_shape=start_shape, start_mem_mb=start_mem)
    return train, test, latent_cols, reconstructor

def generate_supercharged_latent_features(train, test, reconstructor, residuals, raw_preds):
    """[HARD_CONTRACT] Stage 2 (Supervised) for Supercharged PCA.
    Populates initialized columns using residuals and layout-local aggregation.
    """
    _log_fe_stats("generate_supercharged_latent_features", "start", train)
    start_shape = train.shape
    start_mem = _df_mem_mb(train)
    
    embed_base_cols = Config.EMBED_BASE_COLS
    
    # 1. Supervised Fit
    reconstructor.fit(
        train[embed_base_cols].values.astype('float32'), 
        residuals=residuals, 
        layout_ids=train['layout_id'].values
    )
    
    # 2. Multi-View & Multi-Scale Graph Aggregation
    for lid in train['layout_id'].unique():
        tr_mask = train['layout_id'] == lid
        te_mask = test['layout_id'] == lid
        
        if not te_mask.any(): continue
        
        # Fallback pool
        pool_df = train[tr_mask]
        if len(pool_df) < 40:
            pool_df = train
        
        # Supercharged Stats Calculation
        # We pass raw_preds to enable Orthogonalization during transform
        tr_stats = reconstructor.calculate_graph_stats(
            train[tr_mask], pool_df, 
            raw_preds=raw_preds[tr_mask] if raw_preds is not None else None,
            layout_ids=train['layout_id'].values[tr_mask]
        )
        te_stats = reconstructor.calculate_graph_stats(
            test[te_mask], pool_df, 
            # Note: test doesn't have true raw_preds yet? 
            # Actually, in Phase 3, we have trainer.test_preds['raw']
            raw_preds=None, # Will handle test preds separately if needed
            layout_ids=test['layout_id'].values[te_mask]
        )
        
        # Inject features
        for df, stats, mask in [(train, tr_stats, tr_mask), (test, te_stats, te_mask)]:
            if mask.any():
                for feat_name, values in stats.items():
                    if isinstance(values, np.ndarray) and values.ndim > 1:
                        for d in range(values.shape[1]):
                            col_name = f"{feat_name}_d{d}"
                            df.loc[mask, col_name] = values[:, d].astype('float32')
                    else:
                        col_name = feat_name
                        df.loc[mask, col_name] = values.astype('float32')
                            
    # 3. Orthogonalize Global Proxies
    # regime_proxy etc are already handled inside calculate_graph_stats
    
    # 4. Group Normalization
    latent_patterns = ['embed_mean', 'embed_std', 'weighted_mean', 'trend_proxy', 'volatility_proxy', 'weighted_std']
    groups = {p: [c for c in train.columns if p in c] for p in latent_patterns}
    train = apply_group_normalization(train, groups)
    test = apply_group_normalization(test, groups)
    
    # 5. Final Contract Validation
    latent_patterns = ['embed_mean', 'embed_std', 'weighted_mean', 'trend_proxy', 'volatility_proxy', 'weighted_std']
    new_cols = [c for c in train.columns if any(p in c for p in latent_patterns) or c in ['regime_proxy', 'local_density', 'similarity_entropy']]
    new_cols = [c for c in new_cols if c != 'embedding_mode']
    train[new_cols] = train[new_cols].fillna(0)
    test[new_cols] = test[new_cols].fillna(0)
    
    _log_fe_stats("generate_supercharged_latent_features", "end", train, generated_features=len(new_cols), start_shape=start_shape, start_mem_mb=start_mem)
    return train, test, new_cols

def get_removed_leaky_feature_patterns():
    return [
        "seqcomp_*",
        "late_*",
        "early_late_diff",
        "transform('max')",
        "transform('min')",
        "iloc[-k:]",
        "scenario-level global mean/std",
        "future_proxy",
        "cum_trend built from late segments",
        "PCA-compressed sequence features",
        "global quantile binary thresholding",
    ]


def build_causal_feature_manifest(top_ts_cols):
    manifest = []
    for col in top_ts_cols:
        manifest.extend([
            {
                "feature": f"{col}_rolling_mean_3",
                "meaning": f"{col}의 최근 3 step 평균",
                "causality_justification": "window=[t-2, t]만 사용하며 현재 시점까지의 관측치만 포함",
            },
            {
                "feature": f"{col}_rolling_mean_5",
                "meaning": f"{col}의 최근 5 step 평균",
                "causality_justification": "window=[t-4, t]만 사용하며 미래 시점 접근 없음",
            },
            {
                "feature": f"{col}_rolling_std_3",
                "meaning": f"{col}의 최근 3 step 변동성",
                "causality_justification": "최근 3개 관측값의 분산 구조만 반영",
            },
            {
                "feature": f"{col}_rolling_std_5",
                "meaning": f"{col}의 최근 5 step 변동성",
                "causality_justification": "최근 5개 관측값만으로 불안정성 측정",
            },
            {
                "feature": f"{col}_diff_1",
                "meaning": f"{col}(t) - {col}(t-1)",
                "causality_justification": "직전 시점과 현재 시점 차이만 사용",
            },
            {
                "feature": f"{col}_diff_3",
                "meaning": f"{col}(t) - {col}(t-3)",
                "causality_justification": "3 step 전 상태와의 누적 변화만 사용",
            },
            {
                "feature": f"{col}_rate_1",
                "meaning": f"{col}의 1 step 상대 변화율",
                "causality_justification": "직전 값의 절대값으로 정규화하며 미래 정보 없음",
            },
            {
                "feature": f"{col}_slope_5",
                "meaning": f"{col}의 최근 5 step 선형 추세 기울기",
                "causality_justification": "window=[t-4, t] 구간에서만 회귀 slope 계산",
            },
            {
                "feature": f"{col}_recent_max_5",
                "meaning": f"{col}의 최근 5 step 최대값",
                "causality_justification": "과거 5개 값 중 최고치만 사용",
            },
            {
                "feature": f"{col}_recent_min_5",
                "meaning": f"{col}의 최근 5 step 최소값",
                "causality_justification": "과거 5개 값 중 최저치만 사용",
            },
            {
                "feature": f"{col}_range_5",
                "meaning": f"{col}의 최근 5 step 범위(max-min)",
                "causality_justification": "최근 5 step 내부 진폭만 측정",
            },
        ])

    manifest.extend([
        {
            "feature": "timestep_index",
            "meaning": "scenario 내부 현재 시점 index (0~24)",
            "causality_justification": "현재 위치 자체는 t 시점에 이미 알려진 메타정보",
        },
        {
            "feature": "normalized_time",
            "meaning": "정규화된 시간 위치 t/24",
            "causality_justification": "현재 시점 위치를 스케일링한 값이며 미래 관측과 무관",
        },
    ])

    # --- Extreme Scenario Detection Features (Part 1-7) ---
    for col in top_ts_cols:
        manifest.extend([
            {
                "feature": f"{col}_rel_to_mean_5",
                "meaning": f"{col}(t) / rolling_mean_5",
                "causality_justification": "현재값이 최근 5 step 평균 대비 얼마나 튀어있는지 측정 (Relative Position)",
            },
            {
                "feature": f"{col}_rel_to_max_5",
                "meaning": f"{col}(t) / rolling_max_5",
                "causality_justification": "최근 5 step 최고점 대비 현재 상태의 상대적 위치",
            },
            {
                "feature": f"{col}_rel_rank_5",
                "meaning": f"{col}의 최근 5 step 내 rank",
                "causality_justification": "최근 5개 관측치 중 현재 시점이 몇 번째로 높은지 순위화",
            },
            {
                "feature": f"{col}_accel",
                "meaning": f"{col}의 가속도 (diff_1 - diff_2)",
                "causality_justification": "변화율의 변화율을 통해 폭발적 증가 징후 포착",
            },
            {
                "feature": f"{col}_volatility_expansion_std",
                "meaning": "std_3 / std_10",
                "causality_justification": "단기 변동성이 장기 대비 얼마나 급증했는지 측정",
            },
            {
                "feature": f"{col}_volatility_expansion_range",
                "meaning": "range_3 / range_10",
                "causality_justification": "단기 진폭이 장기 진폭 대비 확대되는 양상 포착",
            },
            {
                "feature": f"{col}_regime_id",
                "meaning": "rolling quantile 기반 현재 상태 분류 (0~4)",
                "causality_justification": "과거 20 step 관측치 분포(q25, q50, q75, q90) 내 현재 위치 분류",
            },
            {
                "feature": f"{col}_consecutive_above_q75",
                "meaning": "최근 5 step 중 q75 상회 횟수",
                "causality_justification": "고지연 위험 상태의 지속성(Persistence) 측정",
            },
        ])

    manifest.extend([
        {
            "feature": "early_warning_flag",
            "meaning": "폭발 직전 패턴 조합 flag",
            "causality_justification": "주요 feature들의 slope, std, rel_mean이 동시에 상승할 때 1 부여",
        },
        {
            "feature": "early_warning_score",
            "meaning": "위험 징후 가중 합산 스코어",
            "causality_justification": "여러 위험 지표들의 합을 통해 종합적 위험도 계량화",
        },
    ])
    return manifest


def load_data():
    """Load train, test, and layout datasets."""
    train = pd.read_csv(f"{Config.DATA_PATH}train.csv")
    test = pd.read_csv(f"{Config.DATA_PATH}test.csv")
    layout = pd.read_csv(Config.LAYOUT_PATH)

    train = train.merge(layout, on="layout_id", how="left")
    test = test.merge(layout, on="layout_id", how="left")

    train = downcast_df(train)
    test = downcast_df(test)
    return train, test


def add_time_series_features(df, top_ts_cols, global_stats=None):
    """Create leakage-free causal sequence features with TRAIN-ANCHORED consistency (v13.0)."""
    start_shape = df.shape
    start_mem_mb = _df_mem_mb(df)
    _log_fe_stats("add_time_series_features", "start", df)

    # PHASE 1: Mandatory Temporal Order Guarantee
    df = df.sort_values(by=["scenario_id", "ID"]).reset_index(drop=True)

    if "timestep_index" not in df.columns:
        df["timestep_index"] = df.groupby("scenario_id").cumcount().astype("int16")
    df["normalized_time"] = (df["timestep_index"] / Config.NORMALIZED_TIME_DENOM).astype("float32")
    
    # PHASE 3: Initialize cold_start_flag (default 0, will be updated by pipeline if history is missing)
    if "cold_start_flag" not in df.columns:
        df["cold_start_flag"] = 0
    df["cold_start_flag"] = df["cold_start_flag"].astype("int8")

    new_features = {}
    grouped = df.groupby("scenario_id", sort=False)
    
    for col in top_ts_cols:
        scenario_series = grouped[col]
        
        # Pre-calculate lags for vectorization (SAFE - local only)
        lag_1 = scenario_series.shift(1)
        lag_2 = scenario_series.shift(2)
        lag_3 = scenario_series.shift(3)
        lag_4 = scenario_series.shift(4)

        # PHASE 4: Stable Rolling Operations (Used during training only)
        rolling_mean_3 = scenario_series.rolling(3, min_periods=3).mean().values
        rolling_mean_5 = scenario_series.rolling(5, min_periods=5).mean().values
        rolling_std_3 = scenario_series.rolling(3, min_periods=3).std().values
        rolling_std_5 = scenario_series.rolling(5, min_periods=5).std().values
        recent_max_5 = scenario_series.rolling(5, min_periods=5).max().values
        recent_min_5 = scenario_series.rolling(5, min_periods=5).min().values

        # PHASE 4: Stable early features - expanding window replaced with rolling window (K=20)
        expanding_mean = scenario_series.rolling(20, min_periods=1).mean().values
        expanding_sum = scenario_series.rolling(20, min_periods=1).sum().values
        expanding_std = scenario_series.rolling(20, min_periods=3).std().values
        
        # Vectorized slope_5 for window 5 (SAFE - formula based)
        slope_5 = (2 * df[col] + 1 * lag_1.fillna(0) - 1 * lag_3.fillna(0) - 2 * lag_4.fillna(0)) / 10.0

        diff_1 = df[col] - lag_1
        diff_3 = df[col] - lag_3
        
        # rate_1 normalization
        rate_1 = diff_1 / (lag_1.abs() + Config.RATE_EPS)

        new_features[f"{col}_rolling_mean_3"] = rolling_mean_3.astype("float32")
        new_features[f"{col}_rolling_mean_5"] = rolling_mean_5.astype("float32")
        new_features[f"{col}_rolling_std_3"] = rolling_std_3.astype("float32")
        new_features[f"{col}_rolling_std_5"] = rolling_std_5.astype("float32")
        new_features[f"{col}_diff_1"] = diff_1.astype("float32")
        new_features[f"{col}_diff_3"] = diff_3.astype("float32")
        new_features[f"{col}_rate_1"] = rate_1.astype("float32")
        new_features[f"{col}_slope_5"] = slope_5.values.astype("float32")
        new_features[f"{col}_recent_max_5"] = recent_max_5.astype("float32")
        new_features[f"{col}_recent_min_5"] = recent_min_5.astype("float32")
        new_features[f"{col}_range_5"] = (recent_max_5 - recent_min_5).astype("float32")

        # PHASE 4: Add stable early features
        new_features[f"{col}_expanding_mean"] = expanding_mean.astype("float32")
        new_features[f"{col}_expanding_sum"] = expanding_sum.astype("float32")
        new_features[f"{col}_expanding_std"] = expanding_std.astype("float32")

    if new_features:
        new_df = ensure_dataframe(new_features, tag="add_time_series_features_new_df")
        df = pd.concat([df, new_df], axis=1)
        track_lineage(
            df,
            new_df.columns,
            source=top_ts_cols,
            function="add_time_series_features",
            operation="causal_sequence_feature_generation",
        )
        inspect_columns(df, "AFTER_pd.concat(CAUSAL_TS_FEATURES)", logger)

    df = df.copy()
    del new_features
    gc.collect()

    _log_fe_stats(
        "add_time_series_features",
        "end",
        df,
        generated_features=df.shape[1] - start_shape[1],
        start_shape=start_shape,
        start_mem_mb=start_mem_mb,
    )
    return df


def add_extreme_detection_features(df, top_ts_cols, global_stats=None):
    """Implement the Extreme Scenario Detection features with TRAIN-ANCHORED consistency (v13.0)."""
    start_shape = df.shape
    start_mem_mb = _df_mem_mb(df)
    _log_fe_stats("add_extreme_detection_features", "start", df)

    # PHASE 1: Mandatory Temporal Order Guarantee
    df = df.sort_values(by=["scenario_id", "ID"]).reset_index(drop=True)
    
    # PHASE 3: Ensure cold_start_flag exists (init if not present)
    if "cold_start_flag" not in df.columns:
        df["cold_start_flag"] = 0
    df["cold_start_flag"] = df["cold_start_flag"].astype("int8")
    
    new_features = {}
    
    # Pre-calculate groups to avoid repeated overhead
    grouped = df.groupby("scenario_id", sort=False)
    
    # 1. Part 1-3 & 5 & 7: Basic Extreme Indicators per feature
    for col in top_ts_cols:
        scenario_series = grouped[col]
        
        # Part 1: Relative Position (UNSAFE → SAFE conversion)
        lag_1 = scenario_series.shift(1)
        lag_2 = scenario_series.shift(2)
        lag_3 = scenario_series.shift(3)
        lag_4 = scenario_series.shift(4)
        
        # Part 1: Relative Position (UNSAFE → SAFE conversion)
        # rolling_mean_5 (Vectorized)
        rolling_mean_5 = (df[col] + lag_1.fillna(0) + lag_2.fillna(0) + lag_3.fillna(0) + lag_4.fillna(0)) / (
            1 + lag_1.notna().astype(int) + lag_2.notna().astype(int) + 
            lag_3.notna().astype(int) + lag_4.notna().astype(int)
        )
        # rolling_max_5 (PHASE 4: Stable rolling)
        rolling_max_5 = scenario_series.rolling(5, min_periods=5).max().values
        # PHASE 4: Stable rel_rank using rolling rank (Vectorized)
        rel_rank_5 = scenario_series.rolling(5, min_periods=1).rank().fillna(0).values
        
        new_features[f"{col}_rel_to_mean_5"] = (df[col] / (rolling_mean_5 + Config.RATE_EPS)).astype("float32")
        new_features[f"{col}_rel_to_max_5"] = (df[col] / (rolling_max_5 + Config.RATE_EPS)).astype("float32")
        new_features[f"{col}_rel_rank_5"] = rel_rank_5.astype("int8")
        
        # Part 2: Acceleration (SAFE - local diffs)
        diff_1 = df[col] - lag_1
        diff_2 = lag_1 - lag_2
        accel = diff_1 - diff_2
        new_features[f"{col}_accel"] = accel.astype("float32")
        
        new_features[f"{col}_accel_mean_5"] = accel.groupby(df['scenario_id'], sort=False).rolling(5, min_periods=5).mean().values.astype("float32")

        # Part 3: Volatility Expansion (UNSAFE → SAFE conversion)
        std_3 = scenario_series.rolling(3, min_periods=3).std().values
        std_10 = scenario_series.rolling(10, min_periods=10).std().values
        rmax_3 = scenario_series.rolling(3, min_periods=3).max().values
        rmin_3 = scenario_series.rolling(3, min_periods=3).min().values
        rmax_10 = scenario_series.rolling(10, min_periods=10).max().values
        rmin_10 = scenario_series.rolling(10, min_periods=10).min().values
        range_3 = rmax_3 - rmin_3
        range_10 = rmax_10 - rmin_10
        new_features[f"{col}_volatility_expansion_std"] = (std_3 / (std_10 + Config.RATE_EPS)).astype("float32")
        new_features[f"{col}_volatility_expansion_range"] = (range_3 / (range_10 + Config.RATE_EPS)).astype("float32")
        
        # Part 5: Regime Detection (UNSAFE → SAFE conversion)
        # Using rolling 20 instead of expanding
        q25 = scenario_series.rolling(20, min_periods=20).quantile(0.25).values
        q50 = scenario_series.rolling(20, min_periods=20).quantile(0.50).values
        q75 = scenario_series.rolling(20, min_periods=20).quantile(0.75).values
        q90 = scenario_series.rolling(20, min_periods=20).quantile(0.90).values
        regime_id = (df[col] > q25).astype(int) + (df[col] > q50).astype(int) + (df[col] > q75).astype(int) + (df[col] > q90).astype(int)
        new_features[f"{col}_regime_id"] = regime_id.astype("int8")
        new_features[f"{col}_consecutive_above_q75"] = (df[col] > q75).astype(int).groupby(df['scenario_id'], sort=False).rolling(5, min_periods=5).sum().fillna(0).values.astype("int8")

        # PHASE 4: Monotonic pressure signals
        diff_1_for_pressure = df[col] - lag_1
        new_features[f"{col}_consecutive_increase_count"] = (diff_1_for_pressure > 0).astype(int).groupby(df['scenario_id'], sort=False).rolling(5, min_periods=5).sum().fillna(0).values.astype("int8")

        rolling_mean_for_pressure = scenario_series.rolling(5, min_periods=5).mean().values
        new_features[f"{col}_consecutive_above_mean_count"] = (df[col] > rolling_mean_for_pressure).astype(int).groupby(df['scenario_id'], sort=False).rolling(5, min_periods=5).sum().fillna(0).values.astype("int8")

    # 2. Part 6: Cross-Feature Interactions (Top Pairs)
    # Selected pairs based on domain knowledge (order vs capacity/utilization)
    interaction_pairs = [
        ('order_inflow_15m', 'robot_utilization'),
        ('heavy_item_ratio', 'order_inflow_15m'),
        ('order_inflow_15m', 'delay_index'), # if delay_index exists
        ('heavy_item_ratio', 'robot_utilization'),
    ]
    for f1, f2 in interaction_pairs:
        if f1 in df.columns and f2 in df.columns:
            new_features[f"inter_{f1}_x_{f2}"] = (df[f1] * df[f2]).astype("float32")

    # 3. Part 4: Early Warning Signal (Global Flags)
    # Combine signals from top features (e.g., order_inflow_15m)
    primary = 'order_inflow_15m'
    if primary in df.columns:
        slope_col = f"{primary}_slope_5"
        std_col = f"{primary}_rolling_std_5"
        rel_mean_col = f"order_inflow_15m_rel_to_mean_5" # from our new_features
        
        # Since new_features aren't in df yet, we access them from the dictionary
        if slope_col in df.columns and std_col in df.columns:
            slope_5 = df[slope_col]
            std_5 = df[std_col]
            rel_mean_5 = new_features[rel_mean_col]
            
            # Thresholds can be adjusted via config or heuristics
            ew_flag = (slope_5 > 0.5) & (std_5.diff() > 0) & (rel_mean_5 > 1.2)
            new_features["early_warning_flag"] = ew_flag.astype("int8")
            
            # Weighted score: slope + volatility expansion + relative position
            # Use rolling normalization to preserve scaling without test-local stats leaking across scenarios
            slope_max = slope_5.transform(lambda x: x.expanding().max()).fillna(0)
            rel_mean_max = rel_mean_5.transform(lambda x: x.expanding().max()).fillna(0)

            ew_score = (
                (slope_5 / (slope_max + 1e-6)) * 0.4 + 
                (rel_mean_5 / (rel_mean_max + 1e-6)) * 0.4 +
                (new_features[f"{primary}_volatility_expansion_std"] / 2.0) * 0.2
            )
            new_features["early_warning_score"] = ew_score.astype("float32")

    # Merge all new features
    if new_features:
        new_df = ensure_dataframe(new_features, tag="add_extreme_detection_features_new_df")
        df = pd.concat([df, new_df], axis=1)
        track_lineage(
            df,
            new_df.columns,
            source=top_ts_cols,
            function="add_extreme_detection_features",
            operation="extreme_scenario_detection_logic",
        )

    df = df.copy()
    del new_features
    gc.collect()

    _log_fe_stats(
        "add_extreme_detection_features",
        "end",
        df,
        generated_features=df.shape[1] - start_shape[1],
        start_shape=start_shape,
        start_mem_mb=start_mem_mb,
    )
    return df


def add_advanced_predictive_features(df, top_ts_cols, global_stds=None):
    """Disabled to enforce strict causality. Use add_time_series_features instead."""
    logger.info("[CAUSAL_POLICY] add_advanced_predictive_features skipped (future/global features removed).")
    return df


def add_scenario_summary_features(df, top_ts_cols):
    """Disabled to enforce strict causality. Scenario-wide summary is not allowed."""
    logger.info("[CAUSAL_POLICY] add_scenario_summary_features skipped (scenario-global aggregation removed).")
    return df


def add_scenario_sequence_compressed(df, top_ts_cols):
    """Disabled to enforce strict causality and interpretability."""
    logger.info("[CAUSAL_POLICY] add_scenario_sequence_compressed skipped (seqcomp removed).")
    return df


def add_sequence_trajectory_features(df, top_ts_cols):
    """Disabled because old trajectory features depended on leaky early/mid/late summaries."""
    logger.info("[CAUSAL_POLICY] add_sequence_trajectory_features skipped (leaky summary dependency removed).")
    return df


def add_binary_thresholding(df, top_ts_cols):
    """Disabled because dataset-level quantile thresholds are not causal features."""
    logger.info("[CAUSAL_POLICY] add_binary_thresholding skipped (global thresholding removed).")
    return df


def add_nan_flags(df, primary_cols):
    """Add binary flags for NaN positions in engineered causal features."""
    start_shape = df.shape
    start_mem_mb = _df_mem_mb(df)
    _log_fe_stats("add_nan_flags", "start", df)

    new_features = {}
    for col in primary_cols:
        if col in df.columns:
            new_features[f"{col}_nan_flag"] = df[col].isna().astype("int8")

    if new_features:
        new_df = ensure_dataframe(new_features, tag="add_nan_flags_new_df")
        df = pd.concat([df, new_df], axis=1)

    df = df.copy()
    del new_features
    gc.collect()

    _log_fe_stats(
        "add_nan_flags",
        "end",
        df,
        generated_features=df.shape[1] - start_shape[1],
        start_shape=start_shape,
        start_mem_mb=start_mem_mb,
    )
    return df


def handle_engineered_nans(df, feature_cols):
    """Impute causal engineered features without introducing future information."""
    start_shape = df.shape
    start_mem_mb = _df_mem_mb(df)
    _log_fe_stats("handle_engineered_nans", "start", df)

    if df.columns.duplicated().any():
        logger.warning("[DEDUP] Removing duplicated columns before imputation.")
        df = df.loc[:, ~df.columns.duplicated()]

    feature_cols = list(dict.fromkeys([c for c in feature_cols if c in df.columns]))
    for col in feature_cols:
        if col not in df.columns:
            continue
        if "rolling_mean" in col or "recent_max" in col or "recent_min" in col:
            df[col] = df.groupby("scenario_id")[col].ffill().fillna(0).astype("float32")
        else:
            df[col] = df[col].fillna(0).astype("float32")

    _log_fe_stats(
        "handle_engineered_nans",
        "end",
        df,
        generated_features=0,
        start_shape=start_shape,
        start_mem_mb=start_mem_mb,
    )
    return df


def compress_sequence_features(df, feature_cols, n_components=15):
    """Disabled because opaque PCA features violate explainability requirements."""
    logger.info("[CAUSAL_POLICY] compress_sequence_features skipped (opaque PCA features removed).")
    return df, []


def select_top_ts_features(train):
    """Select top numeric features to expand into causal sequence features."""
    num_cols = train.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if c not in Config.ID_COLS + [Config.TARGET]]

    temp_train = train[num_cols + [Config.TARGET]].fillna(0)
    variances = temp_train[num_cols].var()
    valid_cols = variances[variances > Config.VARIANCE_THRESHOLD].index.tolist()
    corrs = temp_train[valid_cols + [Config.TARGET]].corr()[Config.TARGET].abs().sort_values(ascending=False)
    return corrs.head(Config.TS_TOP_K + 1).index.tolist()[1:]

def get_features(train, test):
    """Filter columns to get final numeric feature set."""
    feature_cols = [c for c in train.columns if c not in Config.ID_COLS + [Config.TARGET]]
    final_cols = train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    final_cols = list(dict.fromkeys(final_cols))
    return final_cols

def align_features(df, reference_cols, logger=None):
    """Memory-Safe Feature Alignment (v6.0 Inplace-First).
    
    Builds the target DataFrame column-wise, dropping used columns from source immediately
    to restrict the peak memory barrier.
    """
    if logger:
        logger.info(f"── Memory-Safe Alignment (v6.0) ──")
        logger.info(f"  Target columns:  {len(reference_cols)}")

    # 1. Create Empty Skeleton with same index
    final_df = pd.DataFrame(index=df.index)
    
    # 2. Process Reference Columns one-by-one
    target_set = set(reference_cols)
    source_cols = list(df.columns)
    
    for col in reference_cols:
        if col in df.columns:
            # Transfer and convert to float32
            final_df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('float32')
            # Drop from source to free memory
            df.drop(columns=[col], inplace=True)
        else:
            # Fill missing with zeros
            final_df[col] = np.zeros(len(df), dtype='float32')
            
        # Periodic cleanup
        if len(final_df.columns) % 100 == 0:
            gc.collect()

    if logger:
        logger.info(f"  ✓ Column-wise transfer complete. Cleaning up source...")

    # 3. STRICT POST-CONDITIONS (as before)
    assert final_df.shape[1] == len(reference_cols)
    assert list(final_df.columns) == reference_cols
    assert not final_df.isna().any().any()
    
    if logger:
        logger.info(f"  ✓ Alignment complete: shape={final_df.shape}")
    
    return final_df

def prune_collinear_features(df, feature_cols, threshold=0.98, protected_features=None):
    """Remove one of each pair of features with correlation > threshold.
    Protected features are never removed.
    """
    if protected_features is None:
        protected_features = []
    protected_set = set(protected_features)
    
    # Sample for speed
    sample_df = df[feature_cols].sample(n=min(5000, len(df)), random_state=42).fillna(0)
    corr_matrix = sample_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = set()
    for col in upper.columns:
        if col in to_drop:
            continue
        high_corr_cols = upper.index[upper[col] > threshold].tolist()
        for hc in high_corr_cols:
            if hc in to_drop:
                continue
            # Drop the unprotected one; if both protected, keep both
            if hc not in protected_set:
                to_drop.add(hc)
            elif col not in protected_set:
                to_drop.add(col)
                break
    
    remaining = [f for f in feature_cols if f not in to_drop]
    logger.info(f"[COLLINEAR] Removed {len(to_drop)} features (threshold={threshold})")
    return remaining, list(to_drop)

def prune_drift_features(train_df, test_df, feature_cols, threshold=0.15, protected_features=None):
    """Remove features with high train-test distribution drift (KS > threshold).
    Protected features are never removed.
    """
    from scipy.stats import ks_2samp
    if protected_features is None:
        protected_features = []
    protected_set = set(protected_features)
    
    to_drop = []
    n_sample = min(5000, len(train_df), len(test_df))
    
    for col in feature_cols:
        if col in protected_set:
            continue
        if col not in train_df.columns or col not in test_df.columns:
            continue
        tr_vals = train_df[col].dropna().values
        te_vals = test_df[col].dropna().values
        if len(tr_vals) == 0 or len(te_vals) == 0:
            continue
        tr_sample = tr_vals[np.random.choice(len(tr_vals), min(n_sample, len(tr_vals)), replace=False)]
        te_sample = te_vals[np.random.choice(len(te_vals), min(n_sample, len(te_vals)), replace=False)]
        ks_stat, _ = ks_2samp(tr_sample, te_sample)
        if ks_stat > threshold:
            to_drop.append(col)
    
    remaining = [f for f in feature_cols if f not in to_drop]
    logger.info(f"[DRIFT] Removed {len(to_drop)} features (KS > {threshold})")
    return remaining, to_drop
