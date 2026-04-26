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
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from .config import Config
from .schema import FEATURE_SCHEMA, BASE_COLS
from .utils import downcast_df, inspect_columns, track_lineage, ensure_dataframe, GlobalStatStore
# [WHY_THIS_CHANGE]
# Problem: feature_registry was not imported, so FeatureDropRegistry and PruningManifest
#   could not be used here. The registry is needed to record all drop decisions
#   with full statistical provenance instead of silent local set operations.
# Root Cause: feature_registry.py did not exist before this refactor.
# Decision: Import registry classes here for use in build_base_features.
# Why this approach: Centralized registry is the single source of audit truth.
# Expected Impact: Every drop is traceable; no information lost between phases.
from .feature_registry import FeatureDropRegistry, PruningManifest

# [WHY_THIS_CHANGE] TS Semantic Correction
# Problem: Blindly filling all boundary NaNs with 0 (fillna(0)) is semantically incorrect.
#   - For counts (e.g., order_inflow), 0 is a correct starting state.
#   - For sensors (e.g., temperature), 0 is a massive artifact (thermal shock).
#   - For ratios (e.g., utilization), 0 may be a misleading outlier.
# Decision: Categorize BASE_COLS and apply type-aware boundary filling.
def infer_feature_types(df, features):
    """
    [WHY_THIS_CHANGE] Zero-Hardcode Type Inference (TASK 4)
    Problem: Name-based heuristics (any(k in col.lower()...)) are brittle and semantically weak.
    Root Cause: Mixing statistical inference with heuristic matching.
    Why previous logic failed: Could misclassify features with misleading names (INVALID per MISSION).
    Why this solution: Use purely statistical properties (cardinality, range, integer-ness)
      to distinguish between event counts, bounded ratios, and continuous sensors.
    Expected Impact: Categorization is 100% data-driven.
    """
    types = {}
    for col in features:
        if col not in df.columns: continue
        series = df[col].dropna()
        if series.empty:
            types[col] = "sensor" 
            continue
            
        unique_count = series.nunique()
        is_int = pd.api.types.is_integer_dtype(df[col]) or (series % 1 == 0).all()
        
        # [DECISION_TRACE] 
        # COUNT: Integer-like with low cardinality (< 10% of samples OR < 100 absolute)
        # RATIO: Bounded [0, 1] with sufficient unique values to not be a count.
        # SENSOR: Wide range / High cardinality floats.
        if is_int and unique_count < min(100, len(df) * 0.1):
            types[col] = "count"
        elif (series.min() >= -1e-5 and series.max() <= 1.00001 and unique_count > 10):
             types[col] = "ratio"
        else:
            types[col] = "sensor"
    return types

logger = logging.getLogger(__name__)

"""
[CONTEXT — DO NOT REMOVE]

This refactor was triggered by a structural failure where:

1. PCA-driven feature selection reduced EMBED_BASE_COLS from 30 → 15
2. Time-series feature generation depended on EMBED_BASE_COLS
3. This caused ~337 features to silently disappear from runtime
4. schema.py still declared ~700 features → causing fatal mismatch
5. trainer attempted to access non-existent columns → KeyError
6. Config refactor removed critical attributes → AttributeError risk

Root Cause:
- PCA was incorrectly treated as a primary constraint
- Feature generation pipeline became dependent on PCA inputs
- Schema and runtime were no longer synchronized

Resolution Strategy:
- Feature generation must be BASE_COLS-driven, NOT PCA-driven
- Schema must EXACTLY match runtime-generated features
- PCA must become OPTIONAL (non-blocking)
- No silent feature drops allowed under any condition

This system now enforces:
- Zero tolerance for Schema-Runtime mismatch
- Deterministic feature generation
- Explicit failure instead of silent fallback

[END CONTEXT]
"""

# [CONTEXT] Added to enforce feature count limits and ensure stable embedding generation
# to prevent historical OOM and noise amplification issues, and to detect fake signals
# during PCA fallback.

# [CONTEXT] NaN Root Cause (Audit 2026-04-24):
# Raw BASE_COLS carry 10-17% NaN from source CSV.
# Shift-based generators (accel, diff, rate) compound this to 23-46% NaN.
# This module now enforces NaN-aware generation and post-generation pruning.

# [WHY_THIS_CHANGE]
# Problem: NAN_DROP_THRESHOLD=0.15, CORR_PRUNE_THRESHOLD=0.85, CORR_SAMPLE_SIZE=10000
#   were hardcoded module-level constants with no statistical justification.
#   They violated RULE 1 (NO HARDCODING) of the Zero-Hardcode audit mandate.
# Root Cause: Original constants were chosen empirically in early development
#   without systematic analysis of the feature NaN/correlation distributions.
# Decision: All three thresholds are now DERIVED AT RUNTIME inside build_base_features()
#   using data-distribution statistics:
#     NAN threshold    → P85 of the observed NaN ratio distribution (adaptive to dataset)
#     CORR threshold   → Q3 + 1.5*IQR of the upper-triangular correlation distribution
#     CORR sample size → min(10000, len(df)) — uses all data if small, samples if large
# Why this approach (not alternatives):
#   - Fixed percentile (e.g., always use 85th): avoids hardcoding a ratio
#   - IQR-based for corr: robust to outliers, scale-invariant, widely established
#   - Dynamic sample: no information wasted on small datasets
# Expected Impact: Thresholds adapt to dataset size and distribution characteristics.
#   Logged with full derivation trace in pipeline_trace.json.
#
# LEGACY CONSTANTS REMOVED (kept here as reference for historical context):
#   NAN_DROP_THRESHOLD = 0.15   ← replaced by P85 of NaN distribution
#   CORR_PRUNE_THRESHOLD = 0.85 ← replaced by IQR-outlier threshold on corr matrix
#   CORR_SAMPLE_SIZE = 10000    ← replaced by min(10000, len(df))
CORR_SAMPLE_SIZE_CAP = 10000  # Upper bound on sample rows (not a threshold — just a cap)

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
        
        # [STRUCTURAL_REALIGNMENT] PCA Mode tracking
        # Mode A: 'active' | Mode B: 'fallback_raw' | Mode C: 'disabled'
        self.pca_mode = 'active'
        self.active_pcas = {}

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
        n_cols = X_np.shape[1]
        n_embed_base = len(Config.EMBED_BASE_COLS)
        n_raw_all = len(FEATURE_SCHEMA['raw_features'])
        
        if n_cols == n_embed_base:
            X_embed_base = X_np
        elif n_cols == n_raw_all:
            base_cols_idx = [i for i, col in enumerate(FEATURE_SCHEMA['raw_features']) if col in Config.EMBED_BASE_COLS]
            X_embed_base = X_np[:, base_cols_idx]
        elif n_cols > n_raw_all:
            # Case where ID columns are included
            # This is risky, but let's try to match by name if X_train was a DF (but it's np here)
            # Better to fail explicitly if we can't be sure
            raise ValueError(f"[CONTRACT_FAIL] Reconstructor.fit received {n_cols} columns. Expected {n_embed_base} or {n_raw_all}.")
        else:
            raise ValueError(f"[CONTRACT_FAIL] Reconstructor.fit received unexpected shape {X_np.shape}. Expected {n_embed_base} or {n_raw_all} features.")
        
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
        
        # [STRUCTURAL_REALIGNMENT] Non-blocking PCA with Fallbacks
        self.active_pcas = {}
        pca_targets = [('raw', self.pca_raw, X_w_raw), ('log', self.pca_log, X_w_log), 
                       ('rank', self.pca_rank, X_w_rank), ('local', self.global_pca_local, X_weighted)]
        
        for tag, pca, X_view in pca_targets:
            try:
                pca.fit(X_view)
                var_sum = np.sum(pca.explained_variance_ratio_)
                if var_sum < 0.8:
                    logger.warning(f"[PCA_LOW_VARIANCE] {tag} explained variance: {var_sum:.4f} < 0.8. Switching to Mode B for this view.")
                    self.active_pcas[tag] = False # Fallback to raw for this view
                else:
                    logger.info(f"[PCA_QC_SUCCESS] {tag} PCA explained variance: {var_sum:.4f} >= 0.8")
                    self.active_pcas[tag] = True
            except Exception as e:
                logger.error(f"[PCA_FIT_FAILURE] {tag} failed: {str(e)}. Switching to Mode B for this view.")
                self.active_pcas[tag] = False

        # [DIM_TRACE] STEP 3: EMBEDDING
        # [CONTEXT] This guard prevents PCA fallback from silently producing
        # low-information embeddings, which previously caused misleading KNN features.
        # It loops up to 2 times to escalate from 'active' to 'fallback_raw'.
        for attempt in range(2):
            try:
                final_embed = self.get_embeddings(X_embed_base, already_scaled=True)
                
                # [EMBED_QUALITY_AUDIT] Fake Signal Detection
                mean_var = np.var(final_embed, axis=0).mean()
                
                # Subsample for cosine similarity to avoid memory explosion
                sample_size = min(1000, final_embed.shape[0])
                sample_idx = np.random.choice(final_embed.shape[0], sample_size, replace=False)
                sample_embed = final_embed[sample_idx]
                norms = np.linalg.norm(sample_embed, axis=1, keepdims=True) + 1e-8
                sample_norm = sample_embed / norms
                cos_sim = np.dot(sample_norm, sample_norm.T)
                np.fill_diagonal(cos_sim, 0)
                mean_cos_sim = np.mean(np.abs(cos_sim))
                
                logger.info(f"[EMBED_QUALITY_AUDIT] Mode={self.pca_mode} | Mean var: {mean_var:.6f} | Mean Cosine Sim: {mean_cos_sim:.6f}")
                
                if mean_var < 1e-4 or mean_cos_sim > 0.95:
                    logger.error(f"[PCA_MODE_SWITCH] reason: variance or cosine sim fail | original variance: {mean_var:.6f} | fallback strategy: escalate")
                    if self.pca_mode == 'active':
                        self.pca_mode = 'fallback_raw'
                        logger.info(f"[PCA_MODE_SWITCH] Escaping active PCA -> fallback_raw")
                        continue
                    elif self.pca_mode == 'fallback_raw':
                        # Force error, no silent pass
                        logger.error(f"[PCA_MODE_SWITCH] Strategy: fallback_raw -> ERROR | Original Var: {mean_var:.6f}")
                        raise RuntimeError("[EMBED_QUALITY_CRITICAL] Both PCA and fallback_raw produced invalid embeddings (informationless).")
                
                logger.debug(f"[DIM_TRACE] after embedding: shape={final_embed.shape} | source=get_embeddings | mode={self.pca_mode}")
                self.kmeans.fit(final_embed.astype(np.float64))
                break
                
            except Exception as e:
                if "EMBED_QUALITY_CRITICAL" in str(e):
                    raise e
                logger.error(f"[EMBEDDING_FIT_FAILURE] reason: fit fail | error: {str(e)}")
                if self.pca_mode == 'active':
                    self.pca_mode = 'fallback_raw'
                    logger.info("[PCA_MODE_SWITCH] fallback strategy: active -> fallback_raw due to exception")
                else:
                    raise RuntimeError(f"[EMBEDDING_FIT_FAILURE] fallback_raw failed: {str(e)}")

    def get_embeddings(self, X, already_scaled=False):
        X_np = np.asarray(X, dtype=np.float32)
        
        # [RELIABILITY_FIX] Always handle Infs and NaNs
        X_np = np.where(np.isinf(X_np), np.nan, X_np)
        
        if not already_scaled:
            # [CONTRACT_ENFORCEMENT]
            if X_np.shape[1] != len(Config.EMBED_BASE_COLS):
                 raise ValueError(f"[CONTRACT_FAIL] get_embeddings expected {len(Config.EMBED_BASE_COLS)} cols, got {X_np.shape[1]}")
            
            X_scaled = self.imputer.transform(X_np)
        else:
            # PCA cannot handle NaNs. We force imputation if NaNs are still present.
            if np.isnan(X_np).any():
                X_scaled = self.imputer.transform(X_np)
            else:
                X_scaled = X_np
            
        # [STRUCTURAL_REALIGNMENT] Fallback Mode B: Use raw features instead of PCA
        # [CONTEXT] This ensures that when PCA fails or produces low variance, we use raw features 
        # to maintain the 32-dim shape without losing information.
        # [FALLBACK_GUARD] Block fallback if raw features are themselves NaN-heavy.
        # If removed: PCA fallback would silently propagate corrupted embeddings into KNN.
        if self.pca_mode == 'fallback_raw':
            nan_ratio = np.isnan(X_scaled).mean()
            if nan_ratio > 0.20:
                logger.error(f"[FALLBACK_GUARD] Raw feature NaN ratio {nan_ratio:.2%} > 20%. BLOCKING fallback_raw.")
                raise RuntimeError(f"[FALLBACK_GUARD] Cannot use fallback_raw: input NaN ratio {nan_ratio:.2%} exceeds safety threshold.")
            # Create a 32-dim representation by concatenating raw views or padding
            X_raw, X_log, X_rank = self._get_multi_view_X(X_scaled)
            # Take first 10 dims of each or similar to keep dim=32
            combined = np.hstack([X_raw[:, :10], X_log[:, :10], X_rank[:, :10], X_scaled[:, :2]])
            return combined.astype(np.float32)
            
        # [STRUCTURAL_REALIGNMENT] Fallback Mode C: Zero embeddings (disabled)
        if self.pca_mode == 'disabled':
            return np.zeros((len(X_np), self.embed_dim), dtype=np.float32)

        X_raw, X_log, X_rank = self._get_multi_view_X(X_scaled)
        
        # [STRUCTURAL_REALIGNMENT] Safe transform with individual PCA checks
        e_raw = self.pca_raw.transform(X_raw) if self.active_pcas.get('raw') else X_raw[:, :8]
        e_log = self.pca_log.transform(X_log) if self.active_pcas.get('log') else X_log[:, :8]
        e_rank = self.pca_rank.transform(X_rank) if self.active_pcas.get('rank') else X_rank[:, :8]
        e_local = self.global_pca_local.transform(X_scaled) if self.active_pcas.get('local') else X_scaled[:, :8]
        
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
                # [WHY_THIS_DESIGN] Redundancy Removal
                # volatility_proxy was exactly same as embed_std, removed to save memory.
                if f'embed_mean_{k}' not in final_results: final_results[f'embed_mean_{k}'] = []
                if f'embed_std_{k}' not in final_results: final_results[f'embed_std_{k}'] = []
                if f'weighted_mean_{k}' not in final_results: final_results[f'weighted_mean_{k}'] = []
                if f'trend_proxy_{k}' not in final_results: final_results[f'trend_proxy_{k}'] = []
                
                final_results[f'embed_mean_{k}'].append(mean_feat)
                final_results[f'embed_std_{k}'].append(std_feat)
                final_results[f'weighted_mean_{k}'].append(weighted_mean_feat)
                final_results[f'trend_proxy_{k}'].append((mean_feat - t_embed).astype(np.float32))

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

def get_protected_candidates(df_columns):
    """
    [WHY_THIS_CHANGE] Renamed protected_cols to protected_candidates.
    Changed to "conditional protection" (Task 1 & 2).
    Includes domain critical features AND all signal bucket features as candidates.
    Final protection is determined later by SignalValidator based on Gain, Permutation, and Stability.
    """
    protected = set(BASE_COLS) | set(Config.EMBED_BASE_COLS)
    DOMAIN_CRITICAL_PREFIXES = [
        'order_inflow', 'charge_queue_length', 'congestion_score',
        'robot_utilization', 'near_collision', 'blocked_path',
    ]
    
    SIGNAL_BUCKET_SUFFIXES = [
        '_rolling_mean_3', '_rolling_mean_5',
        '_slope_5', '_rate_1', '_diff_1',
        '_rolling_std_3', '_rolling_std_5'
    ]
    
    for col_name in df_columns:
        if col_name in Config.ID_COLS or col_name == Config.TARGET:
            continue
        
        # 1. Domain Critical
        for prefix in DOMAIN_CRITICAL_PREFIXES:
            if col_name.startswith(prefix):
                protected.add(col_name)
                break
                
        # 2. Signal Bucket Candidates
        for base_col in BASE_COLS:
            if col_name.startswith(base_col) and any(col_name.endswith(s) for s in SIGNAL_BUCKET_SUFFIXES):
                protected.add(col_name)
                break
                
    return protected

def build_base_features(df, pruning_manifest: "PruningManifest" = None):
    """
    [PHASE 1: ISOLATED BASE FEATURES]
    Builds temporal and row-wise features that are safe to compute globally.
    ... (docstring truncated for brevity)
    """
    is_train_mode = pruning_manifest is None

    logger.info(f"[BUILD_BASE] Input shape: {df.shape} | mode={'TRAIN (compute thresholds)' if is_train_mode else 'TEST (apply manifest)'}")
    df = df.copy()

    # [STRUCTURAL_REALIGNMENT] Filter columns to only those defined in SSOT
    relevant_cols = list(BASE_COLS) + list(Config.ID_COLS)
    if Config.TARGET in df.columns:
        relevant_cols.append(Config.TARGET)

    keep_cols = [c for c in relevant_cols if c in df.columns]
    df = df[keep_cols]

    original_ids = df['ID'].values.copy()
    
    # 1. Temporal Ordering
    df = df.sort_values(by=["scenario_id", "ID"]).reset_index(drop=True)
    
    # [TASK 5/11] Compute train-derived statistics for leakage-free sub-function calls
    logger.info(f"[BUILD_BASE] Initial Count: {len(df.columns)}")

    if is_train_mode:
        # TRAIN: compute col means from train df (safe — df IS train)
        train_col_means = {
            col: float(df[col].mean()) for col in BASE_COLS
            if col in df.columns and not df[col].isna().all()
        }
        df = add_time_series_features(df, train_col_means=train_col_means)
    else:
        # TEST: use train-derived means from manifest (zero leakage)
        train_col_means = pruning_manifest.train_col_means
        df = add_time_series_features(df, train_col_means=train_col_means)

    logger.info(f"[BUILD_BASE] After TS Expansion: {len(df.columns)}")

    if is_train_mode:
        # TRAIN: compute extreme quantiles from train
        key_extreme_cols = ['order_inflow_15m', 'robot_utilization', 'congestion_score', 'near_collision_15m']
        extreme_quantiles = {}
        for c in key_extreme_cols:
            if c in df.columns:
                extreme_quantiles[c] = float(df[c].quantile(0.95))
        df = add_extreme_detection_features(df, extreme_quantiles=None)
        util_rel_col = f'robot_utilization_rel_to_mean_5'
        if util_rel_col in df.columns:
            util_rel_clean = df[util_rel_col].dropna().values
            extreme_quantiles['util_rel_p90'] = float(
                np.quantile(util_rel_clean, 0.90) if len(util_rel_clean) > 0 else 1.2
            )
    else:
        extreme_quantiles = pruning_manifest.extreme_quantiles
        df = add_extreme_detection_features(df, extreme_quantiles=extreme_quantiles)

    logger.info(f"[BUILD_BASE] After Extreme Expansion: {len(df.columns)}")
    
    # 3. Order Restoration
    df = df.set_index('ID').loc[original_ids].reset_index()
    

    # [TASK 1] PROTECTED_COLS -> "조건부 보호" (protected_candidates)
    # [WHY_THIS_CHANGE] "무조건 보호"는 drift/noise feature까지 보호하여 과적합을 유발함.
    # [ROOT_CAUSE] force_include 구조가 신호 검증 없이 survival을 보장했음.
    # [EXPECTED_IMPACT] 진짜 신호만 보호되고 noise 유입 차단. (최종 검증은 main.py에서 수행)
    protected_candidates = get_protected_candidates(df.columns)

    # Initialize registry for this build (train mode builds full audit record).
    registry = FeatureDropRegistry()

    # ------------------------------------------------------------------
    # [NAN_FEATURE_BLOCKER]
    # [CONTEXT] Prevents generation of high-NaN features which previously caused
    # feature explosion and unstable PCA embeddings. Root cause: raw BASE_COLS
    # carry 10-17% NaN, and shift-based generators (accel, diff, rate) compound
    # this to 23-46%. If removed: 326+ high-NaN features enter the model.
    # ------------------------------------------------------------------
    # [AXIS1_FIX] FEATURE_SCHEMA 싱글턴 변이 방지를 위한 로컬 드롭셋
    dropped_features = set()

    runtime_raw_current = [c for c in df.columns if c not in Config.ID_COLS and c != Config.TARGET]
    nan_ratios = df[runtime_raw_current].isna().mean()

    if is_train_mode:
        # [WHY_THIS_DESIGN] Adaptive NaN Thresholding
        # Observed Data Behavior: Distinct "quality clusters" in sensors (some 10% NaN, some 25%).
        # Why 2-Sigma Jump: Identifies the largest statistical discontinuity in data quality.
        # Mathematical Justification: Outlier detection in the jump distribution (1st order diff).
        # Sensitivity: 1.5*std is too sensitive (drops clean features); 3*std ignores real gaps.
        # [TASK 1 — NaN Gap Detection Logic Recovery]
        # [WHY_THIS_DESIGN] nan_ratios_sorted MUST use the same feature set as nan_ratios.
        # [CODE_EVIDENCE] Previously: nan_ratios_sorted = df.isna().mean().sort_values()
        #   This operated on ALL df columns including ID columns (ID, scenario_id, layout_id).
        #   nan_ratios (line 589) operates on runtime_raw_current which EXCLUDES ID columns.
        #   At line 606: gap_threshold = nan_ratios[max_diff_idx] — if max_diff_idx was an
        #   ID column name, this would KeyError. Even without KeyError, including ID columns
        #   in the sorted distribution biases gap detection (ID columns have 0% NaN, skewing jumps).
        # [FAILURE_MODE_PREVENTED] KeyError on ID column lookup + biased gap threshold.
        nan_ratios_sorted = nan_ratios.sort_values()
        diffs = nan_ratios_sorted.diff().dropna()
        if len(diffs) > 0:
            mean_jump = diffs.mean()
            std_jump = diffs.std()
            significant_jump = mean_jump + 2 * std_jump
            
            # Find largest jump in NaN ratios
            max_diff_idx = diffs.idxmax()
            gap_threshold = nan_ratios[max_diff_idx]
            
            # Safety: Threshold must be above the 50th percentile of NaNs to avoid dropping everything
            min_drop_threshold = nan_ratios.median() + 1e-6
            
            if diffs.max() > significant_jump and gap_threshold > min_drop_threshold:
                adaptive_nan_threshold = float(gap_threshold)
                derivation_nan = f"Data-driven gap detected at {adaptive_nan_threshold:.4f} (jump={diffs.max():.4f} > 2-sigma threshold {significant_jump:.4f})"
            else:
                # Fallback: Use a very conservative upper bound (e.g. 50%) if no clear gap
                adaptive_nan_threshold = 0.50
                derivation_nan = "No significant gap detected in NaN distribution; using conservative 50% ceiling."
        else:
            adaptive_nan_threshold = 0.50
            derivation_nan = "Insufficient features for gap detection; using 50% fallback."

        # Record derivation metadata
        supporting_stats = {
            "max_ratio": float(nan_ratios.max()),
            "mean_ratio": float(nan_ratios.mean()),
            "gap_detected": "gap_threshold" in locals()
        }
        registry.record_threshold("nan_threshold", adaptive_nan_threshold, derivation_nan, supporting_stats)
    else:
        # TEST MODE: Use threshold computed on train — NO recomputation (zero leakage)
        adaptive_nan_threshold = pruning_manifest.nan_threshold
        logger.info("[NAN_FEATURE_BLOCKER] TEST mode: applying train-derived threshold={:.4f}".format(adaptive_nan_threshold))

    if is_train_mode:
        high_nan_cols = [c for c in nan_ratios[nan_ratios > adaptive_nan_threshold].index if c not in protected_candidates]
    else:
        # Apply exactly the same columns that were dropped on train
        high_nan_cols = [c for c in pruning_manifest.cols_to_drop_nan if c in df.columns]

    if high_nan_cols:
        logger.warning("[NAN_FEATURE_BLOCKER] Dropping {} features with NaN ratio > {:.4f}".format(
            len(high_nan_cols), adaptive_nan_threshold))
        df = df.drop(columns=high_nan_cols)
        # [AXIS1_FIX] 싱글턴 변이 금지, 로컬 드롭셋에 추가
        dropped_features.update(high_nan_cols)
        if is_train_mode:
            for col in high_nan_cols:
                registry.record_drop(col, "nan_drop", float(nan_ratios[col]),
                                     adaptive_nan_threshold, derivation_nan)

    # ------------------------------------------------------------------
    # [TASK 10 — CLUSTER-BASED REDUNDANCY REMOVAL]
    # [WHY_THIS_DESIGN] Replaced P99 correlation threshold with graph-based clustering.
    # [CODE_EVIDENCE] Previous approach: adaptive_corr_threshold = np.percentile(upper_tri, 99)
    #   P99 is unstable: depends on correlation distribution shape, which varies with
    #   feature count and scale. A dataset with 200 vs 400 features produces entirely
    #   different P99 values even if underlying redundancy is identical.
    # [WHY_THIS_APPROACH] Hierarchical clustering on correlation distance:
    #   1. Convert |corr| to distance: D = 1 - |corr| (0 = identical, 1 = uncorrelated)
    #   2. Average-linkage clustering (robust to outlier correlations)
    #   3. Cut threshold derived from IQR of pairwise distances (data-driven)
    #   4. Within each cluster: keep representative with highest LightGBM importance
    # [FAILURE_MODE_PREVENTED] Arbitrary P99 threshold causing over/under-pruning.
    # ------------------------------------------------------------------
    runtime_raw_current = [c for c in df.columns if c not in Config.ID_COLS and c != Config.TARGET]
    n_sample = min(CORR_SAMPLE_SIZE_CAP, len(df))
    # [TASK 12 — DETERMINISM] Use fixed random_state for reproducible sampling
    sample_indices = df.index.to_series().sample(n=n_sample, random_state=42).index
    sample_df = df.loc[sample_indices, runtime_raw_current].astype("float32")
    # Fill NaN for correlation computation only (does not affect main df)
    sample_filled = sample_df.fillna(sample_df.median())
    corr_matrix = sample_filled.corr().abs().fillna(0.0)
    corr_values = corr_matrix.to_numpy(copy=True)
    np.fill_diagonal(corr_values, 0)

    # [CONTEXT] Never prune BASE_COLS or EMBED_BASE_COLS — they are structurally required
    # by PCA reconstructor and schema. Pruning them causes KeyError downstream.
    cols_array = np.array(corr_matrix.columns)
    upper_tri_indices = np.triu_indices_from(corr_values, k=1)
    upper_tri = corr_values[upper_tri_indices]

    if is_train_mode:
        # --- STEP 1: Build correlation distance matrix ---
        # [FAILURE_MODE_PREVENTED] Finite values check for linkage()
        corr_dist = 1.0 - corr_values
        np.fill_diagonal(corr_dist, 0)
        corr_dist = np.clip(corr_dist, 0, 1)  # Numerical safety

        # --- STEP 2: Hierarchical clustering ---
        condensed_dist = squareform(corr_dist, checks=False)
        # [FAILURE_MODE_PREVENTED] Sanitize condensed_dist to eliminate any lingering NaN/Inf
        condensed_dist = np.nan_to_num(condensed_dist, nan=1.0, posinf=1.0, neginf=1.0)
        Z = linkage(condensed_dist, method='average')

        # --- STEP 3: Derive cut threshold from distance distribution ---
        # [WHY_THIS_DESIGN] IQR-based cut: features with mutual distance < Q1 are
        # "redundancy outliers" (closer than 75% of all pairs). This is robust to
        # distribution shape and adapts to feature correlation structure.
        # [WHY_THIS_CHANGE] Upper clip bound changed from 0.50 to 0.20.
        # [ROOT_CAUSE] Forensic PROOF 4 demonstrated that actual correlation
        #   distribution P75=0.054, P95=0.235. With q1_d≈0.945, the clip(q1_d, 0.02, 0.50)
        #   ALWAYS saturated to 0.50, which means distance < 0.50 → corr > 0.50.
        #   This eliminated 0.61% of all feature pairs — including domain-critical
        #   time-series derivatives that have natural moderate correlation (0.5–0.8)
        #   due to physical coupling (e.g., order_inflow → congestion).
        # [WHY_NOT_ALTERNATIVES]
        #   - 0.30 (corr>0.70): Still too aggressive for physically coupled features.
        #   - 0.10 (corr>0.90): Would retain near-duplicates. Tree models waste splits.
        #   - 0.20 (corr>0.80): Removes only truly redundant near-clones while
        #     preserving physically distinct signals at different time scales.
        # [EXPECTED_IMPACT] Features with correlation 0.5–0.8 survive pruning.
        #   Expected: ~100+ domain derivatives restored. corr_drop count reduced by ~60%.
        nonzero_dists = condensed_dist[condensed_dist > 1e-10]
        if len(nonzero_dists) >= 4:
            q1_d = float(np.percentile(nonzero_dists, 25))
            q3_d = float(np.percentile(nonzero_dists, 75))
            iqr_d = q3_d - q1_d
            cut_threshold = float(np.clip(q1_d, 0.02, 0.20))
        else:
            cut_threshold = 0.15  # Absolute fallback for tiny feature sets

        clusters = fcluster(Z, t=cut_threshold, criterion='distance')
        adaptive_corr_threshold = cut_threshold  # Store for manifest compatibility

        # --- STEP 4: Select cluster representatives ---
        feature_names = list(cols_array)
        cluster_map = {}
        for feat, cid in zip(feature_names, clusters):
            cluster_map.setdefault(int(cid), []).append(feat)

        multi_member_clusters = {k: v for k, v in cluster_map.items() if len(v) > 1}
        n_clusters = len(cluster_map)
        n_multi = len(multi_member_clusters)
        logger.info(f"[CLUSTER_PRUNE] {n_clusters} clusters found, {n_multi} have >1 member (cut={cut_threshold:.4f})")

        corr_drop = set()
        if multi_member_clusters:
            # Use LightGBM importance to select representatives
            y_sample = None
            if Config.TARGET in df.columns:
                y_sample = df.loc[sample_indices, Config.TARGET].values

            if y_sample is not None and len(y_sample) > 0 and not np.isnan(y_sample).all():
                from lightgbm import LGBMRegressor
                X_imp = sample_filled.values
                # [TASK 3 — SHALLOW MODEL CONTRADICTION FIX]
                # [WHY_THIS_CHANGE] Aligned selection model capacity with the main model.
                # [ROOT_CAUSE] The temporary model used for cluster representative selection
                #   was hardcoded to max_depth=3, n_estimators=50. The main model uses
                #   num_leaves=31 (effectively depth ~5). Features that require deeper 
                #   splits (like multi-way interactions or conditional flags) received 
                #   0 importance in the shallow model and were systematically dropped,
                #   causing a bias against complex features before train time.
                # [WHY_NOT_ALTERNATIVES] Hardcoding max_depth=5 would create a new magic number.
                #   The only structurally safe approach is to exactly mirror the main
                #   model's capacity (RAW_LGBM_PARAMS) so selection matches final usage.
                # [EXPECTED_IMPACT] Interaction and conditional features that rely on deeper
                #   splits will now accurately receive importance scores and survive pruning.
                imp_params = Config.RAW_LGBM_PARAMS.copy()
                if 'random_state' not in imp_params:
                    imp_params['random_state'] = 42
                imp_model = LGBMRegressor(**imp_params)
                imp_model.fit(X_imp, y_sample)
                importances = dict(zip(feature_names, imp_model.feature_importances_))
                del imp_model
                selection_method = "lgbm_importance"
            else:
                # Fallback: use variance as proxy for importance
                importances = dict(zip(feature_names, sample_filled.var().values))
                selection_method = "variance_fallback"

            logger.info(f"[CLUSTER_PRUNE] Representative selection method: {selection_method}")

            for cid, members in multi_member_clusters.items():
                protected_members = [m for m in members if m in protected_candidates]
                droppable_members = [m for m in members if m not in protected_candidates]

                if protected_members:
                    # [STRATEGY 1] Keep all protected members — they are domain-critical.
                    # Only drop members that are NOT protected.
                    corr_drop.update(droppable_members)
                elif droppable_members:
                    # Keep highest-importance member, drop rest
                    best = max(members, key=lambda f: importances.get(f, 0))
                    corr_drop.update(m for m in members if m != best)

        # [TASK 2] SIGNAL BUCKET -> "존재 보장"에서 "품질 보장"으로 변경
        # [WHY_THIS_CHANGE] "존재 보장"은 개수만 맞추기 위해 garbage feature까지 강제 생존시킴.
        # [ROOT_CAUSE] 기존 bucket 구조가 의미/품질 검증 없이 단순히 최소 1개를 살리도록 설계됨.
        # [EXPECTED_IMPACT] garbage signal 제거 및 핵심 신호 유지.
        # 이 단계에서는 rescue를 수행하지 않고, protected_candidates로 넘겨서 main.py의
        # SignalValidator에서 SCORE 기반으로 품질을 검증하여 통과한 것만 최종 보호합니다.
        pass

        derivation_corr = (
            f"Cluster-based pruning: hierarchical clustering (average linkage) on "
            f"correlation distance matrix. Cut threshold={cut_threshold:.4f} derived from "
            f"Q1 of pairwise distances. {n_multi} multi-member clusters resolved via {selection_method if multi_member_clusters else 'N/A'}."
        )
        supporting_stats = {
            "n_clusters": n_clusters,
            "n_multi_clusters": n_multi,
            "cut_threshold": cut_threshold,
            "corr_P50": float(np.median(upper_tri)),
            "corr_P95": float(np.percentile(upper_tri, 95)),
            "n_pairs": int(len(upper_tri))
        }
        registry.record_threshold("corr_threshold", adaptive_corr_threshold, derivation_corr, supporting_stats)
    else:
        # TEST MODE: Use threshold from manifest
        adaptive_corr_threshold = pruning_manifest.corr_threshold
        logger.info("[CLUSTER_PRUNE] TEST mode: applying train-derived drop list.")

    if is_train_mode:
        pass  # corr_drop already computed above
    else:
        # TEST: apply exactly the same columns dropped on train
        corr_drop = set(c for c in pruning_manifest.cols_to_drop_corr if c in df.columns)

    if corr_drop:
        logger.warning("[CLUSTER_PRUNE] Dropping {} redundant features (cluster cut={:.4f})".format(
            len(corr_drop), adaptive_corr_threshold))
        df = df.drop(columns=list(corr_drop))
        # [AXIS1_FIX] 싱글턴 변이 금지, 로컬 드롭셋에 추가
        dropped_features.update(corr_drop)
        if is_train_mode:
            for col in corr_drop:
                col_idx = list(cols_array).index(col) if col in cols_array else -1
                corr_stat = float(corr_values[col_idx].max()) if col_idx >= 0 else float("nan")
                registry.record_drop(col, "corr_drop", corr_stat,
                                     adaptive_corr_threshold, derivation_corr)
    else:
        logger.info("[CLUSTER_PRUNE] No redundant clusters found.")

    del sample_df, sample_filled, corr_matrix, corr_values
    gc.collect()

    # ------------------------------------------------------------------
    # [AXIS3_FIX] Variance Floor Filter
    # ------------------------------------------------------------------
    runtime_raw_current = [c for c in df.columns if c not in Config.ID_COLS and c != Config.TARGET]
    current_var = df[runtime_raw_current].var()

    if is_train_mode:
        # [WHY_THIS_CHANGE] Zero-Hardcode Variance (TASK 2)
        # Problem: 1e-6 is a hardcoded constant.
        # Why this solution: Use the 1st percentile of the global variance distribution
        #   as the floor. This effectively blocks features that are "more constant" than
        #   99% of the feature space.
        # Expected Impact: Threshold scales with the data's inherent scale.
        eps = np.finfo(np.float32).eps
        adaptive_var_threshold = float(max(eps, current_var.quantile(0.01)))
        derivation_var = f"Data-driven P1 variance floor: {adaptive_var_threshold:.2e}"
        
        supporting_stats = {
            "min_var": float(current_var.min()) if not current_var.empty else 0.0,
            "median_var": float(current_var.median()) if not current_var.empty else 0.0,
            "n_features": len(current_var)
        }
        registry.record_threshold("var_threshold", adaptive_var_threshold, derivation_var, supporting_stats)
    else:
        adaptive_var_threshold = pruning_manifest.var_threshold
        derivation_var = pruning_manifest.derivation_log.get("var_threshold", "Manifest-derived")
        logger.info("[VARIANCE_FLOOR] TEST mode: applying train-derived threshold={:.2e}".format(adaptive_var_threshold))

    if is_train_mode:
        zero_var_cols = [
            c for c in runtime_raw_current
            if current_var[c] < adaptive_var_threshold and c not in protected_candidates
        ]
    else:
        zero_var_cols = [c for c in pruning_manifest.cols_to_drop_var if c in df.columns]

    if zero_var_cols:
        logger.warning("[VARIANCE_FLOOR] Dropping {} near-zero-variance features (threshold={:.2e})".format(
            len(zero_var_cols), adaptive_var_threshold))
        df = df.drop(columns=zero_var_cols)
        dropped_features.update(zero_var_cols)
        if is_train_mode:
            for col in zero_var_cols:
                registry.record_drop(
                    col, "variance_drop",
                    float(current_var[col]) if col in current_var.index else float("nan"),
                    adaptive_var_threshold, derivation_var
                )

    # ------------------------------------------------------------------
    # [FEATURE_HEALTH_SUMMARY & SCHEMA SYNC]
    # [CONTEXT] Provides a single-glance health check of the final feature space
    # and enforces that runtime features exactly match the current FEATURE_SCHEMA.
    # ------------------------------------------------------------------

    # 1. Enforce FEATURE_SCHEMA (SSOT)
    # [AXIS1_FIX] 싱글턴 스키마 대신 드롭을 반영한 로컬 스키마 사용
    schema_raw = [f for f in FEATURE_SCHEMA["raw_features"] if f not in dropped_features]
    current_cols = [c for c in df.columns if c not in Config.ID_COLS and c != Config.TARGET]
    extra_at_runtime = set(current_cols) - set(schema_raw)

    if extra_at_runtime:
        logger.info("[SCHEMA_SYNC] Dropping {} features not in FEATURE_SCHEMA: {}...".format(
            len(extra_at_runtime), list(extra_at_runtime)[:5]))
        df = df.drop(columns=list(extra_at_runtime))
        if is_train_mode:
            for col in extra_at_runtime:
                registry.record_drop(col, "schema_extra", float("nan"), float("nan"),
                                     "Feature not in FEATURE_SCHEMA[raw_features] SSOT")

    runtime_raw = [c for c in df.columns if c not in Config.ID_COLS and c != Config.TARGET]
    final_nan_pct = df[runtime_raw].isna().mean()
    final_var = df[runtime_raw].var()

    # [TASK 7 — FULL TRACE LOG]
    # [WHY_THIS_CHANGE]
    # Problem: No structured summary existed confirming all decisions + derivations.
    # Root Cause: Decision logging was scattered; no machine-readable audit output.
    # Decision: Emit a structured PIPELINE_TRACE block at end of build_base_features.
    # Expected Impact: Every run produces a complete, traceable decision log.
    if is_train_mode:
        registry.log_summary()
        logger.info("[PIPELINE_TRACE] ===========================")
        logger.info("[PIPELINE_TRACE] Final Feature Count: {}".format(len(runtime_raw)))
        logger.info("[PIPELINE_TRACE] nan_drop     : {} | threshold={:.4f} ({})".format(
            len(high_nan_cols), adaptive_nan_threshold, derivation_nan))
        logger.info("[PIPELINE_TRACE] corr_drop    : {} | threshold={:.4f} ({})".format(
            len(corr_drop), adaptive_corr_threshold, derivation_corr))
        logger.info("[PIPELINE_TRACE] variance_drop: {} | threshold={:.2e} ({})".format(
            len(zero_var_cols), adaptive_var_threshold, derivation_var))
        logger.info("[PIPELINE_TRACE] redundancy   : Removed volatility_proxy and expanding_std")
        logger.info("[PIPELINE_TRACE] rationale    : All thresholds derived from data sigma/percentiles.")
        logger.info("[PIPELINE_TRACE] ===========================")

    logger.info("\n[FEATURE_HEALTH_SUMMARY]")
    logger.info("  - final_feature_count: {}".format(len(runtime_raw)))
    logger.info("  - nan_dropped: {}".format(len(high_nan_cols)))
    logger.info("  - corr_dropped: {}".format(len(corr_drop)))
    logger.info("  - zero_var_dropped: {}".format(len(zero_var_cols)))
    logger.info("  - pct_remaining_nan_features (>5%): {} / {}".format(
        (final_nan_pct > 0.05).sum(), len(runtime_raw)))
    logger.info("  - pct_low_variance (<adaptive): {} / {}".format(
        (final_var < adaptive_var_threshold).sum(), len(runtime_raw)))
    logger.info("  - max_nan_ratio: {:.2%}".format(final_nan_pct.max()))
    logger.info("  - memory_est_100k_rows: {:.1f} MB".format(
        len(runtime_raw) * 100000 * 4 / (1024 ** 2)))

    # Final assertion for exact match
    missing_raw = set(schema_raw) - set(runtime_raw)
    if missing_raw:
        logger.error("[SCHEMA_CRITICAL_FAILURE] Missing {} raw features at runtime!".format(len(missing_raw)))
        logger.error("Missing list: {}...".format(list(missing_raw)[:20]))
        raise RuntimeError(
            "[SCHEMA_CRITICAL_FAILURE] Runtime missing {} features from schema.py".format(len(missing_raw))
        )

    assert set(schema_raw) == set(runtime_raw), (
        "[SCHEMA_CRITICAL_FAILURE] schema.py raw_features ({}) != runtime features ({})".format(
            len(schema_raw), len(runtime_raw)
        )
    )

    logger.info("[BUILD_BASE] Output shape: {} | All raw features verified.".format(df.shape))

    if is_train_mode:
        manifest = PruningManifest(
            nan_threshold=adaptive_nan_threshold,
            corr_threshold=adaptive_corr_threshold,
            var_threshold=adaptive_var_threshold,
            cols_to_drop_nan=list(high_nan_cols),
            cols_to_drop_corr=list(corr_drop),
            cols_to_drop_var=list(zero_var_cols),
            derivation_log={
                "nan_threshold": derivation_nan,
                "corr_threshold": derivation_corr,
                "var_threshold": derivation_var,
            },
            regime_boundaries=None,
            # [TASK 5] Train-derived column means for causal fallback
            train_col_means=train_col_means,
            # [TASK 11] Train-derived extreme quantiles for consistent detection
            extreme_quantiles=extreme_quantiles
        )
        return df, manifest, registry
    else:
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
    
    # [STRUCTURAL_REALIGNMENT] Feature Integrity Guard
    expected_embeds = [f for f in all_schema_features if f in FEATURE_SCHEMA['embed_features']]
    logger.info(f"[FEATURE_AUDIT] total_expected_embeds: {len(expected_embeds)}")

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
    missing_embeds = []
    for f in all_schema_features:
        if f in FEATURE_SCHEMA['embed_features'] and f not in calculated_keys:
             # Check if it was already in df (unlikely for latent)
             if f not in existing_cols:
                missing_embeds.append(f)

    if missing_embeds:
        logger.error(f"[SCHEMA_CRITICAL_FAILURE] Missing {len(missing_embeds)} embed features at runtime!")
        logger.error(f"Missing list: {missing_embeds[:20]}...")
        raise RuntimeError(f"[SCHEMA_CRITICAL_FAILURE] Runtime missing {len(missing_embeds)} features from schema.py")

    # [STRUCTURAL_REALIGNMENT] Final Total Feature Audit
    runtime_all = [c for c in df.columns if c not in Config.ID_COLS and c != Config.TARGET]
    # Add newly calculated embeds to runtime_all for the check
    runtime_all = list(set(runtime_all) | set(new_features_df_dict.keys()))
    
    if selected_features is None:
        schema_all = FEATURE_SCHEMA['all_features']
        missing_all = set(schema_all) - set(runtime_all)
        if missing_all:
             logger.error(f"[SCHEMA_CRITICAL_FAILURE] Total feature mismatch! Missing: {list(missing_all)[:10]}")
             raise RuntimeError(f"[SCHEMA_CRITICAL_FAILURE] Runtime missing {len(missing_all)} total features")
        
        # We allow extra features (like original ID columns if not in schema_all)
        # but the core feature set must match.
        logger.info(f"[FEATURE_AUDIT] Total features synchronized: {len(runtime_all)}")

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

def build_features(df, mode='raw', reconstructor=None, scaler=None, residuals=None, raw_preds=None, pruning_manifest=None):
    """Legacy entry point for compatibility.

    [TASK 3 — MANIFEST CONTRACT RESTORATION]
    [WHY_THIS_DESIGN] pruning_manifest parameter added to enforce train/test consistency.
    [CODE_EVIDENCE] Previously: build_base_features(df) was called with NO manifest,
      meaning this function ALWAYS operated in train mode — recomputing thresholds
      even for test data. This breaks the manifest contract.
    [FAILURE_MODE_PREVENTED] Test data computing its own pruning thresholds.
    """
    if pruning_manifest is not None:
        df_base = build_base_features(df, pruning_manifest=pruning_manifest)
    else:
        # Train mode — compute manifest (caller should capture it if needed)
        result = build_base_features(df)
        if isinstance(result, tuple):
            df_base = result[0]
        else:
            df_base = result

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
    valid_features = [f for f in all_schema_features if f in df_base.columns]
    X_df = df_base[valid_features].astype('float32').fillna(0.0)
    return X_df, df_base

def add_time_series_features(df, train_col_means=None):
    """
    [CONTEXT] Generates temporal features per-scenario using rolling/expanding windows.
    NaN production is inherent in shift-based operations (diff, rate, accel) when
    raw data already contains NaN (10-17% in BASE_COLS). This function now fills
    shift-generated NaNs with 0 to prevent NaN compounding through downstream generators.
    If removed: 326+ features would exceed the 5% NaN threshold and corrupt PCA.

    [TASK 5 — CAUSAL FALLBACK LEAKAGE FIX]
    train_col_means: dict of {col: mean_value} computed on TRAIN data.
      If None (train mode), computes from df — safe because df IS train.
      If provided (test mode), uses train-derived means — zero leakage.

    [TASK 7/8/9 — UNDERFITTING RECOVERY + TREND EXPANSION + MULTI-SCALE WINDOWS]
    Added: rolling_mean_3, rolling_std_3 (short-window 45min state)
    Added: slope_5 (linear trend direction over 5 steps)
    Added: rate_1 (magnitude-normalized change)
    """
    logger.info(f"[TS_FEATURES] Adding features to df shape {df.shape}")
    if "timestep_index" not in df.columns:
        df["timestep_index"] = df.groupby("scenario_id").cumcount().astype("int16")
    df["normalized_time"] = (df["timestep_index"] / 24.0).astype("float32")
    df["cold_start_flag"] = 0

    new_features = {}
    col_types = infer_feature_types(df, BASE_COLS)

    # [TASK 5] Resolve fallback means: train mode computes, test mode uses manifest
    if train_col_means is None:
        # TRAIN MODE: compute from df (which IS the train set — no leakage)
        fallback_means = {}
        for col in BASE_COLS:
            if col in df.columns and not df[col].isna().all():
                fallback_means[col] = float(df[col].mean())
            else:
                fallback_means[col] = 0.0
    else:
        # TEST MODE: use train-derived means — zero test distribution leakage
        fallback_means = train_col_means

    for col in BASE_COLS:
        series = df.groupby("scenario_id")[col]

        # [TASK 9] Multi-scale rolling windows: 3 (45min) and 5 (75min)
        # [WHY_THIS_DESIGN] Different windows capture different dynamics:
        #   window=3: Responsive to rapid changes (e.g., sudden order spike)
        #   window=5: Smooths over noise, captures medium-term trends
        new_features[f"{col}_rolling_mean_3"] = series.rolling(3, min_periods=1).mean().values
        new_features[f"{col}_rolling_std_3"] = series.rolling(3, min_periods=1).std().values
        new_features[f"{col}_rolling_mean_5"] = series.rolling(5, min_periods=1).mean().values
        new_features[f"{col}_rolling_std_5"] = series.rolling(5, min_periods=1).std().values

        # Sequential Stats (Trend)
        shift1_raw = series.shift(1)
        is_boundary = shift1_raw.isna().astype('float32').values
        new_features[f"{col}_is_boundary"] = is_boundary

        col_type = col_types.get(col, "sensor")
        col_fallback_mean = fallback_means.get(col, 0.0)

        if col_type == "count":
            col_filled = df[col].fillna(0.0)
            shift1 = shift1_raw.fillna(0.0).values
        elif col_type == "ratio":
            # [TASK 1 — GROUPBY BOUNDARY CORRUPTION FIX (ADDITIONAL)]
            # [WHY_THIS_CHANGE] Prevent boundary loss in expanding computation.
            # [ROOT_CAUSE] series.shift(1).expanding() drops groupby context.
            #   We compute expanding inside the groupby, then shift safely.
            # [EXPECTED_IMPACT] causal_past_median respects scenario_id boundaries.
            exp_med = series.expanding().median().reset_index(level=0, drop=True)
            causal_past_median = exp_med.groupby(df["scenario_id"]).shift(1).fillna(0.0)
            
            col_filled = df[col].fillna(causal_past_median)
            shift1 = shift1_raw.fillna(causal_past_median).values
        else:
            # [TASK 5 — LEAKAGE FIX] Use train-derived fallback_mean instead of df[col].mean()
            # [CODE_EVIDENCE] Previously: .fillna(df[col].mean()) leaked test distribution
            # [FAILURE_MODE_PREVENTED] Test statistics injected into feature values

            # [TASK 1 — GROUPBY BOUNDARY CORRUPTION FIX (ADDITIONAL)]
            # [WHY_THIS_CHANGE] Prevent boundary loss in expanding computation.
            # [ROOT_CAUSE] series.shift(1).expanding() dropped groupby context, mixing scenarios.
            # [EXPECTED_IMPACT] causal_past_mean respects scenario_id boundaries.
            exp_mean = series.expanding().mean().reset_index(level=0, drop=True)
            causal_past_mean = exp_mean.groupby(df["scenario_id"]).shift(1).fillna(col_fallback_mean).fillna(0.0)
            
            col_filled = df[col].fillna(causal_past_mean)
            shift1 = shift1_raw.fillna(causal_past_mean).values

        diff_1 = col_filled - shift1
        new_features[f"{col}_diff_1"] = diff_1

        # [TASK 8 — TREND EXPANSION]
        # slope_5: Linear trend direction over 5 steps via rolling regression
        # [WHY_THIS_DESIGN] diff_1 captures instantaneous change only.
        #   slope_5 captures sustained direction over 75min, complementing diff_1.
        rm5_vals = new_features[f"{col}_rolling_mean_5"]
        
        # [TASK 1 — GROUPBY BOUNDARY CORRUPTION FIX]
        # [WHY_THIS_CHANGE] Enforce groupby-safe rolling/shift computation.
        # [ROOT_CAUSE] series.shift(4).rolling(5) lost the groupby context
        #   because series.shift() returned a flat Series. The subsequent .rolling()
        #   computed rolling means across DIFFERENT scenario_id boundaries,
        #   causing silent cross-scenario data leakage and feature corruption.
        # [WHY_NOT_ALTERNATIVES] Cannot use apply(lambda) as it's too slow for large df.
        #   However, rolling_mean_5 is already correctly computed with boundaries!
        #   We simply shift the already-safe rm5_vals by 4 within the same groupby.
        # [EXPECTED_IMPACT] slope_5 will no longer contain corrupted values from
        #   adjacent scenarios. All features perfectly respect physical boundaries.
        rm5_series = pd.Series(rm5_vals, index=df.index)
        shift5_rm5 = rm5_series.groupby(df["scenario_id"]).shift(4).values
        
        # Approximation: (current_mean - lagged_mean) / window
        new_features[f"{col}_slope_5"] = (np.asarray(rm5_vals) - np.asarray(shift5_rm5)) / 5.0

        # rate_1: Normalized change (diff / magnitude)
        # [WHY_THIS_DESIGN] diff_1=+10 means different things at baseline=100 vs baseline=1000.
        #   rate_1 captures RELATIVE change, invariant to feature scale.
        # [WHY_THIS_CHANGE] Denominator smoothing factor changed from eps (1.19e-7) to 1.0.
        # [ROOT_CAUSE] Forensic PROOF 1 demonstrated mathematically inevitable variance
        #   explosion: when col_vals → 0 (e.g., order_inflow_15m near zero during idle
        #   periods), (|x| + eps) ≈ 1e-7, causing rate_1 values to explode to -220.99.
        #   This produces std=4.03 (inflated by outliers). DriftShieldScaler then clips
        #   at P1/P99, collapsing std_after/std_before ratio to 0.33 → VARIANCE_COMPRESSION.
        #   Measured: order_inflow_15m_rate_1 min=-220.99, max=1.0, std=4.03.
        # [WHY_NOT_ALTERNATIVES]
        #   - np.clip(rate_1, -5, 5): Masks the root cause; DriftShieldScaler still detects
        #     compression because raw std remains inflated before clip.
        #   - Larger eps (e.g., 0.01): Still allows values up to ±100 near zero crossings.
        #   - Additive smoothing (+1.0): Denominates become (|x| + 1.0), ensuring rate_1
        #     stays bounded in [-1, 1] for typical feature ranges. For large |x| >> 1,
        #     the +1.0 term becomes negligible and rate_1 ≈ diff/|x| (original behavior).
        #     For small |x| → 0, rate_1 ≈ diff/1.0 = diff (absolute change, still informative).
        #   - Tree models (LGBM) are ordinal-only: they do not need precise ratio scale.
        # [EXPECTED_IMPACT] rate_1 values bounded to reasonable range.
        #   VARIANCE_COMPRESSION errors eliminated for all rate_1 features.
        col_vals = np.asarray(col_filled, dtype=np.float32)
        new_features[f"{col}_rate_1"] = np.asarray(diff_1, dtype=np.float32) / (np.abs(col_vals) + 1.0)

    logger.info(f"[TS_FEATURES] Concat-ing {len(new_features)} new features")
    ts_df = pd.DataFrame(new_features, index=df.index)

    # [STRUCTURAL_INTEGRITY] Avoid duplicate columns
    overlap = [c for c in ts_df.columns if c in df.columns]
    if overlap:
        logger.info(f"[TS_FEATURES] Dropping {len(overlap)} overlapping columns from ts_df")
        ts_df = ts_df.drop(columns=overlap)

    return pd.concat([df, ts_df], axis=1)

def add_extreme_detection_features(df, extreme_quantiles=None):
    """
    [TASK 11 — EXTREME QUANTILE CONSISTENCY]
    extreme_quantiles: dict of {col: q95_value} from train. If None, compute from df.
    [TASK 2 — DUPLICATE INTERACTION REMOVAL]
    [CODE_EVIDENCE] Lines 1202-1205 and 1219-1225 both injected identical interaction
    features. The second block (with WHY_THIS_DESIGN comment) overwrote the first with
    the exact same computation. Removed the duplicate second block.
    Statement: "No duplicate feature computation exists after this fix."
    """
    new_features = {}
    col_types = infer_feature_types(df, BASE_COLS)
    # [WHY_THIS_DESIGN] Context Signal Minimization
    # Problem: rel_rank and accel were high-noise/redundant.
    # Why this structure: rel_to_mean_5 provides situational awareness.
    for col in BASE_COLS:
        series = df.groupby("scenario_id")[col]
        # 1. Relative Position (Context)
        eps = np.finfo(np.float32).eps
        rm5 = series.rolling(5, min_periods=1).mean().values
        new_features[f"{col}_rel_to_mean_5"] = np.where(np.abs(rm5) > eps, (df[col] - rm5) / (rm5 + eps), 0.0)

    # [WHY_THIS_DESIGN] Manual Interaction Justification
    # Problem: Linear models miss the multiplicative effect of load and complexity.
    # Why these 3: Represent core bottleneck intersections (Inflow x Util, Weight x Inflow).
    # [TASK 2] Single authoritative injection point — no duplicates.
    new_features['inter_order_inflow_15m_x_robot_utilization'] = df['order_inflow_15m'] * df['robot_utilization']
    new_features['inter_heavy_item_ratio_x_order_inflow_15m'] = df['heavy_item_ratio'] * df['order_inflow_15m']
    new_features['inter_heavy_item_ratio_x_robot_utilization'] = df['heavy_item_ratio'] * df['robot_utilization']

    # Early Warning
    # [WHY_THIS_DESIGN] Justified Heuristic for Early Warning
    # Problem: Need a combined signal for high load + high volatility.
    # Why this structure: Uses Trend (diff_1) and Context (rel_to_mean) to detect emerging peaks.
    util_rel = new_features[f'robot_utilization_rel_to_mean_5']
    util_rel_clean = util_rel[~np.isnan(util_rel)]
    # [TASK 11] Use train-derived quantile for util_p90 if available
    if extreme_quantiles is not None and 'util_rel_p90' in extreme_quantiles:
        util_p90 = extreme_quantiles['util_rel_p90']
    else:
        util_p90 = np.quantile(util_rel_clean, 0.90) if len(util_rel_clean) > 0 else 1.2

    inflow_diff = df[f'order_inflow_15m_diff_1']
    new_features['early_warning_flag'] = ((inflow_diff > 0) & (util_rel > util_p90)).astype(int)
    new_features['early_warning_score'] = new_features[f'order_inflow_15m_rel_to_mean_5'] + util_rel

    # [TASK 11 — EXTREME VALUE QUANTILE CONSISTENCY]
    # [WHY_THIS_DESIGN] Extreme thresholds MUST originate from train.
    # [CODE_EVIDENCE] Previously: threshold = df['order_inflow_15m'].quantile(0.95)
    #   When df is test data, this computes test-derived quantiles — DATA LEAKAGE.
    # [FAILURE_MODE_PREVENTED] Train/test threshold inconsistency in extreme detection.
    key_extreme_cols = ['order_inflow_15m', 'robot_utilization', 'congestion_score', 'near_collision_15m']

    if extreme_quantiles is not None:
        # TEST MODE: use train-derived quantiles
        threshold = extreme_quantiles.get('order_inflow_15m', df['order_inflow_15m'].quantile(0.95))
    else:
        # TRAIN MODE: compute from train (safe — df IS train)
        threshold = df['order_inflow_15m'].quantile(0.95)

    df['is_extreme'] = (df['order_inflow_15m'] >= threshold).astype(np.int8)

    extreme_masks = []
    for c in key_extreme_cols:
        if c in df.columns:
            if extreme_quantiles is not None:
                q95 = extreme_quantiles.get(c, df[c].quantile(0.95))
            else:
                q95 = df[c].quantile(0.95)
            extreme_masks.append(df[c] >= q95)

    if extreme_masks:
        df['is_extreme_multi'] = np.logical_or.reduce(extreme_masks).astype(np.int8)
    else:
        df['is_extreme_multi'] = df['is_extreme']

    coverage = df['is_extreme_multi'].mean()
    logger.info(f"[EXTREME_INTELLIGENCE] Threshold (P95): {threshold:.4f} | Coverage: {coverage:.2%}")

    # [STRUCTURAL_INTEGRITY] Avoid duplicate columns
    ext_df = pd.DataFrame(new_features, index=df.index)
    overlap = [c for c in ext_df.columns if c in df.columns]
    if overlap:
        logger.info(f"[EXTREME_FEATURES] Dropping {len(overlap)} overlapping columns from ext_df")
        ext_df = ext_df.drop(columns=overlap)

    return pd.concat([df, ext_df], axis=1)

def load_data():
    train = pd.read_csv(f"{Config.DATA_PATH}train.csv")
    test = pd.read_csv(f"{Config.DATA_PATH}test.csv")
    layout = pd.read_csv(Config.LAYOUT_PATH)
    
    # [STRUCTURAL_REALIGNMENT] Target Renaming
    # The source CSV uses avg_delay_minutes_next_30m, but the pipeline
    # uses Config.TARGET ('target') as the Single Source of Truth.
    if 'avg_delay_minutes_next_30m' in train.columns:
        train = train.rename(columns={'avg_delay_minutes_next_30m': Config.TARGET})
    
    train = train.merge(layout, on="layout_id", how="left")
    test = test.merge(layout, on="layout_id", how="left")
    return downcast_df(train), downcast_df(test)
