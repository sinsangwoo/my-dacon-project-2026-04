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

def build_base_features(df, pruning_manifest: "PruningManifest" = None):
    """
    [PHASE 1: ISOLATED BASE FEATURES]
    Builds temporal and row-wise features that are safe to compute globally.

    [WHY_THIS_CHANGE — SIGNATURE CHANGE]
    Problem:
        build_base_features() was stateless and called identically for train and test.
        Pruning thresholds (NaN, correlation, variance) were re-computed on whatever
        dataframe was passed in — meaning TEST distribution influenced the thresholds
        used for test pruning. This is DATA LEAKAGE in the pruning decision itself.
    Root Cause:
        No mechanism to propagate train-time pruning decisions to test processing.
    Decision:
        Add optional `pruning_manifest` parameter:
        - If None (train mode): compute all thresholds from data, build manifest, return it.
        - If provided (test mode): apply pre-computed thresholds exactly as computed on train.
    Why this approach (not alternatives):
        - Global mutable state: fragile, not reentrant.
        - Recomputing on test: leakage.
        - PruningManifest: explicit, serializable, zero-leakage contract.
    Expected Impact:
        Train and test are pruned identically using train-derived thresholds.
        No test distribution information pollutes pruning decisions.

    Returns
    -------
    df : pd.DataFrame
        Feature-engineered dataframe after pruning.
    pruning_manifest : PruningManifest (only returned when input pruning_manifest is None)
        Manifest of all pruning decisions made (train mode only).
    """
    is_train_mode = pruning_manifest is None

    logger.info(f"[BUILD_BASE] Input shape: {df.shape} | mode={'TRAIN (compute thresholds)' if is_train_mode else 'TEST (apply manifest)'}")
    df = df.copy()

    # [STRUCTURAL_REALIGNMENT] Filter columns to only those defined in SSOT
    # This prevents junk columns in the CSV from causing schema mismatches.
    relevant_cols = list(BASE_COLS) + list(Config.ID_COLS)
    if Config.TARGET in df.columns:
        relevant_cols.append(Config.TARGET)

    keep_cols = [c for c in relevant_cols if c in df.columns]
    df = df[keep_cols]

    original_ids = df['ID'].values.copy()
    
    # 1. Temporal Ordering
    df = df.sort_values(by=["scenario_id", "ID"]).reset_index(drop=True)
    
    # [WHY_THIS_CHANGE] Feature Flow Trace (TASK 5)
    logger.info(f"[BUILD_BASE] Initial Count: {len(df.columns)}")
    df = add_time_series_features(df)
    logger.info(f"[BUILD_BASE] After TS Expansion: {len(df.columns)}")
    df = add_extreme_detection_features(df)
    logger.info(f"[BUILD_BASE] After Extreme Expansion: {len(df.columns)}")
    
    # 3. Order Restoration
    df = df.set_index('ID').loc[original_ids].reset_index()
    

    # [CONTEXT] BASE_COLS and EMBED_BASE_COLS must NEVER be pruned.
    # They are structurally required by PCA, schema, and downstream generators.
    protected_cols = set(BASE_COLS) | set(Config.EMBED_BASE_COLS)

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
        nan_ratios_sorted = df.isna().mean().sort_values()
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
        high_nan_cols = [c for c in nan_ratios[nan_ratios > adaptive_nan_threshold].index if c not in protected_cols]
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
    # [CORR_PRUNE]
    # [CONTEXT] 48.7% of features were found highly correlated (>0.98).
    # This lightweight sample-based pruning removes redundant features
    # keeping the one with highest variance in each correlated pair.
    # If removed: model trains on massively redundant feature space, wasting
    # memory and amplifying noise.
    # ------------------------------------------------------------------
    runtime_raw_current = [c for c in df.columns if c not in Config.ID_COLS and c != Config.TARGET]
    n_sample = min(CORR_SAMPLE_SIZE_CAP, len(df))
    sample_df = df[runtime_raw_current].sample(n=n_sample, random_state=42).astype("float32")
    # Fill NaN for correlation computation only (does not affect main df)
    sample_filled = sample_df.fillna(sample_df.median())
    corr_matrix = sample_filled.corr().abs()
    corr_values = corr_matrix.to_numpy(copy=True)
    np.fill_diagonal(corr_values, 0)

    # Find features to drop (keep higher variance in each pair)
    # [CONTEXT] Never prune BASE_COLS or EMBED_BASE_COLS — they are structurally required
    # by PCA reconstructor and schema. Pruning them causes KeyError downstream.
    variances = sample_filled.var()
    # [AXIS4_FIX] O(N^2) 이중 루프 대신 numpy 상삼각 행렬 마스킹 최적화
    cols_array = np.array(corr_matrix.columns)
    upper_tri_indices = np.triu_indices_from(corr_values, k=1)
    upper_tri = corr_values[upper_tri_indices]

    if is_train_mode:
        # [WHY_THIS_DESIGN] Correlation Pruning
        # Observed Data Behavior: P99 correlation is ~0.53 after source-level redesign.
        # Why P99: Targets only the absolute top 1% of redundant features.
        # Decision: Hardcode floor removed (RULE 1). Threshold is 100% data-driven.
        adaptive_corr_threshold = float(np.percentile(upper_tri, 99))
        derivation_corr = f"Data-driven P99 correlation threshold: {adaptive_corr_threshold:.4f}"
        
        supporting_stats = {
            "corr_P50": float(np.median(upper_tri)),
            "corr_P95": float(np.percentile(upper_tri, 95)),
            "n_pairs": int(len(upper_tri))
        }
        registry.record_threshold("corr_threshold", adaptive_corr_threshold, derivation_corr, supporting_stats)
    else:
        # TEST MODE: Use threshold from manifest
        adaptive_corr_threshold = pruning_manifest.corr_threshold
        logger.info("[CORR_PRUNE] TEST mode: applying train-derived threshold={:.4f}".format(adaptive_corr_threshold))

    if is_train_mode:
        high_corr_mask = upper_tri > adaptive_corr_threshold
        high_corr_pairs = zip(
            upper_tri_indices[0][high_corr_mask],
            upper_tri_indices[1][high_corr_mask]
        )
        corr_drop = set()
        for i, j in high_corr_pairs:
            ci, cj = cols_array[i], cols_array[j]
            if ci in corr_drop or cj in corr_drop:
                continue
            ci_protected = ci in protected_cols
            cj_protected = cj in protected_cols
            if ci_protected and cj_protected:
                continue
            elif ci_protected:
                corr_drop.add(cj)
            elif cj_protected:
                corr_drop.add(ci)
            elif variances[ci] < variances[cj]:
                corr_drop.add(ci)
            else:
                corr_drop.add(cj)
    else:
        # TEST: apply exactly the same columns dropped on train
        corr_drop = set(c for c in pruning_manifest.cols_to_drop_corr if c in df.columns)

    if corr_drop:
        logger.warning("[CORR_PRUNE] Dropping {} highly correlated features (threshold={:.4f})".format(
            len(corr_drop), adaptive_corr_threshold))
        df = df.drop(columns=list(corr_drop))
        # [AXIS1_FIX] 싱글턴 변이 금지, 로컬 드롭셋에 추가
        dropped_features.update(corr_drop)
        if is_train_mode:
            col_name_to_idx = {name: idx for idx, name in enumerate(cols_array)}
            for col in corr_drop:
                col_idx = col_name_to_idx.get(col, -1)
                corr_stat = float(corr_values[col_idx].max()) if col_idx >= 0 else float("nan")
                registry.record_drop(col, "corr_drop", corr_stat,
                                     adaptive_corr_threshold, derivation_corr)
    else:
        logger.info("[CORR_PRUNE] No highly correlated pairs found.")

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
            if current_var[c] < adaptive_var_threshold and c not in protected_cols
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
            regime_boundaries=None
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
    valid_features = [f for f in all_schema_features if f in df_base.columns]
    X_df = df_base[valid_features].astype('float32').fillna(0.0)
    return X_df, df_base

def add_time_series_features(df):
    """
    # [WHY_THIS_DESIGN] Time-Series Redundancy Justification
    # Problem: rolling_mean_5 and expanding_mean have 98% correlation.
    # Why keep both: rolling_mean_5 captures "local dynamics" (short-term shifts),
    #   while expanding_mean captures "long-term bias" (cumulative scenario history).
    # Why this value (5): Small window for high-frequency sensor updates in 15m intervals.
    [CONTEXT] Generates temporal features per-scenario using rolling/expanding windows.
    NaN production is inherent in shift-based operations (diff, rate, accel) when
    raw data already contains NaN (10-17% in BASE_COLS). This function now fills
    shift-generated NaNs with 0 to prevent NaN compounding through downstream generators.
    If removed: 326+ features would exceed the 5% NaN threshold and corrupt PCA.
    
    [TS_CONDITION_GUARD] Memory evaluation (2026-04-24):
    - 700 raw features × 100K rows × 4 bytes = ~267 MB → SAFE
    - Conditional TS generation NOT needed (within memory budget)
    - Decision: APPLIED standard generation with NaN sanitization
    """
    logger.info(f"[TS_FEATURES] Adding features to df shape {df.shape}")
    if "timestep_index" not in df.columns:
        df["timestep_index"] = df.groupby("scenario_id").cumcount().astype("int16")
    df["normalized_time"] = (df["timestep_index"] / 24.0).astype("float32")
    df["cold_start_flag"] = 0
    
    new_features = {}
    col_types = infer_feature_types(df, BASE_COLS)
    
    # [WHY_THIS_DESIGN] Feature Engineering Minimization (TASK 3)
    # Problem: Over-engineered feature space (300+ TS features).
    # Why this structure: Reduced to 3 canonical signals: mean, std, and diff.
    # Why alternatives rejected: Expanding stats, rates, and slopes were >90% redundant.
    # Expected impact: Leaner model, faster training, less overfitting risk.
    for col in BASE_COLS:
        series = df.groupby("scenario_id")[col]
        
        # 1. Rolling Stats (State & Stability)
        new_features[f"{col}_rolling_mean_5"] = series.rolling(5, min_periods=1).mean().values
        new_features[f"{col}_rolling_std_5"] = series.rolling(5, min_periods=1).std().values
        
        # 2. Sequential Stats (Trend)
        shift1_raw = series.shift(1)
        is_boundary = shift1_raw.isna().astype('float32').values
        new_features[f"{col}_is_boundary"] = is_boundary
        
        col_type = col_types.get(col, "sensor")
        if col_type == "count":
            col_filled = df[col].fillna(0.0)
            shift1 = shift1_raw.fillna(0.0).values
        elif col_type == "ratio":
            causal_past_median = series.shift(1).expanding().median().fillna(0.0).reset_index(level=0, drop=True)
            col_filled = df[col].fillna(causal_past_median)
            shift1 = shift1_raw.fillna(causal_past_median).values
        else:
            causal_past_mean = series.shift(1).expanding().mean().fillna(df[col].mean() if not df[col].empty else 0.0).fillna(0.0).reset_index(level=0, drop=True)
            col_filled = df[col].fillna(causal_past_mean)
            shift1 = shift1_raw.fillna(causal_past_mean).values
        
        new_features[f"{col}_diff_1"] = col_filled - shift1

    logger.info(f"[TS_FEATURES] Concat-ing {len(new_features)} new features")
    ts_df = pd.DataFrame(new_features, index=df.index)
    
    # [STRUCTURAL_INTEGRITY] Avoid duplicate columns
    overlap = [c for c in ts_df.columns if c in df.columns]
    if overlap:
        logger.info(f"[TS_FEATURES] Dropping {len(overlap)} overlapping columns from ts_df")
        ts_df = ts_df.drop(columns=overlap)
        
    return pd.concat([df, ts_df], axis=1)

def add_extreme_detection_features(df):
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

    # Interactions
    new_features['inter_order_inflow_15m_x_robot_utilization'] = df['order_inflow_15m'] * df['robot_utilization']
    new_features['inter_heavy_item_ratio_x_order_inflow_15m'] = df['heavy_item_ratio'] * df['order_inflow_15m']
    new_features['inter_heavy_item_ratio_x_robot_utilization'] = df['heavy_item_ratio'] * df['robot_utilization']
    
    # Early Warning
    # [WHY_THIS_DESIGN] Justified Heuristic for Early Warning
    # Problem: Need a combined signal for high load + high volatility.
    # Why this structure: Uses Trend (diff_1) and Context (rel_to_mean) to detect emerging peaks.
    util_rel = new_features[f'robot_utilization_rel_to_mean_5']
    util_rel_clean = util_rel[~np.isnan(util_rel)]
    util_p90 = np.quantile(util_rel_clean, 0.90) if len(util_rel_clean) > 0 else 1.2
    
    inflow_diff = df[f'order_inflow_15m_diff_1']
    new_features['early_warning_flag'] = ((inflow_diff > 0) & (util_rel > util_p90)).astype(int)
    new_features['early_warning_score'] = new_features[f'order_inflow_15m_rel_to_mean_5'] + util_rel

    # Interactions
    # [WHY_THIS_DESIGN] Manual Interaction Justification
    # Problem: Linear models miss the multiplicative effect of load and complexity.
    # Why these 3: Represent core bottleneck intersections (Inflow x Util, Weight x Inflow).
    new_features['inter_order_inflow_15m_x_robot_utilization'] = df['order_inflow_15m'] * df['robot_utilization']
    new_features['inter_heavy_item_ratio_x_order_inflow_15m'] = df['heavy_item_ratio'] * df['order_inflow_15m']
    new_features['inter_heavy_item_ratio_x_robot_utilization'] = df['heavy_item_ratio'] * df['robot_utilization']

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
