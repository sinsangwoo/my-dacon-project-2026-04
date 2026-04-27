
import logging
import pandas as pd
import numpy as np
from src.schema import FEATURE_SCHEMA, BASE_COLS
from src.config import Config
from src.data_loader import build_base_features, apply_latent_features, SuperchargedPCAReconstructor

logger = logging.getLogger("STRUCTURAL_AUDIT")

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

# [CONTEXT] Added support for real data validation to detect statistical issues
# beyond simple schema matching. This prevents deploying structurally valid but
# statistically invalid pipelines.

def run_structural_audit(mode='real_data'):
    logger.info("\n" + "="*60)
    logger.info(" [MISSION: STRUCTURAL REALIGNMENT AUDIT]")
    logger.info("="*60)
    
    results = {
        "schema_match": "FAIL",
        "feature_count": 0,
        "PCA_mode": "unknown",
        "pipeline_status": "BROKEN"
    }
    
    try:
        # [CONTEXT] Dummy data was insufficient to catch variance collapse or NaNs.
        # mode='real_data' uses a subset of actual data to provide realistic validation.
        if mode == 'dummy':
            # 1. Dummy Data Generation
            dummy_df = pd.DataFrame(np.random.randn(100, len(BASE_COLS)), columns=BASE_COLS)
            dummy_df['ID'] = range(100)
            dummy_df['scenario_id'] = [1]*50 + [2]*50
            dummy_df['layout_id'] = 1
            df_input = dummy_df
        else:
            # Load a SMALL SAMPLE of real train data
            full_train = pd.read_csv(os.path.join(Config.DATA_PATH, "train.csv"), nrows=500)
            layout = pd.read_csv(Config.LAYOUT_PATH)
            full_train = full_train.merge(layout, on="layout_id", how="left")
            # Downcast not strictly necessary for 500 rows, but safe
            df_input = full_train
            
        # 2. Trace Base Features
        logger.info("[AUDIT] Tracing Base Feature Generation...")
        df_base = build_base_features(df_input)
        
        # 3. Trace Latent Features
        logger.info("[AUDIT] Tracing Latent Feature Generation...")
        reconstructor = SuperchargedPCAReconstructor(input_dim=len(Config.EMBED_BASE_COLS))
        reconstructor.fit(df_base[Config.PCA_INPUT_COLS].values, residuals=np.zeros(len(df_base)))
        reconstructor.build_fold_cache(df_base)
        
        df_full = apply_latent_features(df_base, reconstructor)
        
        # 4. Final Comparison
        runtime_features = [c for c in df_full.columns if c not in Config.ID_COLS and c != Config.TARGET]
        schema_features = FEATURE_SCHEMA['all_features']
        
        results["feature_count"] = len(runtime_features)
        results["PCA_mode"] = reconstructor.pca_mode
        
        if set(runtime_features) == set(schema_features):
            results["schema_match"] = "PASS"
            results["pipeline_status"] = "RUNNABLE"
            logger.info("[AUDIT] SUCCESS: Runtime matches Schema 100%.")
        else:
            missing = set(schema_features) - set(runtime_features)
            extra = set(runtime_features) - set(schema_features)
            logger.error(f"[AUDIT] FAIL: Mismatch! Missing: {len(missing)} | Extra: {len(extra)}")
            
        if mode == 'real_data':
            # [CONTEXT] Ensures real-world statistical validity by throwing explicit RuntimeErrors
            # on dangerous data structures (missing data, zero variance, extreme distributions).
            
            # 1. Missing / NaN Checks
            nan_ratio = df_full[runtime_features].isna().mean()
            high_nan_features = nan_ratio[nan_ratio > 0.05].index.tolist()
            
            # 2. Variance Checks
            variances = df_full[runtime_features].var()
            zero_variance_features = variances[variances == 0].index.tolist()
            low_variance_features = variances[variances < 1e-4].index.tolist()
            
            # 3. Correlation Checks
            sample_for_corr = df_full[runtime_features].sample(min(500, len(df_full)))
            corr_matrix = sample_for_corr.corr().abs()
            corr_values = corr_matrix.to_numpy(copy=True)
            np.fill_diagonal(corr_values, 0)
            high_corr_pairs = np.where(corr_values > 0.98)
            highly_correlated = list(set([corr_matrix.index[i] for i in high_corr_pairs[0]]))
            
            # 4. Extreme clipping check
            clip_check = df_full[runtime_features].apply(lambda x: (np.abs(x) > 1e6).sum())
            extreme_clipping_features = clip_check[clip_check > 0].index.tolist()
            
            logger.info("\n[REALITY_AUDIT]")
            logger.info(f"- feature_count: {len(runtime_features)}")
            logger.info(f"- zero_variance_features: {len(zero_variance_features)}")
            logger.info(f"- high_nan_features (>5%): {len(high_nan_features)}")
            logger.info(f"- extreme_clipping_features: {len(extreme_clipping_features)}")
            
            pct_near_constant = len(low_variance_features) / len(runtime_features)
            pct_high_corr = len(highly_correlated) / len(runtime_features)
            
            logger.info(f"\n[NOISE_ALERT]")
            logger.info(f"- % near-constant features: {pct_near_constant:.1%}")
            logger.info(f"- % highly correlated features (>0.98): {pct_high_corr:.1%}")
            
            if pct_near_constant > 0.3 or pct_high_corr > 0.4:
                logger.warning("[NOISE_ALERT] Significant noise detected in feature space. Consider aggressive pruning.")

            if len(zero_variance_features) > 0 or len(high_nan_features) > 0 or len(extreme_clipping_features) > 0:
                raise RuntimeError(f"[REALITY_AUDIT_FAIL] Invalid features detected! ZeroVar: {len(zero_variance_features)}, HighNaN: {len(high_nan_features)}, ExtremeClip: {len(extreme_clipping_features)}")


        logger.info("\n[STRUCTURAL_AUDIT]")
        logger.info(f"- schema_match: {results['schema_match']}")
        logger.info(f"- feature_count: {results['feature_count']}")
        logger.info(f"- PCA_mode: {results['PCA_mode']}")
        logger.info(f"- pipeline_status: {results['pipeline_status']}")
        logger.info("="*60 + "\n")
        
        return results["pipeline_status"] == "RUNNABLE"
        
    except Exception as e:
        logger.error(f"[AUDIT] CRITICAL FAILURE during trace: {str(e)}")
        logger.info("\n[STRUCTURAL_AUDIT]")
        logger.info(f"- pipeline_status: BROKEN")
        logger.info("="*60 + "\n")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    mode = 'real_data' if '--dummy' not in sys.argv else 'dummy'
    run_structural_audit(mode=mode)
