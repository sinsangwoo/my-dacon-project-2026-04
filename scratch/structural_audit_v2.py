
import pandas as pd
import numpy as np
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('src'))

# Mock the package structure for data_loader
import config
import schema
import utils
sys.modules['src.config'] = config
sys.modules['src.schema'] = schema
sys.modules['src.utils'] = utils

from data_loader import build_base_features, apply_latent_features, SuperchargedPCAReconstructor
from schema import FEATURE_SCHEMA, BASE_COLS
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AUDIT")

def run_audit():
    logger.info("Starting Structural Audit...")
    
    # 1. Create dummy data
    dummy_data = pd.DataFrame(np.random.randn(100, len(BASE_COLS)), columns=BASE_COLS)
    dummy_data['ID'] = range(100)
    dummy_data['scenario_id'] = [1]*50 + [2]*50
    dummy_data['layout_id'] = 1
    
    # 2. Trace Base Feature Generation
    logger.info("Tracing build_base_features...")
    df_base = build_base_features(dummy_data)
    
    runtime_raw = [c for c in df_base.columns if c not in Config.ID_COLS and c != Config.TARGET]
    schema_raw = FEATURE_SCHEMA['raw_features']
    
    missing_raw = set(schema_raw) - set(runtime_raw)
    extra_raw = set(runtime_raw) - set(schema_raw)
    
    logger.info(f"Raw Features: Schema={len(schema_raw)}, Runtime={len(runtime_raw)}")
    if missing_raw:
        logger.error(f"MISSING RAW FEATURES ({len(missing_raw)}): {list(missing_raw)[:10]}")
    if extra_raw:
        logger.warning(f"EXTRA RAW FEATURES ({len(extra_raw)}): {list(extra_raw)[:10]}")
        
    # 3. Trace Latent Feature Generation
    logger.info("Tracing apply_latent_features...")
    reconstructor = SuperchargedPCAReconstructor(input_dim=len(Config.EMBED_BASE_COLS))
    
    # Mock residuals for fit
    residuals = np.random.randn(100)
    reconstructor.fit(df_base, residuals=residuals)
    reconstructor.build_fold_cache(df_base)
    
    df_full = apply_latent_features(df_base, reconstructor)
    
    runtime_all = [c for c in df_full.columns if c not in Config.ID_COLS and c != Config.TARGET]
    schema_all = FEATURE_SCHEMA['all_features']
    
    missing_all = set(schema_all) - set(runtime_all)
    extra_all = set(runtime_all) - set(schema_all)
    
    logger.info(f"Total Features: Schema={len(schema_all)}, Runtime={len(runtime_all)}")
    
    if not missing_all and len(runtime_all) == len(schema_all):
        logger.info("[AUDIT_RESULT] PASS: Schema and Runtime are synchronized.")
    else:
        logger.error(f"[AUDIT_RESULT] FAIL: Mismatch detected.")
        if missing_all:
             logger.error(f"MISSING TOTAL FEATURES ({len(missing_all)}): {list(missing_all)[:10]}")
        if extra_all:
             logger.warning(f"EXTRA TOTAL FEATURES ({len(extra_all)}): {list(extra_all)[:10]}")

    # 4. PCA Fallback Test
    logger.info("Testing PCA Fallback...")
    reconstructor_bad = SuperchargedPCAReconstructor(input_dim=len(Config.EMBED_BASE_COLS))
    # Fit with constant data to trigger low variance or failure if possible
    bad_data = df_base.copy()
    bad_data[Config.EMBED_BASE_COLS] = 0
    reconstructor_bad.fit(bad_data, residuals=residuals)
    logger.info(f"PCA Mode after bad fit: {reconstructor_bad.pca_mode}")

if __name__ == "__main__":
    run_audit()
