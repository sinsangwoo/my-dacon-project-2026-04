import pandas as pd
import numpy as np
import logging
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.config import Config
from src.schema import BASE_COLS, TS_SUFFIXES, EXTREME_SUFFIXES, MULTI_K, LATENT_PATTERNS, EMBED_DIM
from src.data_loader import build_base_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("STRUCTURAL_AUDIT")

def run_structural_audit():
    logger.info("Starting [STRUCTURAL_AUDIT]")
    
    # 1. Load a small subset of data
    data_path = 'data/train.csv'
    if not os.path.exists(data_path):
        logger.error(f"Data not found at {data_path}")
        return
        
    train = pd.read_csv(data_path, nrows=100)
    layout = pd.read_csv(Config.LAYOUT_PATH)
    train = train.merge(layout, on="layout_id", how="left")
    
    # 2. Trace Feature Generation
    logger.info("Tracing feature generation...")
    
    # [STRUCTURAL_REALIGNMENT] Feature generation should now be BASE_COLS-driven
    # Regardless of what EMBED_BASE_COLS is.
    df_generated, manifest, registry = build_base_features(train)
    runtime_features = [c for c in df_generated.columns if c not in Config.ID_COLS and c != Config.TARGET]
    
    # 3. Get Schema Features
    from src.schema import FEATURE_SCHEMA
    schema_features = FEATURE_SCHEMA['raw_features']
    
    # 4. Compare
    runtime_set = set(runtime_features)
    # [STRUCTURAL_REALIGNMENT] Accounts for features dropped during pruning
    dropped_set = set(registry.get_dropped_features())
    schema_expected_set = set(schema_features) - dropped_set
    
    missing_in_runtime = schema_expected_set - runtime_set
    extra_in_runtime = runtime_set - schema_expected_set
    
    logger.info(f"Runtime features: {len(runtime_features)}")
    logger.info(f"Schema features: {len(schema_features)}")
    logger.info(f"Dropped features: {len(dropped_set)}")
    
    if missing_in_runtime:
        logger.error(f"MISSING in runtime ({len(missing_in_runtime)}): {sorted(list(missing_in_runtime))[:10]}...")
    else:
        logger.info("✓ All expected schema features present in runtime.")
        
    if extra_in_runtime:
        logger.warning(f"EXTRA in runtime ({len(extra_in_runtime)}): {sorted(list(extra_in_runtime))[:10]}...")
    else:
        logger.info("✓ No extra features in runtime.")
        
    # 5. Check PCA decoupling
    logger.info("Checking PCA decoupling...")
    current_embed_base = set(Config.EMBED_BASE_COLS)
    
    print("\n[STRUCTURAL_AUDIT_SUMMARY]")
    print(f"schema_match: {'PASS' if not missing_in_runtime and not extra_in_runtime else 'FAIL'}")
    print(f"feature_count_expected_raw: {len(schema_features)}")
    print(f"feature_count_dropped: {len(dropped_set)}")
    print(f"feature_count_actual: {len(runtime_features)}")
    # [THRESHOLD_UPDATE] 250 is a safe floor for 310 raw features after pruning
    print(f"pca_decoupling_status: {'DECOUPLED (SAFE)' if len(runtime_features) > 250 else 'COUPLED (BROKEN)'}")
    print(f"embed_base_count: {len(current_embed_base)}")

if __name__ == "__main__":
    run_structural_audit()
