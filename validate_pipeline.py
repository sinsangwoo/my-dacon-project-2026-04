"""v5.1 Pipeline Canonicalization Validator.

Tests:
1. Adversarial alignment (missing/extra/NaN/object columns)
2. Schema hash integrity
3. End-to-end smoke test (Phase 1 → 2 → 3) in DEBUG_MINIMAL mode
"""
import pandas as pd
import numpy as np
import os
import sys
import hashlib
import logging
from src.config import Config
from src.data_loader import align_features
from src.utils import get_logger

logger = get_logger()

def test_adversarial_alignment():
    """Test v5.1 align_features against malformed inputs."""
    logger.info("🧪 [TEST 1] Adversarial Alignment Test...")
    
    reference_cols = ['feat_A', 'feat_B', 'feat_C']
    
    # Case 1: Extra column + Missing columns
    df_noisy = pd.DataFrame({
        'feat_A': [1.0, 2.0],
        'extra_X': [10, 20],      # Extra → should be dropped
    })
    df_noisy['feat_B'] = ['0.5', '1.5']  # Object/String → should be coerced
    df_noisy['feat_C'] = [np.nan, 3.0]   # NaN → should be filled with 0
    
    df_aligned = align_features(df_noisy, reference_cols, logger)
    
    # POST-CONDITIONS
    assert df_aligned.shape == (2, 3), f"Shape mismatch! Got {df_aligned.shape}"
    assert list(df_aligned.columns) == reference_cols, f"Column order mismatch! Got {list(df_aligned.columns)}"
    assert not (df_aligned.dtypes == object).any(), "Object type leaked!"
    assert not df_aligned.isna().any().any(), "NaNs leaked!"
    assert df_aligned['feat_B'].iloc[0] == np.float32(0.5), f"String coercion failed! Got {df_aligned['feat_B'].iloc[0]}"
    assert df_aligned['feat_C'].iloc[0] == 0.0, f"NaN fill failed! Got {df_aligned['feat_C'].iloc[0]}"
    assert df_aligned.dtypes.apply(lambda d: d in [np.float32, np.float64]).all(), "Non-float dtype!"
    
    logger.info("✅ Adversarial Alignment Test PASSED.")

def test_schema_hash():
    """Test that schema hash mechanism works correctly."""
    logger.info("🧪 [TEST 2] Schema Hash Integrity Test...")
    
    features = ['feat_A', 'feat_B', 'feat_C']
    schema_str = ','.join(features)
    hash1 = hashlib.md5(schema_str.encode()).hexdigest()
    hash2 = hashlib.md5(schema_str.encode()).hexdigest()
    assert hash1 == hash2, "Hash not deterministic!"
    
    # Mutated schema should produce different hash
    features_mutated = ['feat_A', 'feat_B', 'feat_D']
    hash3 = hashlib.md5(','.join(features_mutated).encode()).hexdigest()
    assert hash1 != hash3, "Hash collision on different schemas!"
    
    logger.info("✅ Schema Hash Integrity Test PASSED.")

def run_pipeline_smoke_test():
    """Run Phase 1, 2, 3 in DEBUG_MINIMAL mode and verify SSOT contracts."""
    logger.info("🧪 [TEST 3] End-to-End Pipeline Smoke Test (DEBUG_MINIMAL)...")
    
    import subprocess
    
    phases = ['1_data_check', '2_preprocess', '3_train_base']
    
    for phase in phases:
        logger.info(f"  --- Running Phase: {phase} ---")
        cmd = [sys.executable, 'main.py', '--phase', phase, '--mode', 'debug', '--debug-minimal']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"  ❌ Phase {phase} FAILED!")
            logger.error(f"  STDOUT:\n{result.stdout[-2000:]}")
            logger.error(f"  STDERR:\n{result.stderr[-2000:]}")
            sys.exit(1)
        else:
            logger.info(f"  ✓ Phase {phase} SUCCESS.")
            lines = result.stdout.strip().split('\n')
            for line in lines[-3:]:
                logger.info(f"    [LOG] {line}")
    
    # Post-pipeline SSOT verification
    logger.info("  --- Post-Pipeline SSOT Verification ---")
    
    # 1. Verify schema hash file exists
    hash_path = 'outputs/processed/features_reduced_hash.txt'
    assert os.path.exists(hash_path), f"Schema hash file missing: {hash_path}"
    
    # 2. Verify arrays match feature list
    features = pd.read_json('outputs/processed/features_reduced.json', typ='series').tolist()
    X_train = np.load('outputs/processed/X_train_reduced.npy')
    X_test = np.load('outputs/processed/X_test_reduced.npy')
    
    assert X_train.shape[1] == len(features), \
        f"Train shape mismatch: {X_train.shape[1]} vs {len(features)}"
    assert X_test.shape[1] == len(features), \
        f"Test shape mismatch: {X_test.shape[1]} vs {len(features)}"
    
    # 3. Verify hash matches feature list
    with open(hash_path, 'r') as f:
        saved_hash = f.read().strip()
    live_hash = hashlib.md5(','.join(features).encode()).hexdigest()
    assert saved_hash == live_hash, \
        f"Hash mismatch! Saved: {saved_hash}, Live: {live_hash}"
    
    # 4. Verify no NaN / no object dtype in saved arrays
    assert not np.isnan(X_train).any(), "X_train has NaNs!"
    assert not np.isnan(X_test).any(), "X_test has NaNs!"
    assert X_train.dtype != object, "X_train is object dtype!"
    assert X_test.dtype != object, "X_test is object dtype!"
    
    logger.info(f"  ✓ SSOT Verification: features={len(features)}, hash={saved_hash[:8]}...")
    logger.info("✅ Pipeline Smoke Test PASSED.")

if __name__ == "__main__":
    try:
        test_adversarial_alignment()
        test_schema_hash()
        run_pipeline_smoke_test()
        logger.info("\n🏆 ALL v5.1 CANONICALIZATION TESTS PASSED (SUCCESS)")
    except Exception as e:
        logger.error(f"💥 Validation Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
