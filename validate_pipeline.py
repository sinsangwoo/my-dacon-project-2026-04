"""v16.0 Pipeline Canonicalization Validator.

Tests:
1. Adversarial alignment (missing/extra/NaN/object columns)
2. Schema integrity
3. End-to-end smoke test (Phase 1, 2, 2.5, 3, 5, 7, 8, 9) in SMOKE_TEST mode
"""
import pandas as pd
import numpy as np
import os
import sys
import hashlib
import logging
from src.config import Config
from src.utils import get_logger, save_json

logger = get_logger("VALIDATOR")

def test_adversarial_alignment():
    """Test feature alignment utility."""
    logger.info("🧪 [TEST 1] Feature Alignment Test...")
    from src.data_loader import align_features
    
    reference_cols = ['feat_A', 'feat_B', 'feat_C']
    
    # Case: Extra column + Missing columns
    df_noisy = pd.DataFrame({
        'feat_A': [1.0, 2.0],
        'extra_X': [10, 20],      
    })
    df_noisy['feat_B'] = ['0.5', '1.5']  
    df_noisy['feat_C'] = [np.nan, 3.0]   
    
    df_aligned = align_features(df_noisy, reference_cols, logger)
    
    assert df_aligned.shape == (2, 3)
    assert list(df_aligned.columns) == reference_cols
    assert not (df_aligned.dtypes == object).any()
    assert not df_aligned.isna().any().any()
    
    logger.info("✅ Feature Alignment Test PASSED.")

def run_pipeline_smoke_test():
    """Run full pipeline in SMOKE_TEST mode."""
    logger.info("🧪 [TEST 2] End-to-End Pipeline Smoke Test (SMOKE_TEST)...")
    
    import subprocess
    
    # We use a unique RUN_ID for validation to avoid polluting real runs
    VALID_RUN_ID = f"validate_{Config.RUN_ID}"
    os.environ['RUN_ID'] = VALID_RUN_ID
    
    # In practice, pipeline.sh is the authoritative runner.
    # We'll simulate it by calling main.py for each phase.
    from main import VALID_PHASES
    
    for phase in VALID_PHASES:
        logger.info(f"  --- Running Phase: {phase} ---")
        cmd = [sys.executable, 'main.py', '--phase', phase, '--mode', 'debug', '--smoke-test', '--run-id', VALID_RUN_ID]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"  ❌ Phase {phase} FAILED!")
            logger.error(f"  STDOUT:\n{result.stdout[-2000:]}")
            logger.error(f"  STDERR:\n{result.stderr[-2000:]}")
            sys.exit(1)
        else:
            logger.info(f"  ✓ Phase {phase} SUCCESS.")
            
    # Post-pipeline artifacts check
    logger.info("  --- Post-Pipeline Artifact Verification ---")
    
    # 1. Verify pruning manifest
    manifest_path = f"outputs/{VALID_RUN_ID}/processed/pruning_manifest.json"
    assert os.path.exists(manifest_path), f"Missing artifact: {manifest_path}"
    
    # 2. Verify submission
    sub_path = f"outputs/{VALID_RUN_ID}/submission.csv"
    assert os.path.exists(sub_path), f"Missing artifact: {sub_path}"
    
    # 3. Verify intelligence report
    intel_path = f"logs/{VALID_RUN_ID}/validation_report.json"
    assert os.path.exists(intel_path), f"Missing artifact: {intel_path}"
    
    logger.info("✅ Pipeline Smoke Test PASSED.")

if __name__ == "__main__":
    try:
        test_adversarial_alignment()
        run_pipeline_smoke_test()
        logger.info("\n🏆 ALL v16.0 PIPELINE TESTS PASSED (SUCCESS)")
    except Exception as e:
        logger.error(f"💥 Validation Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
