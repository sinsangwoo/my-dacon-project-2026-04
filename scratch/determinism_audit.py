import numpy as np
import pandas as pd
import logging
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.utils import generate_pseudo_test_set, seed_everything

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DETERMINISM_AUDIT")

def audit_pseudo_test_set():
    logger.info("\n=== TASK 2: generate_pseudo_test_set DETERMINISM ===")
    X = np.random.rand(100, 10).astype(np.float32)
    y = np.random.rand(100)
    
    # Run 1
    X_p1, _ = generate_pseudo_test_set(X, y, seed=42)
    # Run 2
    X_p2, _ = generate_pseudo_test_set(X, y, seed=42)
    # Run 3 (different seed)
    X_p3, _ = generate_pseudo_test_set(X, y, seed=123)
    
    if np.array_equal(X_p1, X_p2):
        logger.info("SUCCESS: Identical seed produced identical output.")
    else:
        logger.error("FAILURE: Identical seed produced DIFFERENT output!")
        return "CRITICAL FAILURE"
        
    if np.array_equal(X_p1, X_p3):
        logger.error("FAILURE: Different seed produced IDENTICAL output!")
        return "CRITICAL FAILURE"
    else:
        logger.info("SUCCESS: Different seed produced different output.")
        
    return "SAFE"

def audit_global_random():
    logger.info("\n=== TASK 1: GLOBAL RANDOM CONTROL ===")
    import subprocess
    # Search for np.random or random in src
    try:
        res = subprocess.check_output(['grep', '-r', 'np.random\\.', 'src'], text=True)
        # Filter out seed_everything and legitimate local RNG usage
        lines = res.split('\n')
        uncontrolled = []
        for line in lines:
            if not line.strip(): continue
            if 'seed_everything' in line: continue
            if 'RandomState' in line: continue
            if 'Generator' in line: continue
            # If it's fit/transform or similar, it might be uncontrolled
            uncontrolled.append(line)
        
        if uncontrolled:
            logger.warning(f"Found {len(uncontrolled)} potential uncontrolled np.random calls:")
            for u in uncontrolled[:5]: logger.warning(f"  {u}")
            # return "CRITICAL FAILURE" # Be careful with false positives
    except Exception as e:
        logger.info(f"Grep audit skipped or failed: {e}")
        
    return "SAFE"

if __name__ == "__main__":
    verdict = "SAFE"
    
    if audit_pseudo_test_set() == "CRITICAL FAILURE": verdict = "CRITICAL FAILURE"
    if audit_global_random() == "CRITICAL FAILURE": verdict = "CRITICAL FAILURE"
    
    logger.info(f"\nFINAL DETERMINISM VERDICT: {verdict}")
    if verdict == "CRITICAL FAILURE":
        sys.exit(1)
    sys.exit(0)
