import numpy as np
import pandas as pd
import json
import os
import sys
import logging

# Add project root to path
sys.path.append(os.getcwd())

from src.config import Config
from src.utils import load_npy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TailAudit")

RUN_ID = "run_20260421_003036"
BASE_PATH = f"c:/Github_public/my_dacon_project/my-dacon-project-2026-04/outputs/{RUN_ID}"
PROCESSED_PATH = f"{BASE_PATH}/processed"
PREDICTIONS_PATH = f"{BASE_PATH}/predictions"

def get_stats(arr):
    return {
        "mean": np.mean(arr),
        "std": np.std(arr),
        "p50": np.quantile(arr, 0.5),
        "p90": np.quantile(arr, 0.9),
        "p99": np.quantile(arr, 0.99)
    }

def audit_tail_impact():
    logger.info("Starting Tail Calibration Causal Impact Audit...")
    
    # 1. Load Data
    train_stats = json.load(open(f"{PROCESSED_PATH}/train_stats.json"))
    
    # 2. Replicate the final steps of ExplosionInference.train_and_infer
    # We need: regime_test, test_stable, test_cat, test_raw, etc.
    test_stable = load_npy(f"{PREDICTIONS_PATH}/test_stable.npy")
    test_cat = load_npy(f"{PREDICTIONS_PATH}/test_cat.npy")
    test_raw = load_npy(f"{PREDICTIONS_PATH}/test_raw_preds.npy")
    regime_te = load_npy(f"{PROCESSED_PATH}/regime_proxy_te.npy")
    
    # Simplified Routing logic from ExplosionInference
    # (In the real code, it uses Ridge/LGBM blend, but for causal isolation 
    # of the TAIL logic, we can use the base predictions before alpha correction)
    
    # Let's assume the "base" is the soft routing result.
    # Since we can't easily re-train the Ridge/LGBM exactly without the full meta-features,
    # we'll use a proxy for the routed base:
    # final_preds = (regime_test * test_extreme_blend) + ((1.0 - regime_test) * test_normal_blend)
    
    # Actually, a better way is to just use the uncertainty_test and apply it to a baseline.
    # The uncertainty_test is (q90 - q50).
    # In our restored pipeline: q50 ~ stable, q90 ~ cat
    uncertainty_test = test_cat - test_stable
    
    # Baseline (A): Routing only, NO alpha correction
    # We'll use the final_submission.npy as a starting point and REVERSE the alpha correction
    # OR better: build it from scratch.
    
    # Let's try to get the exact state before alpha correction.
    # From the logs of my previous run:
    # [DISTRIBUTION_FLOW_TRACE] after routing | mean: 26.5424 | std: 27.0041 | p50: 17.7702 | p90: 61.9054 | p99: 95.5683
    # [DISTRIBUTION_FLOW_TRACE] after alpha correction | mean: 24.5861 | std: 27.5721 | p50: 16.7270 | p90: 58.3491 | p99: 91.5322
    
    # Wait, why did mean decrease after alpha correction? 
    # Uncertainty (cat - stable) = 20.27 - 23.95 = -3.68 (negative!)
    # So alpha correction REDUCED the mean because cat (pseudo) was lower than stable.
    
    # Let's define:
    # A: routing result (No alpha correction)
    # B: routing result + alpha * uncertainty (Tail Calibration ON)
    
    # We'll simulate the routing result by blending stable and cat based on regime.
    # (This is a simplified version of the real routing but good for isolation)
    base_routed = (regime_te * test_cat) + ((1.0 - regime_te) * test_stable)
    
    # Mask for extreme samples (regime > 0.5)
    extreme_mask = regime_te > 0.5
    alpha = 1.0 # As forced in the current code
    
    # Experiment A: Tail Calibration OFF
    preds_A = base_routed.copy()
    
    # Experiment B: Tail Calibration ON
    preds_B = base_routed.copy()
    preds_B[extreme_mask] += alpha * uncertainty_test[extreme_mask]
    
    # Apply distribution guard to both for fair comparison (optional, but requested for "pipelines")
    # Actually, let's look at them RAW first to see the "causal" distortion.
    
    stats_A = get_stats(preds_A)
    stats_B = get_stats(preds_B)
    
    metrics = ["mean", "std", "p50", "p90", "p99"]
    
    print("\n--- [TAIL CALIBRATION CAUSAL AUDIT] ---")
    print(f"{'Metric':<10} | {'A (OFF)':<12} | {'B (ON)':<12} | {'Delta':<12}")
    print("-" * 55)
    for m in metrics:
        val_A = stats_A[m]
        val_B = stats_B[m]
        delta = val_B - val_A
        print(f"{m:<10} | {val_A:<12.4f} | {val_B:<12.4f} | {delta:<12.4f}")
        
    # Decision Logic
    if stats_B['mean'] > stats_A['mean'] + 0.1:
        print("\nCONCLUSION: Tail causes INFLATION.")
    elif stats_B['mean'] < stats_A['mean'] - 0.1:
        print("\nCONCLUSION: Tail causes DEFLATION (Correction).")
    else:
        print("\nCONCLUSION: Tail impact on mean is neutral.")
        
    if stats_B['p99'] > stats_A['p99'] + 0.5:
        print("CONCLUSION: Tail improves extreme modeling (P99 Boost).")

if __name__ == "__main__":
    audit_tail_impact()
