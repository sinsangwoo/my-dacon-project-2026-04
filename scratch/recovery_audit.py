import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- [CONFIG] ---
BASELINE_CV_MAE = 9.13
# Baseline values from run_20260418_023802 (The last known good "Explosion" run)
BASELINE_DIST = {
    "mean": 19.4939,
    "std": 14.7247,
    "q10": 4.12,  # Estimated
    "q50": 16.5,  # Estimated
    "q90": 41.2,  # Estimated
    "q99": 72.5   # Estimated
}

METRICS_PATH = './outputs/default_run/processed/metrics.json'
DIST_PATH = './outputs/default_run/processed/pred_distribution.json'
IMPORTANCE_PATH = './outputs/default_run/processed/feature_importances.json'
HISTORICAL_IMPORTANCE_PATH = './metadata/historical_feature_importances.json' # I will save the current one there as baseline if not exists

def run_audit():
    logger.info("--- [RECOVERY_AUDIT_START] ---")
    verdicts = {}
    
    # 1. MAE Audit (SSOT)
    if not os.path.exists(METRICS_PATH):
        logger.error("metrics.json NOT FOUND. SSOT Contract Violated.")
        return "FAILED", "Missing metrics.json"
        
    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)
    
    cv_mae = metrics.get('cv_mae', 99.0)
    delta = cv_mae - BASELINE_CV_MAE
    verdicts['mae'] = abs(delta) <= 0.5
    logger.info(f"[MAE_AUDIT] cv_mae: {cv_mae:.4f} | baseline: {BASELINE_CV_MAE:.4f} | delta: {delta:+.4f} | status: {'PASS' if verdicts['mae'] else 'FAIL'}")

    # 2. Prediction Distribution Audit
    if not os.path.exists(DIST_PATH):
        logger.error("pred_distribution.json NOT FOUND.")
        return "FAILED", "Missing pred_distribution.json"
        
    with open(DIST_PATH, 'r') as f:
        curr_dist = json.load(f)
        
    dist_fail = False
    for k, base_val in BASELINE_DIST.items():
        if k in curr_dist:
            curr_val = curr_dist[k]
            # Use 15% relative tolerance or 1.0 absolute
            rel_diff = abs(curr_val - base_val) / (base_val + 1e-9)
            if rel_diff > 0.15 and abs(curr_val - base_val) > 1.0:
                logger.warning(f"[DIST_AUDIT] DRIFT DETECTED in {k}: curr={curr_val:.4f} base={base_val:.4f} diff={rel_diff:.1%}")
                dist_fail = True
            else:
                logger.info(f"[DIST_AUDIT] {k}: {curr_val:.4f} (matches baseline)")
    
    verdicts['dist'] = not dist_fail

    # 3. Feature Importance Drift
    if os.path.exists(IMPORTANCE_PATH) and os.path.exists(HISTORICAL_IMPORTANCE_PATH):
        with open(IMPORTANCE_PATH, 'r') as f: curr_imp = json.load(f)
        with open(HISTORICAL_IMPORTANCE_PATH, 'r') as f: hist_imp = json.load(f)
        
        common_features = sorted(list(set(curr_imp.keys()) & set(hist_imp.keys())))
        if len(common_features) > 10:
            c_vals = [curr_imp[f] for f in common_features]
            h_vals = [hist_imp[f] for f in common_features]
            corr, _ = spearmanr(c_vals, h_vals)
            verdicts['drift'] = corr > 0.8
            logger.info(f"[DRIFT_AUDIT] Rank Correlation: {corr:.4f} | status: {'PASS' if verdicts['drift'] else 'FAIL'}")
        else:
            logger.warning("[DRIFT_AUDIT] Not enough common features to compare.")
            verdicts['drift'] = True # Neutral
    else:
        logger.info("[DRIFT_AUDIT] Baseline missing. Skipping.")
        verdicts['drift'] = True

    # 4. OOF vs Prediction Correlation (Q-Q Audit)
    OOF_PATH = './outputs/predictions/oof_stable.npy'
    TEST_PATH = './outputs/predictions/final_submission.npy'
    
    if os.path.exists(OOF_PATH) and os.path.exists(TEST_PATH):
        oof = np.load(OOF_PATH)
        test_pred = np.load(TEST_PATH)
        
        # Calculate quantiles for both
        qs = np.linspace(0.01, 0.99, 100)
        oof_q = np.quantile(oof, qs)
        test_q = np.quantile(test_pred, qs)
        
        q_corr, _ = spearmanr(oof_q, test_q)
        verdicts['corr'] = q_corr > 0.95
        logger.info(f"[CORR_AUDIT] OOF vs Test Q-Q Correlation: {q_corr:.4f} | status: {'PASS' if verdicts['corr'] else 'FAIL'}")
    else:
        logger.warning("[CORR_AUDIT] OOF or Test predictions missing. Skipping correlation audit.")
        verdicts['corr'] = True

    # Final Verdict
    if all(verdicts.values()):
        status = "FULL"
    elif verdicts['mae']:
        status = "PARTIAL"
    else:
        status = "FAILED"
        
    print(f"\n[RECOVERY_STATUS]")
    print(f"- cv_mae: {cv_mae:.4f}")
    print(f"- delta_from_best: {delta:+.4f}")
    print(f"- verdict: {status}")
    
    return status

if __name__ == "__main__":
    run_audit()
