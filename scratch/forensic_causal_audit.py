import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import logging
import hashlib
import time
import sys
import shutil
import glob

# Add current directory to path so we can import src
sys.path.append(os.getcwd())

from src.config import Config
from src.utils import load_npy, save_npy
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- [RULE 1 & 4: HASH VERIFICATION & INTEGRITY] ---
def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def verify_and_log_integrity(dir_path):
    files = glob.glob(os.path.join(dir_path, "**/*"), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    
    manifest = {}
    for f in sorted(files):
        rel_path = os.path.relpath(f, dir_path).replace("\\", "/")
        manifest[rel_path] = {
            "hash": get_file_hash(f),
            "size": os.path.getsize(f)
        }
    return manifest

def assert_parity(src_manifest, dst_manifest):
    # Filter manifest to only include Phase 4 essentials
    essential_keys = [
        'processed/X_train_full.npy',
        'processed/y_train.npy',
        'processed/scenario_id.npy',
        'processed/X_test_full.npy'
    ]
    for k in essential_keys:
        if k not in dst_manifest:
            raise RuntimeError(f"ESSENTIAL FILE MISSING: {k}")
        if src_manifest[k]["hash"] != dst_manifest[k]["hash"]:
            raise RuntimeError(f"HASH MISMATCH for {k}! Integrity compromised.")
    logger.info(f"[LOCKDOWN] Hash Parity Verified for essential Phase 4 artifacts.")

# --- [RULE 3: CONTAMINATION GUARD] ---
def guard_isolated_directory(run_id):
    out_base = f'./outputs/{run_id}'
    if os.path.exists(out_base):
        files = glob.glob(os.path.join(out_base, "**/*"), recursive=True)
        files = [f for f in files if os.path.isfile(f)]
        if len(files) > 0:
            logger.warning(f"[LOCKDOWN] Pre-existing files in {out_base}. Cleaning for Fast Mode...")
            shutil.rmtree(out_base)
            os.makedirs(out_base)
    else:
        os.makedirs(out_base, exist_ok=True)
    logger.info(f"[LOCKDOWN] Isolation verified for {run_id}")

# --- [RULE 2: CONFIG SNAPSHOT] ---
def capture_config_snapshot(run_id, factor, is_enabled):
    snapshot = {
        "run_id": run_id,
        "TIMESTAMP": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "MODE": Config.MODE,
        "amplification": {
            "factor": factor,
            "enabled": is_enabled
        },
        "seeds": list(Config.SEEDS),
        "nfolds": Config.NFOLDS,
        "estimators": Config.LGBM_PARAMS['n_estimators'],
        "subsample_Ratio": 0.4,
        "audit_sample_hash": get_file_hash('audit_sample.pkl') if os.path.exists('audit_sample.pkl') else None
    }
    with open(f'./outputs/{run_id}/config_snapshot.json', 'w') as f:
        json.dump(snapshot, f, indent=4)
    logger.info(f"[LOCKDOWN] Config snapshot saved for {run_id}")

# --- [METRIC CORE] ---
def compute_detailed_metrics(y_true, pred):
    mae = mean_absolute_error(y_true, pred)
    mean_val = np.mean(pred)
    std_val = np.std(pred)
    qs = np.quantile(pred, [0.10, 0.50, 0.90])
    
    return {
        "mae": float(mae),
        "mean": float(mean_val),
        "std": float(std_val),
        "q10": float(qs[0]),
        "q50": float(qs[1]),
        "q90": float(qs[2])
    }

# --- [CONFIG INJECTION] ---
# --- [CONFIG INJECTION] ---
def update_config_file(factor, is_enabled, run_id, forced_amp=False):
    config_path = 'src/config.py'
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if 'ENABLE_CONTRASTIVE_AMP =' in line:
            new_lines.append(f"    ENABLE_CONTRASTIVE_AMP = {is_enabled}\n")
        elif 'CONTRASTIVE_AMP_FACTOR =' in line:
            new_lines.append(f"    CONTRASTIVE_AMP_FACTOR = {factor}\n")
        elif 'FORCED_AMP_MODE =' in line:
            new_lines.append(f"    FORCED_AMP_MODE = {forced_amp}\n")
        elif 'RUN_ID = os.getenv' in line:
            new_lines.append(f"    RUN_ID = os.getenv('RUN_ID', '{run_id}')\n")
        else:
            new_lines.append(line)
            
    with open(config_path, 'w') as f:
        f.writelines(new_lines)

# --- [EXPERIMENT RUNNER] ---
def run_experiment(factor, is_enabled, exp_id, forced_amp=False):
    run_id = f"fast_audit_{exp_id}_{factor}"
    logger.info(f"\n--- [FAST EXPERIMENT {exp_id}] Factor={factor} | RunID={run_id} | Forced={forced_amp} ---")
    
    guard_isolated_directory(run_id)
    
    # Parity Guarantee
    src_base_abs = os.path.abspath('./outputs/default_run')
    dst_base_abs = os.path.abspath(f'./outputs/{run_id}')
    
    os.makedirs(f'./outputs/{run_id}/processed', exist_ok=True)
    os.makedirs(f'./outputs/{run_id}/predictions', exist_ok=True)
    
    essential_processed = ["X_train_full.npy", "X_test_full.npy", "y_train.npy", "scenario_id.npy", "oof_raw.npy", "regime_proxy_tr.npy", "regime_proxy_te.npy"]
    for f in essential_processed:
        shutil.copy2(os.path.join(src_base_abs, "processed", f), os.path.join(dst_base_abs, "processed", f))
    
    essential_preds = ["test_raw_preds.npy"]
    for f in essential_preds:
        shutil.copy2(os.path.join(src_base_abs, "predictions", f), os.path.join(dst_base_abs, "predictions", f))

    # Inject Audit Sample
    shutil.copy2('audit_sample.pkl', f'./outputs/{run_id}/audit_sample.pkl')
    
    src_manifest = verify_and_log_integrity(src_base_abs)
    dst_manifest = verify_and_log_integrity(dst_base_abs)
    assert_parity(src_manifest, dst_manifest)
    
    # [FIX] update_config_file with forced_amp preservation
    update_config_file(factor, is_enabled, run_id, forced_amp=forced_amp)
    capture_config_snapshot(run_id, factor, is_enabled)
    
    import subprocess
    env = os.environ.copy()
    env['RUN_ID'] = run_id
    # Fast Mode: Only Phase 5 is needed to generate oof_stable.npy for causal evaluation.
    for phase in ["5_train_final"]:
        cmd = f"python main.py --phase {phase}"
        logger.info(f"[EXECUTE] {cmd}")
        subprocess.run(cmd, env=env, shell=True, check=True)
    
    # Load results - Apply same audit_sample to y_train for metric consistency
    y_full = load_npy(f'./outputs/{run_id}/processed/y_train.npy')
    with open('audit_sample.pkl', 'rb') as f:
        import pickle
        sample_idx = pickle.load(f)
    y_sampled = y_full[sample_idx]
    
    oof_sampled = load_npy(f'./outputs/{run_id}/predictions/oof_stable.npy')
    
    if len(y_sampled) != len(oof_sampled):
        logger.error(f"Shape mismatch! y: {len(y_sampled)}, oof: {len(oof_sampled)}")
        # This might happen if Phase 6 combined data. 
        # But Phase 5 oof_stable should match.
    
    results = compute_detailed_metrics(y_sampled, oof_sampled)
    return results

# --- [MAIN AUDIT] ---
def run_fast_audit():
    # RULE 5: Trigger Logic [FORCED ACTIVATION TEST]
    exp_a = {"factor": 1.0, "enabled": False, "id": "A", "forced": False}
    exp_b = {"factor": 2.0, "enabled": True, "id": "B", "forced": True}
    
    logger.info("[MISSION] SAFE FORCED ACTIVATION TEST STARTING")
    
    res_a = run_experiment(exp_a['factor'], exp_a['enabled'], exp_a['id'], forced_amp=exp_a['forced'])
    res_b = run_experiment(exp_b['factor'], exp_b['enabled'], exp_b['id'], forced_amp=exp_b['forced'])
    
    # Rule 3: Verification Check
    # We load the oof_stable.npy from both runs and ensure they are different
    oof_a = np.load(f"outputs/fast_audit_A_1.0/predictions/oof_stable.npy")
    oof_b = np.load(f"outputs/fast_audit_B_2.0/predictions/oof_stable.npy")
    
    if np.array_equal(oof_a, oof_b):
        logger.error("[RULE_3_FAIL] Predictions are identical! Amplification was NOT applied.")
        sys.exit(1)
    else:
        logger.info("[RULE_3_PASS] Predictions are non-identical. Amplification confirmed active.")

    mae_a = res_a['mae']
    mae_b = res_b['mae']
    delta = abs(mae_b - mae_a)
    
    # Rule 4: Magnitude Check
    magnitude_ratio = res_b['mean'] / (res_a['mean'] + 1e-8)
    
    logger.info(f"\n[RESULTS] MAE_A: {mae_a:.4f} | MAE_B: {mae_b:.4f} | Delta: {delta:.4f}")
    logger.info(f"[MAGNITUDE] Ratio (B/A): {magnitude_ratio:.4f}")
    
    # Final Verdict
    mean_shift = (res_b['mean'] - res_a['mean']) / res_a['mean']
    verdict = "CONFIRMED" if mean_shift > 0.05 or magnitude_ratio > 1.05 else "REJECTED"
    
    report = {
        "mode": "FORCED_ACTIVATION_DIAGNOSTIC",
        "A": res_a,
        "B": res_b,
        "analysis": {
            "mae_delta": mae_b - mae_a,
            "mean_shift": mean_shift,
            "magnitude_ratio": magnitude_ratio,
            "verdict": verdict,
            "diagnostic_context": "Worst-case isolation. Not representative of standard production trigger logic."
        }
    }
    
    with open('outputs/fast_causal_audit.json', 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"\n[AMPLIFICATION_FORCED_RISK_REPORT]")
    print(f"- verdict: {verdict}")
    print(f"- magnitude_ratio: {magnitude_ratio:.4f}")
    print(f"- mae_shift: {mae_b - mae_a:.4f}")
    print(f"- mean_shift: {mean_shift:.2%}")
    print(f"- status: RULE_3_PASS")

if __name__ == "__main__":
    run_fast_audit()
