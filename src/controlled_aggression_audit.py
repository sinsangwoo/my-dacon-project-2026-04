import os
import json
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, mean_absolute_error

def _sigmoid_gate(p, k, theta):
    return 1.0 / (1.0 + np.exp(-k * (p - theta)))

def run_controlled_aggression_audit(run_id, log_dir):
    print(f"\n🔬 Starting Forensic Audit for [RUN_ID: {run_id}]")
    
    fold_dir = os.path.join("outputs", run_id, "models", "reconstructors")
    
    # --- TASK 1: Classifier Calibration ---
    all_p = []
    all_y_binary = []
    
    # --- TASK 2: Sigmoid vs Power ---
    all_weights_sigmoid = []
    all_weights_power = []
    
    # --- TASK 4: Risk Controller ---
    all_gaps = []
    damping_applied_counts = 0
    total_samples = 0
    
    for fold in range(5):
        path = os.path.join(fold_dir, f"forensic_fold_{fold}.json")
        if not os.path.exists(path):
            continue
            
        with open(path, "r") as f:
            data = json.load(f)
            
        p = np.array(data["p_val"])
        y_val = np.array(data["y_val"])
        q90_val = np.percentile(y_val, 90)
        y_binary = (y_val >= q90_val).astype(int)
        
        all_p.extend(p)
        all_y_binary.extend(y_binary)
        
        # Blending comparison
        w_sigmoid = _sigmoid_gate(p, 8.0, 0.55)
        w_power = p ** 2.0
        all_weights_sigmoid.extend(w_sigmoid)
        all_weights_power.extend(w_power)
        
        # Risk Controller
        gaps = np.array(data["gap"])
        weights = np.array(data["final_weight"])
        
        # Damping check: if weight < sigmoid_weight (approximate)
        damping_applied_counts += np.sum(weights < w_sigmoid * 0.99)
        total_samples += len(p)
        all_gaps.extend(gaps)

    if not all_p:
        print("❌ No forensic data found. Ensure Phase 5 ran with instrumented Trainer.")
        return

    all_p = np.array(all_p)
    all_y_binary = np.array(all_y_binary)
    all_w_sig = np.array(all_weights_sigmoid)
    all_w_pow = np.array(all_weights_power)
    all_gaps = np.array(all_gaps)

    # TASK 1: Metrics
    prob_true, prob_pred = calibration_curve(all_y_binary, all_p, n_bins=10)
    brier = brier_score_loss(all_y_binary, all_p)
    auc = roc_auc_score(all_y_binary, all_p)
    
    high_p_mask = all_p > 0.7
    tail_precision = np.mean(all_y_binary[high_p_mask]) if np.any(high_p_mask) else 0.0
    
    print("\n--- [TASK 1: CLASSIFIER CALIBRATION] ---")
    print(f"Brier Score: {brier:.4f}")
    print(f"AUC (Tail Detection): {auc:.4f}")
    print(f"Precision @ p > 0.7: {tail_precision:.4f}")
    
    # TASK 2: Sigmoid vs Power
    print("\n--- [TASK 2: SIGMOID GATE BEHAVIOR] ---")
    print(f"Mean Weight (Sigmoid): {np.mean(all_w_sig):.4f}")
    print(f"Mean Weight (Power):   {np.mean(all_w_pow):.4f}")
    print(f"P90 Weight (Sigmoid):  {np.percentile(all_w_sig, 90):.4f}")
    print(f"Aggression Factor:     {np.mean(all_w_sig) / (np.mean(all_w_pow) + 1e-9):.2f}x")

    # TASK 3: Variance Recovery
    pred_dir = os.path.join("outputs", run_id, "predictions")
    proc_dir = os.path.join("outputs", run_id, "processed")
    
    stable_v, y_v = None, None
    try:
        y_true = np.load(os.path.join(proc_dir, "y_train.npy"))
        oof_raw = np.load(os.path.join(pred_dir, "oof_raw.npy"))
        oof_stable = np.load(os.path.join(pred_dir, "oof_stable.npy"))
        
        valid_mask = ~np.isnan(oof_raw)
        y_v = y_true[valid_mask]
        raw_v = oof_raw[valid_mask]
        stable_v = oof_stable[valid_mask]
        
        q90 = np.percentile(y_v, 90)
        tail_mask = y_v >= q90
        
        print("\n--- [TASK 3: VARIANCE RECOVERY] ---")
        print(f"Std Ratio (Raw):    {np.std(raw_v)/np.std(y_v):.4f}")
        print(f"Std Ratio (Stable): {np.std(stable_v)/np.std(y_v):.4f}")
        print(f"P99 Ratio (Stable): {np.percentile(stable_v, 99)/np.percentile(y_v, 99):.4f}")
        print(f"MAE Bulk:           {mean_absolute_error(y_v[~tail_mask], stable_v[~tail_mask]):.4f}")
        print(f"MAE Tail:           {mean_absolute_error(y_v[tail_mask], stable_v[tail_mask]):.4f}")
    except Exception as e:
        print(f"⚠️ Task 3 failed: {str(e)}")

    # TASK 4: Risk Controller
    print("\n--- [TASK 4: RISK CONTROLLER effectiveness] ---")
    print(f"Activation Rate: {damping_applied_counts/total_samples*100:.2f}%")
    print(f"Max Gap Observed: {np.max(all_gaps):.4f}")
    print(f"P95 Gap:          {np.percentile(all_gaps, 95):.4f}")

    # Verdicts
    print("\n" + "="*50)
    print("      FORENSIC VERDICT (Controlled Aggression)")
    print("="*50)
    
    fm1_verdict = "SAFE" if brier < 0.1 and tail_precision > 0.5 else "RISK"
    fm2_verdict = "SAFE" if (stable_v is not None and np.std(stable_v)/np.std(y_v) < 1.1) else "CRITICAL"
    fm3_verdict = "SAFE" if damping_applied_counts > 0 else "CRITICAL"
    
    print(f"FM-1 (Sigmoid Gate):   {fm1_verdict}")
    print(f"FM-2 (Var Recovery):   {fm2_verdict}")
    print(f"FM-3 (Risk Controller): {fm3_verdict}")
    print("="*50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    args = parser.parse_args()
    run_controlled_aggression_audit(args.run_id, f"logs/{args.run_id}")
