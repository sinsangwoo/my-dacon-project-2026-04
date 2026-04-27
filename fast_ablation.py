"""v16.0 Fast Post-Run Analyzer.

Quickly compares OOF vs Test distributions for a given RUN_ID.
"""
import numpy as np
import os
import sys
import pandas as pd

def analyze_run(run_id):
    print(f"\n--- [ANALYZER] Analyzing Run: {run_id} ---")
    
    pred_dir = f"outputs/{run_id}/predictions"
    proc_dir = f"outputs/{run_id}/processed"
    
    if not os.path.exists(pred_dir):
        print(f"Error: Run {run_id} artifacts not found.")
        return
        
    try:
        oof = np.load(f"{pred_dir}/oof_stable.npy")
        test = np.load(f"{pred_dir}/test_stable.npy")
        y_train = np.load(f"{proc_dir}/y_train.npy")
        
        print(f"OOF shape: {oof.shape}")
        print(f"Test shape: {test.shape}")
        
        metrics = {
            "OOF MAE": np.mean(np.abs(oof - y_train)),
            "OOF Mean": np.mean(oof),
            "Test Mean": np.mean(test),
            "OOF Std": np.std(oof),
            "Test Std": np.std(test),
            "Std Ratio": np.std(test) / (np.std(oof) + 1e-9)
        }
        
        for k, v in metrics.items():
            print(f"{k:<15}: {v:.4f}")
            
        if metrics["Std Ratio"] < 0.6:
            print("⚠️ WARNING: Severe Variance Compression detected in Test.")
        elif metrics["Std Ratio"] > 1.4:
            print("⚠️ WARNING: Severe Variance Inflation detected in Test.")
        else:
            print("✅ Variance distribution is reasonably stable.")
            
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, help="RUN_ID to analyze")
    args = parser.parse_args()
    
    if args.run_id:
        analyze_run(args.run_id)
    else:
        print("Usage: python fast_ablation.py --run-id <RUN_ID>")
