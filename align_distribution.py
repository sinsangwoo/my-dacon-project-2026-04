import numpy as np
import pandas as pd
import os
import json
from src.config import Config

def align_distribution():
    print("=== [MISSION: DATA-DRIVEN DISTRIBUTION ALIGNMENT] ===")
    
    # 🔴 STEP 1 — LOAD TRAIN STATS
    print("\nStep 1: Computing/Loading training statistics...")
    # Try to load from artifact first, but compute p90 if missing
    stats_path = os.path.join(Config.PROCESSED_PATH, 'train_stats.json')
    
    train_mean = None
    train_std = None
    train_p90 = None
    train_p99 = None
    
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
            train_mean = stats.get('mean')
            train_std = stats.get('std')
            train_p99 = stats.get('p99')
            train_p90 = stats.get('p90')

    if train_mean is None or train_std is None or train_p90 is None or train_p99 is None:
        print("  Missing statistics in artifact. Computing from data/train.csv...")
        train = pd.read_csv(os.path.join(Config.DATA_PATH, 'train.csv'))
        target = train[Config.TARGET]
        train_mean = float(target.mean())
        train_std = float(target.std())
        train_p90 = float(target.quantile(0.90))
        train_p99 = float(target.quantile(0.99))
        
        # Update artifact for future use
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump({
                "mean": train_mean,
                "std": train_std,
                "p90": train_p90,
                "p99": train_p99
            }, f, indent=2)
        print(f"  Artifact updated: {stats_path}")

    print(f"  [TRAIN] Mean: {train_mean:.4f} | Std: {train_std:.4f} | P90: {train_p90:.4f} | P99: {train_p99:.4f}")

    # 🔴 STEP 2 — COMPUTE TEST STATS
    print("\nStep 2: Loading test predictions...")
    # Check common locations for the full prediction file
    possible_paths = [
        'outputs/predictions/final_submission.npy',
        os.path.join(Config.PREDICTIONS_PATH, 'final_submission.npy'),
        os.path.join(Config.PREDICTIONS_PATH, 'test_stable.npy')
    ]
    
    pred_path = None
    for p in possible_paths:
        if os.path.exists(p):
            # Check if it has 50k rows
            try:
                temp_p = np.load(p)
                if len(temp_p) == 50000:
                    pred_path = p
                    break
            except:
                continue
    
    if pred_path is None:
        # Fallback to the first existing one
        for p in possible_paths:
            if os.path.exists(p):
                pred_path = p
                break
                
    if not pred_path:
        print(f"Error: No prediction file found.")
        return

    final_preds = np.load(pred_path).astype(np.float64)
    pred_mean = np.mean(final_preds)
    pred_std = np.std(final_preds)
    
    print(f"  [PRED]  Mean: {pred_mean:.4f} | Std: {pred_std:.4f} (Source: {os.path.basename(pred_path)})")

    # 🔴 STEP 3 — GLOBAL DISTRIBUTION MATCHING
    print("\nStep 3: Applying affine transformation...")
    # scale = train_std / pred_std
    # shift = train_mean - scale * pred_mean
    
    scale = train_std / (pred_std + 1e-9)
    shift = train_mean - scale * pred_mean
    
    aligned_preds = final_preds * scale + shift
    
    print(f"  Transformation: final_preds * {scale:.4f} + {shift:.4f}")

    # 🔴 STEP 4 — TAIL PRESERVATION
    print("\nStep 4: Tail preservation (Affine only)... Done.")

    # 🔴 STEP 5 — EXTREME CONSISTENCY CHECK
    print("\nStep 5: Extreme Consistency Check...")
    # 1. Monotonicity/Ranking
    original_rank = np.argsort(final_preds)
    aligned_rank = np.argsort(aligned_preds)
    if np.array_equal(original_rank, aligned_rank):
        print("  OK: Monotonicity preserved.")
    else:
        print("  Warning: Monotonicity violated! (Unexpected for affine transform)")

    # 2. Top 5% preservation
    n_top5 = int(len(final_preds) * 0.05)
    if n_top5 > 0:
        top5_orig = set(np.argsort(final_preds)[-n_top5:])
        top5_aligned = set(np.argsort(aligned_preds)[-n_top5:])
        overlap = len(top5_orig.intersection(top5_aligned)) / n_top5
        print(f"  OK: Top 5% overlap: {overlap*100:.1f}%")
    else:
        print("  - Dataset too small for top 5% check.")

    # 🔴 STEP 6 — FINAL CLIP
    print("\nStep 6: Final Clip...")
    clip_limit = train_p99 * 1.2
    final_clipped = np.clip(aligned_preds, 0, clip_limit)
    print(f"  Clipped at [0, {clip_limit:.4f}]")

    # 🔴 STEP 7 — VALIDATION OUTPUT
    print("\nStep 7: Final Validation Statistics:")
    new_mean = np.mean(final_clipped)
    new_std = np.std(final_clipped)
    new_p90 = np.quantile(final_clipped, 0.90)
    new_p99 = np.quantile(final_clipped, 0.99)
    
    print(f"  [FINAL] Mean: {new_mean:.4f} (Target: {train_mean:.4f})")
    print(f"  [FINAL] Std:  {new_std:.4f} (Target: {train_std:.4f})")
    print(f"  [FINAL] P90:  {new_p90:.4f} (Target: {train_p90:.4f})")
    print(f"  [FINAL] P99:  {new_p99:.4f} (Target: {train_p99:.4f})")

    # Save to submission file
    print("\nSaving aligned submission...")
    sample_sub_path = os.path.join(Config.DATA_PATH, 'sample_submission.csv')
    sample_sub = pd.read_csv(sample_sub_path)
    
    if len(sample_sub) != len(final_clipped):
        print(f"Warning: Prediction length ({len(final_clipped)}) does not match sample_submission ({len(sample_sub)}).")
        # If it's a debug run, we might only have a subset
        sample_sub = sample_sub.iloc[:len(final_clipped)].copy()

    sample_sub[Config.TARGET] = np.round(final_clipped, 3)
    
    os.makedirs('outputs/aligned', exist_ok=True)
    save_path = 'outputs/aligned/submission_aligned.csv'
    sample_sub.to_csv(save_path, index=False)
    print(f"OK: Saved to {save_path}")

    # Log results
    log_path = 'outputs/aligned/alignment_log.txt'
    with open(log_path, 'w') as f:
        f.write("ALIGNMENT LOG\n")
        f.write("=============\n\n")
        f.write(f"Train Stats:\n")
        f.write(f"  Mean: {train_mean:.4f}\n")
        f.write(f"  Std:  {train_std:.4f}\n")
        f.write(f"  P90:  {train_p90:.4f}\n")
        f.write(f"  P99:  {train_p99:.4f}\n\n")
        f.write(f"Original Pred Stats:\n")
        f.write(f"  Mean: {pred_mean:.4f}\n")
        f.write(f"  Std:  {pred_std:.4f}\n\n")
        f.write(f"Transformation:\n")
        f.write(f"  Scale: {scale:.4f}\n")
        f.write(f"  Shift: {shift:.4f}\n\n")
        f.write(f"Final Aligned Stats (after clip):\n")
        f.write(f"  Mean: {new_mean:.4f}\n")
        f.write(f"  Std:  {new_std:.4f}\n")
        f.write(f"  P90:  {new_p90:.4f}\n")
        f.write(f"  P99:  {new_p99:.4f}\n")
    print(f"OK: Log saved to {log_path}")

if __name__ == "__main__":
    align_distribution()
