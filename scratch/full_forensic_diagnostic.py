import os
import json
import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
from glob import glob

base_run = "run_20260429_103656"
curr_run = "run_20260430_120906"

def load_data(run_id):
    out_dir = f"outputs/{run_id}"
    log_dir = f"logs/{run_id}"
    
    oof = np.load(f"{out_dir}/predictions/oof_stable.npy")
    y = np.load(f"{out_dir}/processed/y_train.npy")
    
    with open(f"{log_dir}/validation_report.json", "r") as f:
        val_report = json.load(f)
        
    try:
        drift_csv = pd.read_csv(f"{log_dir}/summary/distribution/drift_audit_raw.csv")
    except:
        drift_csv = pd.DataFrame()

    with open(f"{out_dir}/processed/stability_manifest.json", "r") as f:
        stability = json.load(f)
        
    return {
        "oof": oof,
        "y": y,
        "val": val_report,
        "drift": drift_csv,
        "stability": stability
    }

base = load_data(base_run)
curr = load_data(curr_run)

print("="*60)
print("[MISSION 1] DISTRIBUTION COLLAPSE vs EXPANSION")
print("="*60)

def print_dist(name, data):
    y = data['y']
    oof = data['oof']
    
    mean_ratio = np.mean(oof) / np.mean(y)
    std_ratio = np.std(oof) / np.std(y)
    p90_ratio = np.percentile(oof, 90) / np.percentile(y, 90)
    p99_ratio = np.percentile(oof, 99) / np.percentile(y, 99)
    max_pred = np.max(oof)
    
    print(f"[{name}] Mean Ratio: {mean_ratio:.4f} | Std Ratio: {std_ratio:.4f}")
    print(f"[{name}] P90 Ratio: {p90_ratio:.4f}  | P99 Ratio: {p99_ratio:.4f}")
    print(f"[{name}] Max Pred: {max_pred:.4f}    | Target Max: {np.max(y):.4f}")

print_dist("BASELINE", base)
print_dist("CURRENT ", curr)


print("\n"+"="*60)
print("[MISSION 3] ERROR SEGMENTATION")
print("="*60)

def segment_errors(name, data):
    y = data['y']
    oof = data['oof']
    
    q50 = np.percentile(y, 50)
    q90 = np.percentile(y, 90)
    q99 = np.percentile(y, 99)
    
    mask_b50 = y <= q50
    mask_50_90 = (y > q50) & (y <= q90)
    mask_90_99 = (y > q90) & (y <= q99)
    mask_top1 = y > q99
    
    def calc_seg(mask):
        if not np.any(mask): return 0, 0
        mae = np.mean(np.abs(y[mask] - oof[mask]))
        bias = np.mean(oof[mask] - y[mask])
        return mae, bias
        
    m_b50, b_b50 = calc_seg(mask_b50)
    m_50_90, b_50_90 = calc_seg(mask_50_90)
    m_90_99, b_90_99 = calc_seg(mask_90_99)
    m_top1, b_top1 = calc_seg(mask_top1)
    
    print(f"[{name}]")
    print(f"  Bottom 50% : MAE = {m_b50:6.4f} | Bias = {b_b50:8.4f}")
    print(f"  Q50-90     : MAE = {m_50_90:6.4f} | Bias = {b_50_90:8.4f}")
    print(f"  Q90-99     : MAE = {m_90_99:6.4f} | Bias = {b_90_99:8.4f}")
    print(f"  Top 1%     : MAE = {m_top1:6.4f} | Bias = {b_top1:8.4f}")

segment_errors("BASELINE", base)
segment_errors("CURRENT ", curr)


print("\n"+"="*60)
print("[MISSION 4] DRIFT 정량화 & [MISSION 7] CONFIG 비교")
print("="*60)

print("--- BASELINE ---")
print(f"ADV AUC: {base['val'].get('adv_auc', 'N/A')}")
print(f"Stable Features: {len(base['stability'].get('stable_features', []))}")
print(f"Unstable Features: {len(base['stability'].get('unstable_features', []))}")

print("--- CURRENT ---")
print(f"ADV AUC: {curr['val'].get('adv_auc', 'N/A')}")
print(f"Stable Features: {len(curr['stability'].get('stable_features', []))}")
print(f"Unstable Features: {len(curr['stability'].get('unstable_features', []))}")

# Top dropped features
base_feats = set(base['stability'].get('stable_features', []))
curr_feats = set(curr['stability'].get('stable_features', []))

print("\nFeatures in BASELINE but not in CURRENT:")
print(list(base_feats - curr_feats)[:10])

print("\nFeatures in CURRENT but not in BASELINE:")
print(list(curr_feats - base_feats)[:10])


print("\n"+"="*60)
print("[MISSION 5] FEATURE CONTRIBUTION / ALIGNMENT CHECK")
print("="*60)
# We will use model binaries to check alignment/contribution changes
def check_alignment(run_id):
    model_paths = glob(f"outputs/{run_id}/models/lgbm/model_fold_*.pkl")
    if not model_paths:
        print(f"[{run_id}] No models found.")
        return
        
    with open(model_paths[0], 'rb') as f:
        model = pickle.load(f)
    
    # Get feature importance
    imp = model.feature_importances_
    features = model.feature_name_
    
    df = pd.DataFrame({"feature": features, "importance": imp}).sort_values('importance', ascending=False)
    print(f"[{run_id}] Top 5 Features:")
    for _, row in df.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']}")

check_alignment(base_run)
print("-")
check_alignment(curr_run)
