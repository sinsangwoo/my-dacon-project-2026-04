import pandas as pd
import numpy as np
import json
import os

def find_divergent_features(run_id):
    # 1. Load forensic data
    recon_dir = f"outputs/{run_id}/models/reconstructors"
    p, y = [], []
    for fold in range(5):
        path = f"{recon_dir}/forensic_fold_{fold}.json"
        if not os.path.exists(path): continue
        with open(path, "r") as f: d = json.load(f)
        p.extend(d['p_val'])
        y.extend(d['y_val'])
    
    p = np.array(p)
    y = np.array(y)
    
    # 2. Load train data
    proc_dir = f"outputs/{run_id}/processed"
    train_base = pd.read_pickle(f"{proc_dir}/train_base.pkl")
    y_train = np.load(f"{proc_dir}/y_train.npy")
    
    # Since we don't have the exact index in forensic_json, we approximate by finding high-p samples in train_base 
    # (Actually, let's just use the whole train_base and classify them)
    # But wait, we have 5000 samples in forensic_fold_0. These are OOF samples.
    # Let's just use a simple approach: find features correlated with (p > 0.7 & y < 44) vs (p > 0.7 & y >= 44)
    
    # [MISSION] Identify features that are HIGH in FPs but LOW in TPs
    # Or features that are LOW in FPs but HIGH in TPs (Discriminators)
    
    # To do this accurately, I need the OOF indices or just a large sample.
    # Let's just use the whole dataset and simulate the classifier if needed, 
    # OR better, look at the features that have the highest difference in distribution between Tail and Non-Tail.
    
    q90 = np.percentile(y_train, 90)
    tail_mask = y_train >= q90
    
    df_tail = train_base[tail_mask]
    df_bulk = train_base[~tail_mask]
    
    # Find features where Tail is LOWER than Bulk (Unexpected)
    # These might be "Tail-Inhibitors"
    
    diffs = []
    cols = train_base.select_dtypes(include=[np.number]).columns
    for col in cols:
        m_tail = df_tail[col].mean()
        m_bulk = df_bulk[col].mean()
        std_bulk = df_bulk[col].std() + 1e-9
        z_score = (m_tail - m_bulk) / std_bulk
        diffs.append((col, z_score))
        
    sorted_diffs = sorted(diffs, key=lambda x: x[1], reverse=True)
    
    print("\n[TOP DISCRIMINATORS] (High in Tail, Low in Bulk)")
    for col, z in sorted_diffs[:10]:
        print(f"{col:<40} | Z-Score: {z:.4f}")
        
    print("\n[POTENTIAL INHIBITORS] (Low in Tail, High in Bulk)")
    for col, z in sorted_diffs[-10:]:
        print(f"{col:<40} | Z-Score: {z:.4f}")

if __name__ == "__main__":
    find_divergent_features("run_20260503_172033")
