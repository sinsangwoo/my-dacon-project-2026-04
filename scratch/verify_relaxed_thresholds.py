import json
import os
import pandas as pd
import numpy as np

def verify_new_rules(run_id):
    print(f"=== [RE-EVALUATING (TIER 2): {run_id}] ===")
    
    log_path = f"outputs/{run_id}/processed/signal_validation_logs.json"
    with open(log_path, 'r') as f:
        data = json.load(f)
        
    val_logs = pd.DataFrame(data['val_logs'])
    
    gains = val_logs['gain'].values
    min_gain_threshold = np.mean(gains) * 0.05
    print(f"New min_gain_threshold: {min_gain_threshold:.4f}")
    
    def get_rejections(r):
        gain = r['gain']
        ks = r['ks_stat']
        marg_corr = r['marg_corr']
        perm_delta = r['perm_delta']
        splits = r['splits']
        avg_depth = r['avg_depth']
        
        rejections = []
        if not (gain > min_gain_threshold): rejections.append("Gain")
        
        is_danger_zone = 0.50 < ks <= 0.85
        ratio_threshold = 0.0001 if not is_danger_zone else 0.0002
        gen_ratio = perm_delta / (gain + 1e-9)
        if not (perm_delta > 0 and gen_ratio > ratio_threshold): rejections.append("GenRatio")
        
        if not (abs(marg_corr) >= 0.001): rejections.append("MargCorr")
        if not (splits >= 1): # Assume 100 estimators, 1% is 1 split
            rejections.append("Splits")
        if not (avg_depth < 12): rejections.append("Depth")
        if not (ks <= 0.85): rejections.append("KS")
        
        return rejections

    val_logs['new_rejections'] = val_logs.apply(get_rejections, axis=1)
    val_logs['passed_new'] = val_logs['new_rejections'].apply(len) == 0
    
    n_total = len(val_logs)
    n_old = val_logs['passed'].sum()
    n_new = val_logs['passed_new'].sum()
    
    print(f"Old Pass Count: {n_old}")
    print(f"New Pass Count: {n_new}")
    print(f"Survival Rate: {n_new/n_total:.1%}")
    
    useful_rejected = val_logs[(val_logs['perm_delta'] > 0.001) & ~val_logs['passed_new']]
    print(f"\nUseful Features (Perm Delta > 0.001) still rejected: {len(useful_rejected)}")
    if not useful_rejected.empty:
        print("\nRejection Reasons for Useful Features:")
        print(useful_rejected['new_rejections'].apply(lambda x: ', '.join(x)).value_counts().head(5))

if __name__ == "__main__":
    verify_new_rules('run_20260426_235007')
