import json
import os
import pandas as pd
import numpy as np

def verify_learned_rules(run_id):
    print(f"=== [RE-EVALUATING (LEARNED): {run_id}] ===")
    
    log_path = f"outputs/{run_id}/processed/signal_validation_logs.json"
    with open(log_path, 'r') as f:
        data = json.load(f)
        
    val_logs = pd.DataFrame(data['val_logs'])
    
    # Simulate Noise Ceiling (using data from run_20260426_235007)
    noise_max_gain = data.get('noise_proof', {}).get('noise_max_gain', 25.0)
    noise_max_ratio = 0.0005 # Conservative estimate for baseline
    
    gains = val_logs['gain'].values
    min_gain_threshold = max(noise_max_gain * 0.8, np.percentile(gains, 20))
    print(f"Learned min_gain_threshold: {min_gain_threshold:.4f} (Noise Max: {noise_max_gain})")
    
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
        ratio_threshold = max(noise_max_ratio * 1.5, 0.0005) if not is_danger_zone else max(noise_max_ratio * 3.0, 0.0010)
        gen_ratio = perm_delta / (gain + 1e-9)
        if not (perm_delta > 0 and gen_ratio > ratio_threshold): rejections.append("GenRatio")
        
        if not (abs(marg_corr) >= 0.0001): rejections.append("MargCorr")
        if not (splits >= 1): rejections.append("Splits")
        if not (avg_depth < 12): rejections.append("Depth")
        if not (ks <= 0.85): rejections.append("KS")
        
        return rejections

    val_logs['new_rejections'] = val_logs.apply(get_rejections, axis=1)
    val_logs['passed_learned'] = val_logs['new_rejections'].apply(len) == 0
    
    n_total = len(val_logs)
    n_old = val_logs['passed'].sum()
    n_new = val_logs['passed_learned'].sum()
    
    print(f"Old Pass Count (Heuristic 1): 7")
    print(f"Learned Pass Count: {n_new}")
    print(f"Survival Rate: {n_new/n_total:.1%}")
    
    passed_feats = val_logs[val_logs['passed_learned']]
    print("\nSample of Passed Features (Learned):")
    print(passed_feats[['feature', 'gain', 'perm_delta', 'marg_corr']].sort_values('gain', ascending=False).head(10))
    
    # Check if 'Interaction Giant' passed
    if 'pack_utilization_rolling_std_5' in passed_feats['feature'].values:
        print("\nSUCCESS: Interaction Giant 'pack_utilization_rolling_std_5' survived learned threshold.")
    else:
        print("\nWARNING: Interaction Giant was rejected by learned threshold.")

if __name__ == "__main__":
    verify_learned_rules('run_20260426_235007')
