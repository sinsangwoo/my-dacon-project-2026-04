import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_audit(run_id):
    print(f"=== [FORENSIC AUDIT: {run_id}] ===")
    
    log_path = f"outputs/{run_id}/processed/signal_validation_logs.json"
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return
        
    with open(log_path, 'r') as f:
        data = json.load(f)
        
    val_logs = pd.DataFrame(data['val_logs'])
    
    # ---------------------------------------------------------
    # TASK 1: KS THRESHOLD VALIDATION
    # ---------------------------------------------------------
    print("\n--- TASK 1: KS THRESHOLD VALIDATION ---")
    
    # 1.1 Distribution of KS
    ks_stats = val_logs['ks_stat'].describe()
    print("KS Statistic Distribution:")
    print(ks_stats)
    
    # Histogram-like summary
    bins = [0, 0.05, 0.15, 0.30, 0.50, 0.85, 1.0]
    labels = ['Stable (<0.05)', 'Low (0.05-0.15)', 'Med (0.15-0.3)', 'High (0.3-0.5)', 'Danger (0.5-0.85)', 'Catastrophic (>0.85)']
    val_logs['ks_bin'] = pd.cut(val_logs['ks_stat'], bins=bins, labels=labels)
    bin_counts = val_logs['ks_bin'].value_counts().sort_index()
    print("\nKS Bin Counts:")
    print(bin_counts)
    
    # 1.2 Relationship between KS and Importance/MAE
    # Importance = gain, Impact = perm_delta (OOF MAE increase when shuffled)
    
    danger_zone = val_logs[val_logs['ks_stat'] > 0.5]
    print(f"\nFeatures with KS > 0.5: {len(danger_zone)}")
    if not danger_zone.empty:
        print(danger_zone[['feature', 'ks_stat', 'gain', 'perm_delta', 'passed', 'rejection_reasons']].sort_values('perm_delta', ascending=False).head(10))
    
    # Identify high KS but useful signals
    useful_high_ks = val_logs[(val_logs['ks_stat'] > 0.5) & (val_logs['perm_delta'] > 0.001)]
    print(f"\nPotential 'Useful but High KS' features (KS > 0.5, perm_delta > 0.001): {len(useful_high_ks)}")
    if not useful_high_ks.empty:
        print(useful_high_ks[['feature', 'ks_stat', 'gain', 'perm_delta', 'passed']])

    # 1.3 Is KS > 0.85 justified?
    catastrophic = val_logs[val_logs['ks_stat'] > 0.85]
    print(f"\nFeatures with KS > 0.85: {len(catastrophic)}")
    if not catastrophic.empty:
        print("Sample of Catastrophic features:")
        print(catastrophic[['feature', 'ks_stat', 'gain', 'perm_delta']].head(5))

    # ---------------------------------------------------------
    # TASK 2: MARGINAL CORRELATION FLOOR (0.01)
    # ---------------------------------------------------------
    print("\n--- TASK 2: MARGINAL CORRELATION FLOOR (0.01) VALIDATION ---")
    
    # identify low corr but high importance
    low_corr_high_imp = val_logs[(abs(val_logs['marg_corr']) < 0.01) & (val_logs['gain'] > val_logs['gain'].median())]
    print(f"Low Marginal Correlation (<0.01) but High Gain (>{val_logs['gain'].median():.1f}) features: {len(low_corr_high_imp)}")
    if not low_corr_high_imp.empty:
        # Table output
        print("\n| feature | corr | gain | perm_delta | passed | rejection_reasons |")
        print("|---------|------|------|------------|--------|-------------------|")
        for _, r in low_corr_high_imp.sort_values('gain', ascending=False).head(20).iterrows():
            print(f"| {r['feature']} | {r['marg_corr']:.4f} | {r['gain']} | {r['perm_delta']:.6f} | {r['passed']} | {r['rejection_reasons']} |")

    # [TASK 2 Interaction Check]
    # If a feature has low corr but high gain, it might be interacting.
    # Check rejection reasons for these
    if not low_corr_high_imp.empty:
        print("\nRejection Reasons for Low-Corr/High-Gain features:")
        print(low_corr_high_imp['rejection_reasons'].apply(lambda x: ', '.join(x)).value_counts())

    # ---------------------------------------------------------
    # TASK 3: UNDERFITTING RISK ASSESSMENT
    # ---------------------------------------------------------
    print("\n--- TASK 3: UNDERFITTING RISK ASSESSMENT ---")
    
    n_total = len(val_logs)
    n_passed = val_logs['passed'].sum()
    print(f"Feature count change: {n_total} -> {n_passed} ({n_passed/n_total:.1%})")
    
    # Impact on MAE
    potential_mae_loss = val_logs[~val_logs['passed'] & (val_logs['perm_delta'] > 0)]['perm_delta'].sum()
    print(f"Potential MAE increase from rejections: {potential_mae_loss:.6f}")
    
    # Noise Survival
    if 'noise_metrics' in data:
        nm = data['noise_metrics']
        print(f"\nNoise Immunity Proof:")
        print(f"→ Noise Survival Rate: {nm['noise_survival_rate']:.2%}")
        print(f"→ Noise Max Gain: {nm['noise_max_gain']:.4f}")
        
    print("\n=== AUDIT COMPLETE ===")

if __name__ == "__main__":
    run_audit('run_20260426_235007')
