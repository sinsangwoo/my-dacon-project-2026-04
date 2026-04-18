import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

def main():
    print("--- [STEP 1: GENERALIZATION FAILURE] ---")
    
    X_tr = np.load('outputs/processed/X_train_reduced.npy')
    X_te = np.load('outputs/processed/X_test_reduced.npy')
    try:
        # subset 5000 rows to speed up KS test
        idx_tr = np.random.choice(X_tr.shape[0], 5000, replace=False)
        idx_te = np.random.choice(X_te.shape[0], min(5000, X_te.shape[0]), replace=False)
        
        ks_stats = []
        for i in range(min(30, X_tr.shape[1])): # do top 30 features
            ks_stat, ks_p = ks_2samp(X_tr[idx_tr, i], X_te[idx_te, i])
            ks_stats.append(ks_stat)
        
        print(f"KS-Test Train vs Test Features (Top 30 subset) - Mean KS-stat: {np.mean(ks_stats):.4f}, Max KS-stat: {np.max(ks_stats):.4f}")
        high_drift = sum(k > 0.1 for k in ks_stats)
        print(f"Features with KS-stat > 0.1 (High Drift): {high_drift} / 30")
    except Exception as e:
        print(f"KS-Test error: {e}")

    oof_cat = np.load('outputs/predictions/oof_cat.npy')
    test_cat = np.load('outputs/predictions/test_cat.npy')
    print(f"OOF_CAT: mean={np.mean(oof_cat):.4f}, std={np.std(oof_cat):.4f}")
    print(f"TEST_CAT: mean={np.mean(test_cat):.4f}, std={np.std(test_cat):.4f}")
    
    oof_lgb = np.load('outputs/predictions/oof_lgb.npy')
    test_lgb = np.load('outputs/predictions/test_lgb.npy')
    print(f"OOF_LGB: mean={np.mean(oof_lgb):.4f}, std={np.std(oof_lgb):.4f}")
    print(f"TEST_LGB: mean={np.mean(test_lgb):.4f}, std={np.std(test_lgb):.4f}")

    print("\n--- [STEP 2: RESIDUAL ANALYSIS] ---")
    y_tr = np.load('outputs/processed/y_train.npy')
    
    err_cat = oof_cat - y_tr
    err_lgb = oof_lgb - y_tr
    
    # Bins: Low (< 10), Mid (10~30), High (> 30)
    bins = [
        ("Low (<10)", y_tr < 10),
        ("Mid (10-30)", (y_tr >= 10) & (y_tr < 30)),
        ("High (>30)", y_tr >= 30)
    ]
    
    for name, mask in bins:
        mae_cat = np.mean(np.abs(err_cat[mask]))
        mae_lgb = np.mean(np.abs(err_lgb[mask]))
        print(f"[{name}] CAT MAE: {mae_cat:.4f} | LGB MAE: {mae_lgb:.4f} | LGB is worse by: {mae_lgb - mae_cat:.4f}")
        
    print("\n--- [STEP 3: MULTICOLLINEARITY STRUCTURE] ---")
    try:
        # Sample for meta-stack analysis
        sample_idx = np.random.choice(X_tr.shape[0], 5000, replace=False)
        X_sample = X_tr[sample_idx]
        
        # Calculate condition number of X
        cond_number = np.linalg.cond(X_sample)
        print(f"Condition Number of Raw Train Features (X): {cond_number:.4e}")
        
        # OOF + X
        oof_meta = np.c_[oof_cat[sample_idx], oof_lgb[sample_idx]]
        X_aug = np.c_[X_sample, oof_meta]
        cond_aug = np.linalg.cond(X_aug)
        print(f"Condition Number of (X + OOF Meta): {cond_aug:.4e}")
        
        # Correlation of OOF with raw features
        corr_cat = np.array([np.corrcoef(X_sample[:, i], oof_cat[sample_idx])[0, 1] for i in range(X_sample.shape[1])])
        corr_lgb = np.array([np.corrcoef(X_sample[:, i], oof_lgb[sample_idx])[0, 1] for i in range(X_sample.shape[1])])
        
        high_corr_cat = np.sum(np.abs(corr_cat) > 0.8)
        high_corr_lgb = np.sum(np.abs(corr_lgb) > 0.8)
        print(f"Features highly correlated (>0.8) with CAT OOF: {high_corr_cat}")
        print(f"Features highly correlated (>0.8) with LGB OOF: {high_corr_lgb}")
    except Exception as e:
        print(f"Multicollinearity error: {e}")

    print("\n--- [STEP 4: SEQUENCE SIGNAL ANALYSIS] ---")
    try:
        groups = np.load('outputs/processed/scenario_id.npy')
        df_seq = pd.DataFrame({'scenario_id': groups, 'target': y_tr})
        df_seq['oof_cat'] = oof_cat
        
        # Calculate target variance per scenario (is it static or dynamic?)
        var_target = df_seq.groupby('scenario_id')['target'].var().mean()
        var_pred = df_seq.groupby('scenario_id')['oof_cat'].var().mean()
        
        print(f"Mean Target Variance within Scenario: {var_target:.4f}")
        print(f"Mean Prediction Variance within Scenario: {var_pred:.4f}")
        
        if var_pred < var_target * 0.3:
            print("=> ALERT: Model predictions are very static across time within scenarios!")
        else:
            print("=> Model captures some intra-scenario variance.")
            
        print("Detailed sequence behavior check done.")
    except Exception as e:
        print(f"Sequence analysis error: {e}")

if __name__ == "__main__":
    main()
