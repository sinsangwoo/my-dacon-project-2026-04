import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error

def main():
    print("--- STEP 1 & 2 & 3: Array & Diagnostic Forensic ---")
    
    # Load targets
    y_train = np.load('outputs/processed/y_train.npy')
    
    # Load OOFs
    oof_lgb = np.load('outputs/predictions/oof_lgb.npy')
    oof_cat = np.load('outputs/predictions/oof_cat.npy')
    try:
        oof_stack = np.load('outputs/predictions/test_stack.npy')
    except:
        oof_stack = None
        
    print(f"Target MAE LGB: {mean_absolute_error(y_train, oof_lgb):.5f}")
    print(f"Target MAE CAT: {mean_absolute_error(y_train, oof_cat):.5f}")
    print(f"Correlation LGB vs CAT: {np.corrcoef(oof_lgb, oof_cat)[0, 1]:.5f}")
    
    # KS Test against target
    ks_lgb = ks_2samp(y_train, oof_lgb)
    ks_cat = ks_2samp(y_train, oof_cat)
    print(f"KS-Test Target vs OOF_LGB: statistic={ks_lgb.statistic:.4f}, pvalue={ks_lgb.pvalue:.4e}")
    print(f"KS-Test Target vs OOF_CAT: statistic={ks_cat.statistic:.4f}, pvalue={ks_cat.pvalue:.4e}")
    
    print("\n--- STEP 5: Leakage & Dimensionality Check ---")
    try:
        train_features = pd.read_json('outputs/processed/features_reduced.json', typ='series').tolist()
        print(f"Features reduced json len: {len(train_features)}")
    except:
        print("features_reduced.json not found")
        
    try:
        df = pd.read_pickle('outputs/processed/train_full.pkl')
        print(f"train_full.pkl shape: {df.shape}")
    except:
        pass

if __name__ == "__main__":
    main()
