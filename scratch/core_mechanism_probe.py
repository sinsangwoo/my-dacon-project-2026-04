import numpy as np
import pandas as pd
import sys
import os
import json
import gc
from lightgbm import LGBMRegressor

# Add project root to path
sys.path.append(os.getcwd())
from src.config import Config
from src.utils import load_npy
from sklearn.metrics import mean_absolute_error

RUN_ID = "run_20260429_133848"
PROCESSED_PATH = f"outputs/{RUN_ID}/processed"

def analyze_lgbm_internals():
    print("\n--- [CORE MECHANISM PROBE] ---")
    
    # 1. Load Data
    train_base = pd.read_pickle(f"{PROCESSED_PATH}/train_base.pkl")
    y = load_npy(f"{PROCESSED_PATH}/y_train.npy")
    
    # Use 80/20 split for quick analysis
    split_idx = int(len(y) * 0.8)
    X_tr = train_base.iloc[:split_idx].select_dtypes(include=[np.number]).fillna(0)
    y_tr = y[:split_idx]
    X_val = train_base.iloc[split_idx:].select_dtypes(include=[np.number]).fillna(0)
    y_val = y[split_idx:]
    
    # [STEP 1 & 3: Baseline with current Huber Loss]
    # In the pipeline, target is log-transformed. We should replicate that.
    # Wait, the pipeline does np.log1p(y_tr).
    y_tr_log = np.log1p(y_tr)
    y_val_log = np.log1p(y_val)
    
    model = LGBMRegressor(**Config.RAW_LGBM_PARAMS)
    if 'early_stopping_rounds' in model.get_params():
        model.set_params(early_stopping_rounds=None)
        
    model.fit(X_tr, y_tr_log)
    booster = model.booster_
    
    preds_log = model.predict(X_val)
    preds = np.expm1(preds_log)
    
    # Extract tree structure
    tree_df = booster.trees_to_dataframe()
    # In some versions, 'is_leaf' might not exist. Leaves have non-null 'leaf_value' or null 'split_feature'
    if 'is_leaf' in tree_df.columns:
        leaf_df = tree_df[tree_df['is_leaf'] == True].copy()
    else:
        # Fallback: leaves have no split_feature
        leaf_df = tree_df[tree_df['split_feature'].isna()].copy()
    
    # Ensure value column exists (might be 'value' or 'leaf_value')
    if 'value' not in leaf_df.columns and 'leaf_value' in leaf_df.columns:
        leaf_df['value'] = leaf_df['leaf_value']
    
    # Leaf value stats
    leaf_values = leaf_df['value']
    leaf_counts = leaf_df['count']
    
    print("\n[STEP 1: OUTPUT MAX 억제 원인 분석]")
    print(f"Target Max: {y_val.max():.4f} | Prediction Max: {preds.max():.4f}")
    
    print("\nLeaf Value Distribution (Log Scale internally):")
    print(f"Min:  {leaf_values.min():.4f}")
    print(f"Max:  {leaf_values.max():.4f} (Equivalent to Expm1: {np.expm1(leaf_values.max()):.4f})")
    print(f"Mean: {leaf_values.mean():.4f}")
    print(f"Std:  {leaf_values.std():.4f}")
    
    # Top leaf analysis
    max_leaf = leaf_df.loc[leaf_df['value'].idxmax()]
    print("\nTop Leaf (Max Value) Stats:")
    print(f"Leaf Value: {max_leaf['value']:.4f} | Sample Count: {max_leaf['count']}")
    print(f"Total leaves across all trees: {len(leaf_df)}")
    
    # Determine STEP 1 Answer
    if np.expm1(leaf_values.max()) < y_val.max() * 0.5:
        print("-> 판정: A. leaf value 자체가 작음")
    elif max_leaf['count'] < 10:
        print("-> 판정: B. 큰 leaf가 있지만 샘플이 거의 안 감")
    else:
        print("-> 판정: C. split 구조가 tail region으로 분기 못함")
        
    # [STEP 3: std_ratio 원인 분해]
    print("\n[STEP 3: std_ratio = 0.5 원인 분해]")
    print(f"Target Std: {y_val.std():.4f} | Prediction Std: {preds.std():.4f}")
    std_ratio = preds.std() / (y_val.std() + 1e-9)
    print(f"Std Ratio: {std_ratio:.4f}")
    
    print(f"Leaf Value Std (Log Scale): {leaf_values.std():.4f}")
    # Entropy of sample routing
    total_samples = leaf_counts.sum()
    probs = leaf_counts / total_samples
    entropy = -np.sum(probs * np.log2(probs + 1e-9))
    print(f"Leaf Sample Routing Entropy: {entropy:.4f}")
    
    # [STEP 2: Loss Function Experiment]
    print("\n[STEP 2: LEAF VALUE가 작은 이유 (Loss Experiment)]")
    
    losses = ['regression_l1', 'huber', 'regression']
    results = []
    
    for loss in losses:
        params = Config.RAW_LGBM_PARAMS.copy()
        params['objective'] = loss
        if 'early_stopping_rounds' in params:
            del params['early_stopping_rounds']
            
        m = LGBMRegressor(**params)
        m.fit(X_tr, y_tr_log)
        
        p_log = m.predict(X_val)
        p = np.expm1(p_log)
        
        ldf = m.booster_.trees_to_dataframe()
        if 'is_leaf' in ldf.columns:
            leaves = ldf[ldf['is_leaf'] == True].copy()
        else:
            leaves = ldf[ldf['split_feature'].isna()].copy()
        
        if 'value' not in leaves.columns and 'leaf_value' in leaves.columns:
            leaves['value'] = leaves['leaf_value']
        
        res = {
            'Loss': loss,
            'Max Leaf (Log)': leaves['value'].max(),
            'Max Leaf (Exp)': np.expm1(leaves['value'].max()),
            'Max Pred': p.max(),
            'Std Ratio': p.std() / (y_val.std() + 1e-9)
        }
        results.append(res)
        
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    analyze_lgbm_internals()
