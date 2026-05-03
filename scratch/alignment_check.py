import os
import json
import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
from glob import glob

base_run = "run_20260429_103656"
curr_run = "run_20260430_120906"

def calc_alignment(run_id):
    model_paths = glob(f"outputs/{run_id}/models/lgbm/model_fold_*.pkl")
    if not model_paths:
        return None
        
    with open(model_paths[0], 'rb') as f:
        model = pickle.load(f)
        
    # We need validation data. Since we don't have it easily without running the pipeline,
    # let's try to generate random dummy data just to see if the model has internal structure changes
    # Wait, LGBM predict(pred_contrib) requires real data.
    # We can load train_base.pkl
    
    try:
        with open(f"outputs/{run_id}/processed/train_base.pkl", "rb") as f:
            train_df = pickle.load(f)
        
        # sample 1000 rows
        X = train_df.head(1000).select_dtypes(include=[np.number]).fillna(0).values
        
        # We don't know the exact columns used, but model.feature_name_ tells us
        # LGBM needs exact columns or a numpy array of same dimension
        # Actually, let's just use X if shapes match
        if X.shape[1] != model.n_features_in_:
            print(f"Shape mismatch in {run_id}: Data {X.shape[1]}, Model {model.n_features_in_}")
            return None
            
        contrib = model.predict(X, pred_contrib=True)
        # contrib shape: (n_samples, n_features + 1)
        
        # alignment_score = |sum(contrib)| / sum(|contrib|)
        # Calculate per sample, then average
        
        # exclude the bias term (last column)
        feature_contrib = contrib[:, :-1]
        
        sum_abs = np.sum(np.abs(feature_contrib), axis=1)
        abs_sum = np.abs(np.sum(feature_contrib, axis=1))
        
        # avoid division by zero
        mask = sum_abs > 1e-9
        align_scores = np.zeros_like(abs_sum)
        align_scores[mask] = abs_sum[mask] / sum_abs[mask]
        
        return np.mean(align_scores)
    except Exception as e:
        print(f"Error calculating alignment for {run_id}: {e}")
        return None

print(f"BASELINE Alignment Score: {calc_alignment(base_run)}")
print(f"CURRENT  Alignment Score: {calc_alignment(curr_run)}")

# Look at drift
try:
    base_drift = pd.read_csv(f"logs/{base_run}/summary/distribution/drift_audit_raw.csv")
    curr_drift = pd.read_csv(f"logs/{curr_run}/summary/distribution/drift_audit_raw.csv")
    
    # count drift > 0.1
    b_drift_cnt = sum(base_drift['ks_stat'] > 0.1)
    c_drift_cnt = sum(curr_drift['ks_stat'] > 0.1)
    print(f"BASELINE Drift > 0.1 features: {b_drift_cnt}")
    print(f"CURRENT  Drift > 0.1 features: {c_drift_cnt}")
except Exception as e:
    print("Drift info error:", e)
    
