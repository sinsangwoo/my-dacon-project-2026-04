import os
import pickle
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import sys

# Add src to path
sys.path.append('.')
from src.data_loader import apply_latent_features

# Config
run_id = "run_20260426_111910"
base_path = f"outputs/{run_id}"

def forensic_overfitting_detection():
    print("=== PHASE 1: OVERFITTING DETECTION (SUBSET 50K) ===")
    
    # Load base data
    print("Loading base data...")
    with open(f"{base_path}/processed/train_base.pkl", "rb") as f:
        X_base_full = pickle.load(f)
    y_full = np.load(f"{base_path}/processed/y_train.npy")
    
    # Sample 50k
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_base_full), min(50000, len(X_base_full)), replace=False)
    X_base = X_base_full.iloc[sample_indices].reset_index(drop=True)
    y = y_full[sample_indices]
    
    # Load feature names for Fold 0
    with open(f"{base_path}/models/reconstructors/features_fold_0.pkl", "rb") as f:
        features_f0 = pickle.load(f)
    
    # Load reconstructor and scalers for Fold 0
    with open(f"{base_path}/models/reconstructors/recon_fold_0.pkl", "rb") as f:
        recon = pickle.load(f)
    with open(f"{base_path}/models/reconstructors/scaler_fold_0.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    # Regenerate FULL features for Fold 0 on subset
    print("Regenerating latent features for subset...")
    X_full_f0 = apply_latent_features(X_base, recon, scaler=scaler, selected_features=features_f0, is_train=True)
    X_f0 = X_full_f0[features_f0]
    
    # 1. Generalization Gap Analysis (on subset)
    print("\n--- Task 1: Generalization Gap Analysis (Fold 0, Subset) ---")
    with open(f"{base_path}/models/lgbm/model_fold_0.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Since we sampled, we'll just do a new split on the subset for gap analysis
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    tr_idx, val_idx = list(kf.split(X_f0, y))[0]
    
    X_tr, X_val = X_f0.iloc[tr_idx], X_f0.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    
    tr_preds = model.predict(X_tr)
    val_preds = model.predict(X_val)
    
    tr_mae = mean_absolute_error(y_tr, tr_preds)
    val_mae = mean_absolute_error(y_val, val_preds)
    
    print(f"Subset Train MAE={tr_mae:.4f}, Valid MAE={val_mae:.4f}, Gap={val_mae - tr_mae:.4f}")
    
    # 2. Permutation Importance Attack
    print("\n--- Task 2: Permutation Importance Attack ---")
    base_mae = val_mae
    imp = model.booster_.feature_importance(importance_type='gain')
    top_indices = np.argsort(imp)[-10:][::-1]
    top_features = [features_f0[idx] for idx in top_indices]
    
    X_val_p = X_val.copy()
    for feat in top_features:
        save = X_val_p[feat].copy()
        X_val_p[feat] = np.random.permutation(X_val_p[feat].values)
        perm_mae = mean_absolute_error(y_val, model.predict(X_val_p))
        X_val_p[feat] = save
        delta = perm_mae - base_mae
        print(f"Feature: {feat:40} | Delta MAE: {delta:.4f}")

    # 3. Random Noise Injection Test
    print("\n--- Task 3: Random Noise Injection Test ---")
    X_noise = X_f0.iloc[tr_idx].copy()
    for i in range(10):
        X_noise[f'RANDOM_NOISE_{i}'] = np.random.randn(len(X_noise))
    
    noise_model = LGBMRegressor(n_estimators=100, num_leaves=31, random_state=42, verbose=-1)
    noise_model.fit(X_noise, y_tr)
    
    noise_imp = noise_model.booster_.feature_importance(importance_type='split')
    noise_feat_names = list(X_noise.columns)
    
    used_noise = 0
    for i, name in enumerate(noise_feat_names):
        if 'RANDOM_NOISE' in name:
            if noise_imp[i] > 0:
                used_noise += 1
                print(f"Noise Feature {name} used in splits: {noise_imp[i]} times")
    
    print(f"Total Noise Features Used: {used_noise} / 10")

    # 4. Feature Importance Stability
    print("\n--- Task 4: Feature Importance Stability ---")
    fold_top_30 = []
    for i in range(5):
        with open(f"{base_path}/models/lgbm/model_fold_{i}.pkl", "rb") as f:
            m = pickle.load(f)
        with open(f"{base_path}/models/reconstructors/features_fold_{i}.pkl", "rb") as f:
            f_names = pickle.load(f)
        imp = m.booster_.feature_importance(importance_type='gain')
        top_30_idx = np.argsort(imp)[-30:][::-1]
        fold_top_30.append(set([f_names[idx] for idx in top_30_idx]))
    
    common_top_30 = set.intersection(*fold_top_30)
    print(f"Common Top 30 features across all folds: {len(common_top_30)}")
    print(f"Overlap Ratio: {len(common_top_30)/30:.2%}")

    # 5. Usage Coverage
    print("\n--- Task 5: Feature Usage Coverage ---")
    imp_split = model.booster_.feature_importance(importance_type='split')
    used_features = np.sum(imp_split > 0)
    print(f"Used features in Fold 0: {used_features} / {len(features_f0)}")

if __name__ == "__main__":
    forensic_overfitting_detection()
