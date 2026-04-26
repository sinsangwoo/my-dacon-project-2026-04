import sys
import os
sys.path.append(os.getcwd())
import pickle
import pandas as pd
import numpy as np

run_id = "run_20260426_111910"
base_path = f"outputs/{run_id}"

with open(f"{base_path}/processed/train_base.pkl", "rb") as f:
    X_base = pickle.load(f).iloc[:1000]
with open(f"{base_path}/models/reconstructors/features_fold_0.pkl", "rb") as f:
    features = pickle.load(f)
with open(f"{base_path}/models/reconstructors/recon_fold_0.pkl", "rb") as f:
    recon = pickle.load(f)
with open(f"{base_path}/models/reconstructors/scaler_fold_0.pkl", "rb") as f:
    scaler = pickle.load(f)

from src.data_loader import apply_latent_features
X_full = apply_latent_features(X_base, recon, scaler=scaler, selected_features=features, is_train=True)

corr = X_full[features].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_corr = [column for column in upper.columns if any(upper[column] > 0.95)]

print(f"Total High-Corr Features (>0.95): {len(high_corr)}")
for col in high_corr:
    sibling = upper[col][upper[col] > 0.95].index.tolist()[0]
    print(f"{col} <-> {sibling} (corr={upper[col][sibling]:.4f})")
