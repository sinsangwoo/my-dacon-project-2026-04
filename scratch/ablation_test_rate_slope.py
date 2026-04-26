import os
import sys
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

sys.path.append('.')
from src.config import Config
from src.data_loader import build_base_features, apply_latent_features
from src.schema import FEATURE_SCHEMA

run_id = 'run_20260426_111910'

with open(f'outputs/{run_id}/models/reconstructors/features_fold_0.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print("Loading test data...")
train = pd.read_csv(f'./data/train.csv')
if 'avg_delay_minutes_next_30m' in train.columns:
    train = train.rename(columns={'avg_delay_minutes_next_30m': Config.TARGET})
layout = pd.read_csv('./data/layout_info.csv')
train = train.merge(layout, on="layout_id", how="left")

print("Building base features on a 15k subset for fast ablation...")
train_sub = train.sample(15000, random_state=42).copy()
y = train_sub[Config.TARGET].values

df_base, _, _ = build_base_features(train_sub)

X_df = df_base.copy()
if Config.TARGET in X_df.columns:
    X_df = X_df.drop(columns=[Config.TARGET])

def evaluate_feature_set(name, selected_features):
    print(f"\n--- Evaluating: {name} ({len(selected_features)} features) ---")
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    maes = []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_df, y)):
        tr_df = X_df.iloc[tr_idx].copy()
        val_df = X_df.iloc[val_idx].copy()
        y_tr = y[tr_idx]
        y_val = y[val_idx]
        
        raw_cols = [c for c in FEATURE_SCHEMA['raw_features'] if c in tr_df.columns]
        from src.utils import DriftShieldScaler
        from sklearn.preprocessing import StandardScaler
        
        s = DriftShieldScaler()
        s.fit(tr_df, raw_cols)
        tr_drifted = s.transform(tr_df, raw_cols)
        val_drifted = s.transform(val_df, raw_cols)
        
        n_s = StandardScaler()
        tr_scaled = tr_drifted.copy()
        val_scaled = val_drifted.copy()
        tr_scaled[raw_cols] = n_s.fit_transform(tr_drifted[raw_cols])
        val_scaled[raw_cols] = n_s.transform(val_drifted[raw_cols])
        
        from src.data_loader import SuperchargedPCAReconstructor
        recon = SuperchargedPCAReconstructor(input_dim=len(raw_cols))
        pca_cols = [c for c in Config.PCA_INPUT_COLS if c in tr_scaled.columns]
        recon.fit(tr_scaled[pca_cols].values)
        recon.build_fold_cache(tr_scaled) # FIX
        
        avail_features = [f for f in selected_features if f in tr_scaled.columns or 'd13' in f or 'd' in f] 
        
        tr_final = apply_latent_features(tr_scaled, recon, scaler=None, selected_features=selected_features, is_train=True)
        val_final = apply_latent_features(val_scaled, recon, scaler=None, selected_features=selected_features, is_train=False)
        
        X_tr = tr_final[selected_features].values.astype(np.float32)
        X_val = val_final[selected_features].values.astype(np.float32)
        
        model = LGBMRegressor(**Config.EMBED_LGBM_PARAMS)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        maes.append(mae)
        print(f"Fold {fold} MAE: {mae:.4f}")
        recon.clear_fold_cache()
        
    print(f"Mean MAE: {np.mean(maes):.4f}")
    return np.mean(maes)

all_feats = [f for f in feature_names if f in FEATURE_SCHEMA['all_features']]
no_rate_slope = [f for f in all_feats if '_rate_' not in f and '_slope_' not in f]

evaluate_feature_set("All Features (Baseline)", all_feats)
evaluate_feature_set("Removed Rate/Slope (Case A)", no_rate_slope)
