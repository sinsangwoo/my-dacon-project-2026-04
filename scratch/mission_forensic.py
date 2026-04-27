import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

# Add project root to path
sys.path.append("c:/Github_public/my_dacon_project/my-dacon-project-2026-04")
from src.config import Config
from src.data_loader import apply_latent_features
from src.distribution import DistributionAuditor
import pickle
import lightgbm as lgb
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")

def main():
    RUN_ID = "run_20260427_124401"
    OUTPUT_DIR = f"outputs/{RUN_ID}"
    
    print("Loading data...")
    train_df = pd.read_pickle(f"{OUTPUT_DIR}/processed/train_base.pkl")
    test_df = pd.read_pickle(f"{OUTPUT_DIR}/processed/test_base.pkl")
    
    raw_train = pd.read_csv("c:/Github_public/my_dacon_project/my-dacon-project-2026-04/data/train.csv")
    y_full = raw_train['avg_delay_minutes_next_30m'].values
    y = y_full[train_df.index]
    
    oof_base = np.zeros(len(y))
    oof_tail = np.zeros(len(y))
    test_base = np.zeros(len(test_df))
    test_tail = np.zeros(len(test_df))
    
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    
    print("Running exact inference loop...")
    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_df)):
        # Load artifacts
        with open(f"{OUTPUT_DIR}/models/reconstructors/features_fold_{fold}.pkl", 'rb') as f:
            fold_features = pickle.load(f)
        with open(f"{OUTPUT_DIR}/models/reconstructors/scaler_fold_{fold}.pkl", 'rb') as f:
            scaler = pickle.load(f)
        with open(f"{OUTPUT_DIR}/models/reconstructors/norm_scaler_fold_{fold}.pkl", 'rb') as f:
            norm_scaler = pickle.load(f)
        with open(f"{OUTPUT_DIR}/models/reconstructors/recon_fold_{fold}.pkl", 'rb') as f:
            reconstructor = pickle.load(f)
        with open(f"{OUTPUT_DIR}/models/lgbm/model_fold_{fold}.pkl", 'rb') as f:
            base_model = pickle.load(f)
        with open(f"{OUTPUT_DIR}/models/lgbm/tail_model_fold_{fold}.pkl", 'rb') as f:
            tail_model = pickle.load(f)
            
        raw_cols = list(scaler.stats.keys())
        
        # Valid features
        val_df_fold = train_df.iloc[val_idx].copy()
        val_df_drifted = scaler.transform(val_df_fold, raw_cols)
        val_df_scaled = val_df_drifted.copy()
        val_df_scaled[raw_cols] = norm_scaler.transform(val_df_drifted[raw_cols])
        val_df_full = apply_latent_features(val_df_scaled, reconstructor, scaler=None, selected_features=fold_features, is_train=False)
        X_val_fold = val_df_full[fold_features].values.astype(np.float32)
        
        oof_base[val_idx] = base_model.predict(X_val_fold)
        oof_tail[val_idx] = tail_model.predict(X_val_fold)
        
        # Test features
        test_df_drifted = scaler.transform(test_df, raw_cols)
        test_df_scaled = test_df_drifted.copy()
        test_df_scaled[raw_cols] = norm_scaler.transform(test_df_drifted[raw_cols])
        test_df_full = apply_latent_features(test_df_scaled, reconstructor, scaler=None, selected_features=fold_features, is_train=False)
        X_te = test_df_full[fold_features].values.astype(np.float32)
        
        test_base += base_model.predict(X_te) / 2
        test_tail += tail_model.predict(X_te) / 2
        
    print("\n" + "="*50)
    print("[MISSION 1] Tail Improvement -> MAE 영향 검증")
    print("="*50)
    
    q99_y = np.percentile(y, 99)
    tail_mask = y >= q99_y
    
    base_mae = mean_absolute_error(y, oof_base)
    tail_mae = mean_absolute_error(y, oof_tail)
    base_q99_mae = mean_absolute_error(y[tail_mask], oof_base[tail_mask])
    tail_q99_mae = mean_absolute_error(y[tail_mask], oof_tail[tail_mask])
    
    print(f"1. Base Only: MAE = {base_mae:.4f} | Q99_MAE = {base_q99_mae:.4f}")
    print(f"2. Tail Only: MAE = {tail_mae:.4f} | Q99_MAE = {tail_q99_mae:.4f}")
    
    print("\n3. Ensemble (Damping Sweep):")
    results = []
    
    for d in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        p80 = np.percentile(oof_base, 80)
        p95 = np.percentile(oof_base, 95)
        tf = np.clip((oof_base - p80) / (p95 - p80 + 1e-9), 0.0, 1.0)
        oof_ens = oof_base * (1.0 - tf) + (oof_base + (oof_tail - oof_base) * d) * tf
        
        mae = mean_absolute_error(y, oof_ens)
        q99 = mean_absolute_error(y[tail_mask], oof_ens[tail_mask])
        
        # Calculate contribution (Mission 4 data)
        diff = np.abs(oof_ens - oof_base)
        contrib = np.mean(diff / (oof_ens + 1e-9))
        
        results.append((d, mae, q99, contrib, oof_ens))
        
        mae_diff = mae - base_mae
        q99_diff = base_q99_mae - q99
        status = "VALID (Q99↓ MAE↓)" if q99_diff > 0 and mae_diff < 0 else "INVALID (Q99↓ MAE↑)"
        
        print(f"Damping {d:.1f} | MAE: {mae:.4f} (Δ {mae_diff:+.4f}) | Q99_MAE: {q99:.4f} (Δ {-q99_diff:+.4f}) -> {status}")

    print("\n" + "="*50)
    print("[MISSION 1] Tail Mean Model 구축")
    print("="*50)
    
    q95_y = np.percentile(y, 95)
    tail_idx_all = np.where(y >= q95_y)[0]
    
    oof_tail_mean = np.zeros(len(y))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_df)):
        with open(f"outputs/{RUN_ID}/models/reconstructors/features_fold_{fold}.pkl", 'rb') as f_obj:
            features = pickle.load(f_obj)
        
        # Identify tail samples in the TRAINING set of this fold
        tr_tail_idx = [i for i in tr_idx if i in tail_idx_all]
        X_tr_tail = train_df.iloc[tr_tail_idx][features].values.astype(np.float32)
        y_tr_tail = y[tr_tail_idx]
        
        # Train Tail Mean Model (Standard Regression)
        # Using L1 (MAE) objective for LB optimization
        tail_mean_m = lgb.LGBMRegressor(objective='regression', metric='mae', n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
        tail_mean_m.fit(X_tr_tail, y_tr_tail)
        
        X_val = train_df.iloc[val_idx][features].values.astype(np.float32)
        oof_tail_mean[val_idx] = tail_mean_m.predict(X_val)
        
    # Evaluation on Tail Region (Actual y >= Q95)
    base_tail_mae = mean_absolute_error(y[tail_idx_all], oof_base[tail_idx_all])
    mean_tail_mae = mean_absolute_error(y[tail_idx_all], oof_tail_mean[tail_idx_all])
    
    print(f"Base Model MAE on Tail: {base_tail_mae:.4f}")
    print(f"Tail Mean Model MAE on Tail: {mean_tail_mae:.4f}")
    print(f"Improvement on Tail: {base_tail_mae - mean_tail_mae:.4f}")

    print("\n" + "="*50)
    print("[MISSION 2] Oracle Selective + Mean Model")
    print("="*50)
    
    oof_oracle_mean = oof_base.copy()
    oof_oracle_mean[tail_idx_all] = oof_tail_mean[tail_idx_all]
    oracle_mean_mae = mean_absolute_error(y, oof_oracle_mean)
    
    print(f"Base Only MAE: {mean_absolute_error(y, oof_base):.4f}")
    print(f"Oracle Selective (Mean) MAE: {oracle_mean_mae:.4f}")
    print(f"-> 이론적 MAE 개선 가능량: {mean_absolute_error(y, oof_base) - oracle_mean_mae:+.4f}")

    print("\n" + "="*50)
    print("[MISSION 3] Predictive Selective 적용 (Tail Mean)")
    print("="*50)
    
    # Using classifier probs from previous mission (Mission 2 logic remained in script conceptually)
    # Re-run classifier if needed, but let's assume clf_probs is available or re-calculate
    y_bin = (y >= q95_y).astype(int)
    clf_probs = np.zeros(len(y))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_df)):
        with open(f"outputs/{RUN_ID}/models/reconstructors/features_fold_{fold}.pkl", 'rb') as f_obj:
            features = pickle.load(f_obj)
        tr_indices = list(set(range(len(y))) - set(val_idx))
        X_tr = train_df.iloc[tr_indices][features].values.astype(np.float32)
        y_tr_bin = y_bin[tr_indices]
        clf = lgb.LGBMClassifier(n_estimators=50, max_depth=5, random_state=42, verbose=-1)
        clf.fit(X_tr, y_tr_bin)
        clf_probs[val_idx] = clf.predict_proba(train_df.iloc[val_idx][features].values.astype(np.float32))[:, 1]

    for thr in [0.2, 0.4, 0.6, 0.8]:
        oof_pred_mean = oof_base.copy()
        mask = clf_probs >= thr
        oof_pred_mean[mask] = oof_tail_mean[mask]
        
        p_mae = mean_absolute_error(y, oof_pred_mean)
        print(f"Threshold {thr:.1f} | MAE: {p_mae:.4f}")

    print("\n" + "="*50)
    print("[MISSION 4] Quantile vs Mean 비교")
    print("="*50)
    
    # Quantile Bias (from previous results conceptual)
    q_bias = np.mean(oof_tail[tail_idx_all] - y[tail_idx_all])
    m_bias = np.mean(oof_tail_mean[tail_idx_all] - y[tail_idx_all])
    
    print(f"Quantile Model Bias (Tail): {q_bias:.4f}")
    print(f"Mean Model Bias (Tail): {m_bias:.4f}")
    
    q_mae_tail = mean_absolute_error(y[tail_idx_all], oof_tail[tail_idx_all])
    m_mae_tail = mean_absolute_error(y[tail_idx_all], oof_tail_mean[tail_idx_all])
    
    print(f"Quantile Model MAE (Tail): {q_mae_tail:.4f}")
    print(f"Mean Model MAE (Tail): {m_mae_tail:.4f}")



if __name__ == "__main__":
    main()
