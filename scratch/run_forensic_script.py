import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

run_dir = 'outputs/run_20260429_103656'
logs_dir = 'logs/run_20260429_103656'

print("=== FORENSIC ANALYSIS OF RUN 20260429_103656 ===")
print("Loading data...")
oof_pred = np.load(f'{run_dir}/predictions/oof_stable.npy')
y_true = np.load(f'{run_dir}/processed/y_train.npy')
test_pred = np.load(f'{run_dir}/predictions/test_stable.npy')

train_base = pd.read_pickle(f'{run_dir}/processed/train_base.pkl')
test_base = pd.read_pickle(f'{run_dir}/processed/test_base.pkl')

print("Loaded. Computing metrics...")

# 1. CV vs LB Gap (CV=9.03, LB=10.48)
print("\n[STEP 1 - CV vs LB GAP]")
print(f"Overall OOF MAE: {mean_absolute_error(y_true, oof_pred):.4f}")
print(f"y_true mean: {np.mean(y_true):.4f}, std: {np.std(y_true):.4f}")
print(f"OOF pred mean: {np.mean(oof_pred):.4f}, std: {np.std(oof_pred):.4f}")
print(f"Test pred mean: {np.mean(test_pred):.4f}, std: {np.std(test_pred):.4f}")

# Simulating folds (assuming 5-fold sequential)
fold_size = len(y_true) // 5
for i in range(5):
    start = i * fold_size
    end = (i+1) * fold_size if i < 4 else len(y_true)
    y_fold = y_true[start:end]
    p_fold = oof_pred[start:end]
    mae = mean_absolute_error(y_fold, p_fold)
    print(f"Fold {i} - Target Mean: {np.mean(y_fold):.2f}, Pred Mean: {np.mean(p_fold):.2f}, Pred Std: {np.std(p_fold):.2f}, MAE: {mae:.4f}")

# 2. Prediction Distribution Collapse
print("\n[STEP 2 - PREDICTION DISTRIBUTION]")
quantiles = [0.5, 0.9, 0.99]
for q in quantiles:
    t_q = np.quantile(y_true, q)
    p_q = np.quantile(oof_pred, q)
    print(f"Quantile {q}: y_true={t_q:.2f}, y_pred={p_q:.2f}, ratio={p_q/t_q:.4f}")

# 3. Tail Failure Forensic
print("\n[STEP 3 - TAIL FAILURE]")
p99_idx = np.argsort(y_true)[-int(0.01*len(y_true)):]
y_tail = y_true[p99_idx]
p_tail = oof_pred[p99_idx]
print(f"Top 1% (N={len(y_tail)}) Target Range: {np.min(y_tail):.2f} - {np.max(y_tail):.2f}")
print(f"Top 1% Pred Range: {np.min(p_tail):.2f} - {np.max(p_tail):.2f}")
print(f"Top 1% MAE: {mean_absolute_error(y_tail, p_tail):.4f}")

# Let's check tail features
train_tail = train_base.iloc[p99_idx]
train_normal = train_base.drop(train_base.index[p99_idx])
print("Tail Feature Discrepancies (Top 5 largest absolute diff in means):")
diffs = (train_tail.mean(numeric_only=True) - train_normal.mean(numeric_only=True)).abs() / (train_normal.std(numeric_only=True) + 1e-6)
diffs = diffs.sort_values(ascending=False).head(5)
print(diffs)

# 4. Temporal Drift
print("\n[STEP 4 - TEMPORAL DRIFT]")
# split train into 10 temporal segments
decile_size = len(y_true) // 10
for i in range(10):
    start = i * decile_size
    end = (i+1) * decile_size if i < 9 else len(y_true)
    y_seg = y_true[start:end]
    p_seg = oof_pred[start:end]
    print(f"Decile {i} - Target Mean: {np.mean(y_seg):.2f}, MAE: {mean_absolute_error(y_seg, p_seg):.4f}")

# 5. Model Capacity / Leaf Utilization & Feature Importance
print("\n[STEP 7 - MODEL CAPACITY & FEATURE IMPORTANCE]")
import pickle
model_files = [f for f in os.listdir(f'{run_dir}/models/lgbm') if f.endswith('.pkl')]
if model_files:
    with open(f'{run_dir}/models/lgbm/{model_files[0]}', 'rb') as f:
        bst = pickle.load(f)
    print("Model loaded successfully.")
    if hasattr(bst, 'feature_importances_'):
        importances = bst.feature_importances_
        features = train_base.columns.tolist()
        # if train_base has more columns than model features, just truncate or ignore
        if len(importances) == len(features):
            fi_df = pd.DataFrame({'feature': features, 'importance': importances})
            fi_df = fi_df.sort_values(by='importance', ascending=False).head(10)
            print("Top 10 Global Features:")
            print(fi_df)
        else:
            print(f"Num importances: {len(importances)}, Num columns: {len(features)}. Mismatch, skipping feature names.")
            print("Top 10 raw importances:")
            print(sorted(importances, reverse=True)[:10])
else:
    print("No .pkl model files found.")

