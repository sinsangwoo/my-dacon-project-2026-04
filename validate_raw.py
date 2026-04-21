import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import json
import os

# Load data
path = './outputs/fast_audit_B_2.0/processed/'
oof = np.load(os.path.join(path, 'oof_raw.npy'))
y = np.load(os.path.join(path, 'y_train.npy'))
groups = np.load(os.path.join(path, 'scenario_id.npy'), allow_pickle=True)

# Task 1: OOF Distribution
m_oof = np.mean(oof)
s_oof = np.std(oof)
m_y = np.mean(y)
s_y = np.std(y)

# Task 2: Feature Identity
# Since they are saved as numpy, we check shape and the manifest if available
X_tr_shape = np.load(os.path.join(path, 'X_train_raw.npy'), mmap_mode='r').shape
X_te_shape = np.load(os.path.join(path, 'X_test_raw.npy'), mmap_mode='r').shape

# Task 3: CV Stability
from sklearn.model_selection import GroupKFold
kf = GroupKFold(n_splits=2) # Using 2 splits as per the last run logic
fold_maes = []
for tr_idx, val_idx in kf.split(oof, y, groups=groups):
    fold_maes.append(mean_absolute_error(y[val_idx], oof[val_idx]))

print("[VALIDATION_RAW]")
print(f"TASK_1_OOF_MEAN: {m_oof:.6f}")
print(f"TASK_1_OOF_STD: {s_oof:.6f}")
print(f"TASK_1_Y_MEAN: {m_y:.6f}")
print(f"TASK_1_Y_STD: {s_y:.6f}")
print(f"TASK_1_RATIO_MEAN: {m_oof/m_y:.6f}")
print(f"TASK_1_RATIO_STD: {s_oof/s_y:.6f}")
print(f"TASK_2_TRAIN_SHAPE: {X_tr_shape}")
print(f"TASK_2_TEST_SHAPE: {X_te_shape}")
print(f"TASK_2_EQUAITY: {X_tr_shape[1] == X_te_shape[1]}")
print(f"TASK_3_FOLD_MAES: {fold_maes}")
print(f"TASK_3_MAE_VAR: {np.var(fold_maes):.6f}")
