import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import sys
sys.path.append('.')
from src.data_loader import build_base_features
from src.schema import FEATURE_SCHEMA
from src.config import Config

print("Loading data...")
train = pd.read_csv('./data/train.csv')
if 'avg_delay_minutes_next_30m' in train.columns:
    train = train.rename(columns={'avg_delay_minutes_next_30m': Config.TARGET})
layout = pd.read_csv('./data/layout_info.csv')
train = train.merge(layout, on="layout_id", how="left")

# Sample for speed
train = train.sample(20000, random_state=42).copy()
y = train[Config.TARGET].values

# Ultra fast feature gen
print("Generating features...")
df, _, _ = build_base_features(train)

X = df.drop(columns=[Config.TARGET, 'ID', 'scenario_id', 'layout_id'], errors='ignore')
# Fillna
X = X.fillna(0)

# Feature sets
all_f = list(X.columns)
no_rs = [c for c in all_f if '_rate_' not in c and '_slope_' not in c]
only_rs = [c for c in all_f if '_rate_' in c or '_slope_' in c]
# need at least some base features to make sense for only_rs
if len(only_rs) == 0:
    only_rs = all_f[:5]

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def run_ablation(name, feats):
    model = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_tr[feats], y_tr)
    p = model.predict(X_val[feats])
    mae = mean_absolute_error(y_val, p)
    print(f"{name} MAE: {mae:.4f} (Features: {len(feats)})")

print("\n=== Task 3: Ablation Test ===")
run_ablation("Case 0: All Features", all_f)
run_ablation("Case A: No Rate/Slope", no_rs)
run_ablation("Case B: Only Rate/Slope", only_rs)
