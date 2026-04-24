import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.schema import BASE_COLS
from src.utils import DriftShieldScaler

train = pd.read_csv('data/train.csv', nrows=50000)
scaler = DriftShieldScaler()
scaler.fit(train, BASE_COLS)
df = scaler.transform(train, BASE_COLS)

current = list(BASE_COLS)
target = 'order_inflow_15m'

while True:
    X = df[current].fillna(0).values
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    pca = PCA(n_components=8)
    pca.fit(X)
    var_sum = np.sum(pca.explained_variance_ratio_)
    
    if var_sum >= 0.8:
        print(f"RESULT: {current}")
        print(f"VARIANCE: {var_sum}")
        break
        
    loadings = np.sum(np.abs(pca.components_), axis=0)
    # Heuristic to keep target
    load_scores = list(loadings)
    target_idx = current.index(target)
    load_scores[target_idx] = 999.0 # Don't remove target
    
    worst_idx = np.argmin(load_scores)
    current.pop(worst_idx)
