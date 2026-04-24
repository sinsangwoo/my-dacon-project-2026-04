import numpy as np
import pandas as pd

try:
    preds = np.load('outputs/predictions/final_submission.npy')
    print(f"Shape: {preds.shape}")
    print(f"Mean: {np.mean(preds)}")
    print(f"Std: {np.std(preds)}")
    print(f"Max: {np.max(preds)}")
    print(f"Min: {np.min(preds)}")
except Exception as e:
    print(f"Error loading final_submission.npy: {e}")

try:
    train = pd.read_csv('data/train.csv')
    target = train['avg_delay_minutes_next_30m']
    print(f"Train Mean: {target.mean()}")
    print(f"Train Std: {target.std()}")
    print(f"Train P90: {target.quantile(0.9)}")
    print(f"Train P99: {target.quantile(0.99)}")
except Exception as e:
    print(f"Error loading train.csv: {e}")
