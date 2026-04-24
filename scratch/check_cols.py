import pandas as pd
import numpy as np

train = pd.read_csv('./data/train.csv')
layout = pd.read_csv('./data/layout_info.csv')
train = train.merge(layout, on='layout_id', how='left')

non_numeric = []
for col in train.columns:
    try:
        train[col].astype(np.float32)
    except:
        non_numeric.append((col, str(train[col].dtype)))

print("Non-numeric columns:")
for col, dtype in non_numeric:
    print(f"  - {col} ({dtype})")
