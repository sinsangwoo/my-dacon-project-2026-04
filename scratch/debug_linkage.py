import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

df = pd.read_csv('data/train.csv')
num_cols = df.select_dtypes(include=[np.number]).columns
sample_df = df[num_cols].sample(n=min(20000, len(df)), random_state=42).astype("float32")
sample_filled = sample_df.fillna(sample_df.median())
corr_matrix = sample_filled.corr().abs().fillna(0.0)
corr_values = corr_matrix.to_numpy(copy=True)
np.fill_diagonal(corr_values, 0)

corr_dist = 1.0 - corr_values
np.fill_diagonal(corr_dist, 0)
corr_dist = np.clip(corr_dist, 0, 1)

print("Any NaN in corr_dist?", np.isnan(corr_dist).any())
print("Any Inf in corr_dist?", np.isinf(corr_dist).any())

condensed_dist = squareform(corr_dist, checks=False)
try:
    Z = linkage(condensed_dist, method='average')
    print("Success")
except Exception as e:
    print("Error:", e)
