import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def raw_separation_audit():
    print(f"--- [RAW SEPARATION AUDIT] ---")
    
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    q90 = np.percentile(y_true, 90)
    y_binary = (y_true >= q90).astype(int)
    
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    numeric_df = train_df.select_dtypes(include=[np.number])
    
    results = []
    for col in numeric_df.columns:
        if col in ['ID', 'scenario_id', 'layout_id']: continue
        
        vals = numeric_df[col].values
        # Point Biserial Correlation (using Spearman as proxy for non-linear/rank)
        corr, _ = spearmanr(vals, y_binary)
        
        # Mean difference
        m_tail = np.mean(vals[y_binary == 1])
        m_non = np.mean(vals[y_binary == 0])
        dist = np.abs(m_tail - m_non) / (np.std(vals) + 1e-9)
        
        results.append({"feature": col, "spearman": corr, "dist": dist})
        
    df_res = pd.DataFrame(results).sort_values("dist", ascending=False)
    
    print("\n[TOP 10 RAW FEATURES BY SEPARATION DISTANCE]")
    print(df_res.head(10))
    
    print("\n[SUMMARY STATISTICS]")
    print(f"Max Separation Distance: {df_res['dist'].max():.4f}")
    print(f"Median Separation Distance: {df_res['dist'].median():.4f}")
    print(f"Percentage of features with dist > 0.5: {len(df_res[df_res['dist'] > 0.5]) / len(df_res) * 100:.2f}%")

if __name__ == "__main__":
    raw_separation_audit()
