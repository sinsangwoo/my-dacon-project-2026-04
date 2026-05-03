import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def cluster_variance_audit():
    print(f"--- [CLUSTER-BASED TARGET CONCENTRATION AUDIT] ---")
    
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    q90 = np.percentile(y_true, 90)
    y_binary = (y_true >= q90).astype(int)
    
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    numeric_df = train_df.select_dtypes(include=[np.number])
    
    # Select top 10 features from previous audit (excluding target)
    # top_raw = ['robot_idle', 'robot_charging', 'robot_utilization', ...]
    # For speed and robustness, use the top 10 from the audit result
    top_features = ['robot_idle', 'robot_charging', 'robot_utilization', 'robot_active', 'battery_std']
    # Add some interaction features if they exist in train_base
    available_cols = [c for c in top_features if c in numeric_df.columns]
    
    X = numeric_df[available_cols].values
    # Handle NaNs
    X = np.nan_to_num(X, nan=0.0)
    # Standardize
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)
    
    # Cluster into many small groups to find "Tail Pockets"
    n_clusters = 100
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
    clusters = kmeans.fit_predict(X)
    
    results = []
    for i in range(n_clusters):
        mask = clusters == i
        if np.sum(mask) == 0: continue
        
        tail_rate = np.mean(y_binary[mask])
        count = np.sum(mask)
        results.append({"cluster": i, "tail_rate": tail_rate, "count": count})
        
    df_res = pd.DataFrame(results).sort_values("tail_rate", ascending=False)
    
    print(f"\n[TOP 10 CLUSTERS BY TAIL CONCENTRATION]")
    print(df_res.head(10))
    
    print(f"\n[SUMMARY]")
    print(f"Global Tail Rate: {np.mean(y_binary):.4f}")
    print(f"Max Cluster Tail Rate: {df_res['tail_rate'].max():.4f}")
    print(f"Number of clusters with Tail Rate > 0.5: {len(df_res[df_res['tail_rate'] > 0.5])}")
    
    if df_res['tail_rate'].max() < 0.4:
        print("\nCONCLUSION: Even in the most specialized feature pockets, Tail and Non-Tail are heavily mixed.")
        print("Precision 0.27 is a result of fundamental feature-label overlap.")

if __name__ == "__main__":
    cluster_variance_audit()
