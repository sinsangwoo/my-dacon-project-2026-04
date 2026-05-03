import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from scipy.stats import entropy

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def numerical_dissection():
    # 1. Load Data
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    q90 = np.percentile(y_true, 90)
    y_binary = (y_true >= q90).astype(int)
    
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    X_raw = train_df.select_dtypes(include=[np.number]).fillna(0)
    cols_to_drop = ['avg_delay_minutes_next_30m', 'ID', 'scenario_id', 'layout_id']
    X_raw = X_raw.drop(columns=[c for c in cols_to_drop if c in X_raw.columns])
    
    # [EXP 1: Coordinate Overlap]
    # Check samples with exactly the same features
    duplicates = X_raw.duplicated(keep=False)
    X_dupes = X_raw[duplicates]
    y_dupes = y_binary[duplicates]
    
    exact_match_variance = 0
    if len(y_dupes) > 0:
        # Group by all features and check label mean
        group_means = pd.Series(y_dupes).groupby(X_dupes.apply(tuple, axis=1)).mean()
        exact_match_variance = np.mean((group_means > 0) & (group_means < 1))

    # [EXP 2: Feature Resolution vs Consistency]
    results_res = []
    # Sample for speed
    idx_s = np.random.choice(len(y_binary), 10000, replace=False)
    X_s = X_raw.values[idx_s]
    y_s = y_binary[idx_s]
    
    for k in [10, 50, 100, 214]:
        knn = KNeighborsClassifier(n_neighbors=10).fit(X_s[:, :k], y_s)
        _, neigh_idx = knn.kneighbors(X_s[:, :k])
        consistency = np.mean(np.std(y_s[neigh_idx], axis=1) < 0.1)
        results_res.append({"K_Features": k, "Consistency": consistency})
    
    # [EXP 3: Latent Space Purity]
    pca = PCA(n_components=30).fit_transform(X_s)
    knn_pca = KNeighborsClassifier(n_neighbors=10).fit(pca, y_s)
    _, neigh_idx_pca = knn_pca.kneighbors(pca)
    consistency_pca = np.mean(np.std(y_s[neigh_idx_pca], axis=1) < 0.1)

    # [EXP 4: Temporal Target Entropy]
    # We need to sort by scenario and timestep
    df_temp = pd.DataFrame({'scen': train_df['scenario_id'], 'y': y_binary})
    # Since we don't have timestep, we use the order in pkl (assuming it's chronological)
    df_temp['prev_y'] = df_temp.groupby('scen')['y'].shift(1)
    df_temp = df_temp.dropna()
    target_flip_rate = np.mean(df_temp['y'] != df_temp['prev_y'])

    # Output Numbers
    print("\n[RESULT TABLE: COORDINATE OVERLAP]")
    print(f"Exact Match Sample Ratio: {len(y_dupes)/len(y_binary):.4f}")
    print(f"Label Inconsistency in Exact Matches: {exact_match_variance:.4f}")
    
    print("\n[RESULT TABLE: FEATURE RESOLUTION]")
    print(pd.DataFrame(results_res))
    
    print("\n[RESULT TABLE: REPRESENTATION]")
    print(f"Raw Space Consistency (K=214): {results_res[-1]['Consistency']:.4f}")
    print(f"Latent Space Consistency (PCA=30): {consistency_pca:.4f}")
    
    print("\n[RESULT TABLE: TEMPORAL]")
    print(f"Target Flip Rate (t vs t-1): {target_flip_rate:.4f}")

if __name__ == "__main__":
    numerical_dissection()
