import pandas as pd
import numpy as np
import logging
import sys
import os
from sklearn.decomposition import PCA

# Add src to path
sys.path.append(os.getcwd())

from src.config import Config
from src.utils import DriftShieldScaler
from src.data_loader import load_data, build_base_features
from src.schema import BASE_COLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PCA_FINAL_RECON")

def rebuild_pca_set():
    logger.info("Rebuilding PCA Feature Set...")
    
    # 1. Load Data
    train, _ = load_data()
    train_subset = train.head(50000).copy()
    train_full = build_base_features(train_subset)
    
    # 2. Audit Raw Features (Step 1)
    bad_raw = ['task_reassign_15m'] # Identified previously
    good_raw = [f for f in BASE_COLS if f not in bad_raw]
    
    # 3. Audit Derivatives (Step 2)
    # Derivative suffixes
    suffixes = ["_rolling_mean_5", "_diff_1"]
    derivative_candidates = []
    for f in good_raw:
        for s in suffixes:
            derivative_candidates.append(f"{f}{s}")
            
    # 4. Filter Derivatives by quality
    scaler = DriftShieldScaler()
    all_candidates = good_raw + derivative_candidates
    scaler.fit(train_full, all_candidates)
    df_drifted = scaler.transform(train_full.copy(), all_candidates)
    
    report = []
    for col in all_candidates:
        s = scaler.stats[col]
        ratio = np.std(df_drifted[col]) / (s['std'] + 1e-9)
        variance = np.var(df_drifted[col])
        report.append({"feature": col, "variance": variance, "std_ratio": ratio, "is_raw": col in good_raw})
        
    audit_df = pd.DataFrame(report)
    good_mask = (audit_df['variance'] > 1e-6) & (audit_df['std_ratio'] >= 0.6)
    clean_features = audit_df[good_mask]['feature'].tolist()
    
    # 5. Select Subset to reach 0.8
    # We prioritize raw, then derivatives.
    final_raw = [f for f in clean_features if f in good_raw]
    final_derivatives = [f for f in clean_features if f not in good_raw]
    
    # Try adding derivatives until we hit 0.8
    selected_derivatives = []
    
    # Sort derivatives by loading-contribution to the current raw PCA?
    # Or just add them in order of variance.
    
    current_set = list(final_raw)
    
    def get_pca_var(subset):
        X = df_drifted[subset].fillna(0).values
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        pca = PCA(n_components=8)
        pca.fit(X)
        return np.sum(pca.explained_variance_ratio_)

    base_var = get_pca_var(current_set)
    logger.info(f"Base Variance (29 raw): {base_var:.4f}")
    
    # Add derivatives one by one (greedy improvement)
    # Actually, let's just add all clean derivatives and see.
    all_clean_var = get_pca_var(current_set + final_derivatives)
    logger.info(f"Variance with ALL clean derivatives: {all_clean_var:.4f}")
    
    # If all clean derivatives is > 0.8, we can use a smaller set if we want.
    # But adding more redundant features is fine for PCA variance ratio.
    
    if all_clean_var >= 0.8:
        # We'll take top derivatives to keep the set manageable
        # Heuristic: top derivatives by variance
        final_derivatives_sorted = audit_df[audit_df['feature'].isin(final_derivatives)].sort_values("variance", ascending=False)['feature'].tolist()
        
        subset_derivatives = []
        for d in final_derivatives_sorted:
            subset_derivatives.append(d)
            if get_pca_var(current_set + subset_derivatives) >= 0.8:
                break
        
        final_set = current_set + subset_derivatives
        logger.info(f"FINAL SET ({len(final_set)}): {final_set}")
        logger.info(f"FINAL VARIANCE: {get_pca_var(final_set):.4f}")
        return final_set
    else:
        logger.error("FAILED to reach 0.8 even with all derivatives.")
        return None

if __name__ == "__main__":
    rebuild_pca_set()
