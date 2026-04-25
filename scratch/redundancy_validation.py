import pandas as pd
import numpy as np
import logging
import os
from src.config import Config
from src.data_loader import build_base_features, load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_redundancy():
    Config.SMOKE_TEST = True
    train_raw = load_data()[0]
    
    # SAMPLE FIRST - VERY SMALL
    train_raw = train_raw.head(2000).copy()
    
    # Process through pipeline
    train_base, manifest, registry = build_base_features(train_raw)
    
    # Compute correlation
    cols = [c for c in train_base.columns if c not in Config.ID_COLS and c != Config.TARGET]
    corr = train_base[cols].corr().abs()
    
    # Identify pairs > 0.90
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    redundant_pairs = []
    for col in upper.columns:
        high_corr = upper[col][upper[col] > 0.90]
        for row, val in high_corr.items():
            redundant_pairs.append((row, col, val))
            
    redundant_df = pd.DataFrame(redundant_pairs, columns=['f1', 'f2', 'corr']).sort_values('corr', ascending=False)
    
    print("\n--- REDUNDANCY VALIDATION REPORT ---")
    print(f"Total Features: {len(cols)}")
    print(f"Redundant Pairs (>0.90): {len(redundant_df)}")
    
    if not redundant_df.empty:
        print("\nTop 20 Redundant Pairs:")
        print(redundant_df.head(20))
        
        # Analyze clusters
        from sklearn.cluster import AgglomerativeClustering
        dist = 1 - corr.fillna(0)
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.1, metric='precomputed', linkage='complete')
        clusters = clustering.fit_predict(dist)
        
        cluster_map = {}
        for f, c in zip(cols, clusters):
            cluster_map.setdefault(c, []).append(f)
            
        print("\nFeature Clusters (Distance < 0.1):")
        for c, members in sorted(cluster_map.items(), key=lambda x: len(x[1]), reverse=True):
            if len(members) > 1:
                print(f"Cluster {c} ({len(members)} features): {members[:10]} ...")
                
if __name__ == "__main__":
    validate_redundancy()
