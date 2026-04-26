import pickle
import numpy as np
import pandas as pd
import json
import os
import sys

# Config
run_id = "run_20260426_111910"
base_path = f"outputs/{run_id}"

def structural_overfitting_audit():
    print("=== STRUCTURAL OVERFITTING AUDIT ===")
    
    # Load model
    with open(f"{base_path}/models/lgbm/model_fold_0.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{base_path}/models/reconstructors/features_fold_0.pkl", "rb") as f:
        features = pickle.load(f)
    
    booster = model.booster_
    dump = booster.dump_model()
    
    # 1. Interaction Split Analysis
    print("\n--- Task 1: Interaction Split Analysis ---")
    total_splits = 0
    interaction_splits = 0
    interaction_depths = []
    raw_depths = []
    
    def traverse(node, depth):
        nonlocal total_splits, interaction_splits
        if 'split_feature' in node:
            total_splits += 1
            feat_idx = node['split_feature']
            feat_name = features[feat_idx]
            
            is_interaction = any(x in feat_name for x in ['_x_', 'embed', 'trend_proxy', 'weighted_mean'])
            if is_interaction:
                interaction_splits += 1
                interaction_depths.append(depth)
            else:
                raw_depths.append(depth)
            
            traverse(node['left_child'], depth + 1)
            traverse(node['right_child'], depth + 1)

    for tree in dump['tree_info']:
        traverse(tree['tree_structure'], 0)
    
    print(f"Total Splits: {total_splits}")
    print(f"Interaction Splits: {interaction_splits} ({interaction_splits/total_splits:.2%})")
    print(f"Avg Interaction Depth: {np.mean(interaction_depths):.2f}")
    print(f"Avg Raw/TS Depth: {np.mean(raw_depths):.2f}")

    # 2. Rare Pattern Analysis (Sample Count per Leaf)
    print("\n--- Task 3: Rare Pattern Analysis ---")
    leaf_samples = []
    leaf_values = []
    
    def get_leaf_stats(node):
        if 'leaf_value' in node:
            leaf_samples.append(node.get('leaf_count', 0))
            leaf_values.append(node.get('leaf_value', 0))
        else:
            get_leaf_stats(node['left_child'])
            get_leaf_stats(node['right_child'])

    for tree in dump['tree_info']:
        get_leaf_stats(tree['tree_structure'])
    
    ls = pd.Series(leaf_samples)
    print(f"Leaf Sample Stats:\n{ls.describe()}")
    print(f"Leaves with < 20 samples: {len(ls[ls < 20])} ({len(ls[ls < 20])/len(ls):.2%})")
    print(f"Leaves with < 50 samples: {len(ls[ls < 50])} ({len(ls[ls < 50])/len(ls):.2%})")

    # 3. Feature Importance Dispersion (Redundancy check)
    print("\n--- Task 2: Feature Importance Dispersion ---")
    gain = booster.feature_importance(importance_type='gain')
    imp_df = pd.DataFrame({'feature': features, 'gain': gain}).sort_values('gain', ascending=False)
    
    # Identify "Same Signal" groups (e.g. rolling_mean_3 vs rolling_mean_5)
    imp_df['base_feature'] = imp_df['feature'].str.split('_rolling|_slope|_rate|_diff|_rel|_is_boundary|_x_').str[0]
    base_agg = imp_df.groupby('base_feature').agg({'gain': ['count', 'sum', 'max']}).sort_values(('gain', 'sum'), ascending=False)
    
    print(f"Base Feature Redundancy (Top 10):")
    print(base_agg.head(10))

if __name__ == "__main__":
    structural_overfitting_audit()
