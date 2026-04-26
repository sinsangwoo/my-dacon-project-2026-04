import pickle
import json
import pandas as pd
import numpy as np
import os
from lightgbm import LGBMRegressor

run_id = "run_20260426_111910"
base_path = f"outputs/{run_id}"

def analyze_overfitting():
    print("=== TASK 2: Tree Overfitting & Structural Validation ===")
    
    # 1. Feature Scale Analysis
    with open(f"{base_path}/models/reconstructors/features_fold_0.pkl", "rb") as f:
        features_f0 = pickle.load(f)
    
    # Try to load all folds to check variance in selection
    all_fold_features = []
    for i in range(5):
        path = f"{base_path}/models/reconstructors/features_fold_{i}.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                all_fold_features.append(set(pickle.load(f)))
    
    print(f"\n[FEATURE SCALE]")
    print(f"Unique features selected in Fold 0: {len(features_f0)}")
    if len(all_fold_features) > 1:
        common_features = set.intersection(*all_fold_features)
        all_features = set.union(*all_fold_features)
        print(f"Common features across all {len(all_fold_features)} folds: {len(common_features)}")
        print(f"Total unique features across all folds: {len(all_features)}")
        print(f"Selection variance (union - intersection): {len(all_features) - len(common_features)}")
    
    # 2. Split Diversity & Usage
    print(f"\n[SPLIT DIVERSITY & USAGE]")
    # Load model and inspect tree structure
    with open(f"{base_path}/models/lgbm/model_fold_0.pkl", "rb") as f:
        model = pickle.load(f)
    
    booster = model.booster_
    importance = booster.feature_importance(importance_type='gain')
    feature_names = booster.feature_name()
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values('importance', ascending=False)
    
    print(f"Top 10 features by Gain:")
    print(imp_df.head(10))
    
    # Tail analysis
    tail_30_idx = int(len(imp_df) * 0.7)
    tail_30 = imp_df.iloc[tail_30_idx:]
    print(f"\n[NOISE SURVIVAL]")
    print(f"Bottom 30% features importance sum: {tail_30.importance.sum():.4f}")
    print(f"Average importance of bottom 30% features: {tail_30.importance.mean():.4f}")
    print(f"Number of features with 0 importance: {len(imp_df[imp_df.importance == 0])}")
    
    # Tree metrics
    dump = booster.dump_model()
    depths = []
    leaves = []
    for tree in dump['tree_info']:
        depths.append(tree.get('tree_depth', 0))
        leaves.append(tree.get('num_leaves', 0))
    
    print(f"\n[TREE STATS]")
    print(f"Avg Tree Depth: {np.mean(depths):.2f} (Max: {np.max(depths)})")
    print(f"Avg Leaves: {np.mean(leaves):.2f} (Max: {np.max(leaves)})")
    
    # 3. Generalization Gap
    print(f"\n[GENERALIZATION GAP]")
    # I don't have train_mae/valid_mae in a report file, but I can check the logs if I had them.
    # Since I don't have the report, I'll look for any text logs.
    # For now, I'll check pruning_manifest.json
    with open(f"{base_path}/processed/pruning_manifest.json", "r") as f:
        pruning = json.load(f)
    print(f"Pruning Manifest - Total dropped: {len(pruning.get('dropped_features', []))}")

if __name__ == "__main__":
    analyze_overfitting()
