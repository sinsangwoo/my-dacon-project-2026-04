import pickle
import numpy as np
import pandas as pd
import json

with open('outputs/run_20260426_111910/models/lgbm/model_fold_0.pkl', 'rb') as f:
    model = pickle.load(f)
with open('outputs/run_20260426_111910/models/reconstructors/features_fold_0.pkl', 'rb') as f:
    feature_names = pickle.load(f)

tree_info = model.booster_.dump_model()
trees = tree_info['tree_info']

feature_stats = {f: {'splits': 0, 'gain': 0.0, 'depths': []} for f in feature_names}
total_leaves = 0
tree_depths = []
max_leaves_per_tree = 0
unique_features_used = set()

def traverse_tree(node, depth):
    if 'split_feature' in node:
        feat_idx = node['split_feature']
        feat_name = feature_names[feat_idx]
        gain = node.get('split_gain', 0.0)
        
        feature_stats[feat_name]['splits'] += 1
        feature_stats[feat_name]['gain'] += gain
        feature_stats[feat_name]['depths'].append(depth)
        unique_features_used.add(feat_name)
        
        traverse_tree(node['left_child'], depth + 1)
        traverse_tree(node['right_child'], depth + 1)
    elif 'leaf_value' in node:
        global total_leaves
        total_leaves += 1

for tree in trees:
    traverse_tree(tree['tree_structure'], 0)
    tree_depths.append(tree.get('tree_depth', 0))
    max_leaves_per_tree = max(max_leaves_per_tree, tree.get('num_leaves', 0))

total_gain = sum(s['gain'] for s in feature_stats.values())

print("=== Task 1 & 2: Feature Group Analysis ===")
groups = {'rolling_std': 0.0, 'rate': 0.0, 'slope': 0.0, 'interaction': 0.0, 'conditional': 0.0}

for f, stats in feature_stats.items():
    if '_rolling_std' in f: groups['rolling_std'] += stats['gain']
    if '_rate_' in f: groups['rate'] += stats['gain']
    if '_slope_' in f: groups['slope'] += stats['gain']
    if '_x_' in f: groups['interaction'] += stats['gain']
    if any(x in f for x in ['flag', 'boundary', 'extreme', 'is_']): groups['conditional'] += stats['gain']

for g, gain in groups.items():
    pct = (gain / total_gain) * 100 if total_gain > 0 else 0
    print(f"Group {g}: Gain % = {pct:.2f}%")

print("\n=== Task 6: Model Capacity ===")
print(f"Total Trees: {len(trees)}")
print(f"Avg Leaves / Tree: {total_leaves / len(trees):.1f}")
print(f"Max Leaves Configured / Used: 31 / {max_leaves_per_tree}")
print(f"Unique Features Used: {len(unique_features_used)} / {len(feature_names)}")

def print_depth_stats(pattern):
    depths = []
    for f, stats in feature_stats.items():
        if pattern in f: depths.extend(stats['depths'])
    if depths:
        print(f"Depth for {pattern}: Avg {np.mean(depths):.2f}, Min {np.min(depths)}, Max {np.max(depths)}")
    else:
        print(f"Depth for {pattern}: Not used")

print_depth_stats('_rolling_std')
print_depth_stats('_rate_')
print_depth_stats('_slope_')
print_depth_stats('_x_')
print_depth_stats('flag')

