import os
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import sys

# Add src to path
sys.path.append('.')
from src.data_loader import apply_latent_features
from src.config import Config

# Config
run_id = "run_20260426_111910"
base_path = f"outputs/{run_id}"

def validate_compression_safety_v2():
    print("=== TASK A~C: Precision Compression Validation ===")
    
    # 1. Load Data
    print("Loading data...")
    with open(f"{base_path}/processed/train_base.pkl", "rb") as f:
        X_base_all = pickle.load(f)
    y_all = np.load(f"{base_path}/processed/y_train.npy")
    
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_base_all), 30000, replace=False)
    X_base = X_base_all.iloc[sample_indices].reset_index(drop=True)
    y = y_all[sample_indices]
    
    with open(f"{base_path}/models/reconstructors/features_fold_0.pkl", "rb") as f:
        full_features = pickle.load(f)
    with open(f"{base_path}/models/reconstructors/recon_fold_0.pkl", "rb") as f:
        recon = pickle.load(f)
    with open(f"{base_path}/models/reconstructors/scaler_fold_0.pkl", "rb") as f:
        scaler = pickle.load(f)
        
    X_full = apply_latent_features(X_base, recon, scaler=scaler, selected_features=full_features, is_train=True)
    X = X_full[full_features]
    
    # 2. TASK C: Redefine "Same Signal" (Multi-criteria)
    # [WHY_THIS_DEFINITION_IS_SAFE] Combines lineage with statistical redundancy (corr + co-occurrence)
    print("\n--- Task C: Redefining Same Signal ---")
    corr_matrix = X.corr().abs()
    
    # Placeholder for co-occurrence (needs path tracing, simplified here to path-sharing score)
    # We'll use Lineage + Correlation as primary, and Gain similarity as secondary.
    def get_base(f):
        return f.split('_rolling|_slope|_rate|_diff|_rel|_is_boundary|_x_')[0]

    # 3. TASK B: Residual Signal Protection (Strict Depth)
    print("--- Task B: Detecting Residual Signals (Depth > 8) ---")
    with open(f"{base_path}/models/lgbm/model_fold_0.pkl", "rb") as f:
        model = pickle.load(f)
    booster = model.booster_
    gain = booster.feature_importance(importance_type='gain')
    
    dump = booster.dump_model()
    feat_depths = {f: [] for f in full_features}
    def traverse(node, depth):
        if 'split_feature' in node:
            feat_name = full_features[node['split_feature']]
            feat_depths[feat_name].append(depth)
            traverse(node['left_child'], depth + 1)
            traverse(node['right_child'], depth + 1)
    for tree in dump['tree_info']:
        traverse(tree['tree_structure'], 0)
    
    avg_depths = {f: (np.mean(d) if d else 0) for f, d in feat_depths.items()}
    
    # Residual signals: Found mostly at depth > 8, low gain, but consistently used
    res_signals = [f for f in full_features if avg_depths[f] > 8 and gain[full_features.index(f)] < np.percentile(gain, 25)]
    print(f"Immune Features (Residual): {len(res_signals)}")

    # 4. TASK A: Elite Selection (Top 2 per base + Residuals)
    print("\n--- Task A: Compression Ablation ---")
    feat_info = pd.DataFrame({
        'feature': full_features,
        'gain': gain,
        'base': [get_base(f) for f in full_features]
    })
    
    pruned_features = []
    for base, group in feat_info.groupby('base'):
        if len(group) <= 2:
            continue
        # Candidates for pruning: Not in Top 2 gain, Not a Residual Signal
        top_2 = group.sort_values('gain', ascending=False).head(2)['feature'].tolist()
        for f in group['feature']:
            if f not in top_2 and f not in res_signals:
                # [TASK C Check] Only prune if it has a high-corr sibling in Top 2
                siblings_in_top_2 = [s for s in top_2 if s in corr_matrix.columns and f in corr_matrix.index and corr_matrix.loc[f, s] > 0.9]
                if siblings_in_top_2:
                    pruned_features.append(f)
                    
    elite_features = [f for f in full_features if f not in pruned_features]
    print(f"Compressed Feature Count: {len(elite_features)} (Removed: {len(pruned_features)})")
    
    # 5. Run CV
    def run_cv(features_to_use, name):
        print(f"Testing {name}...")
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        maes = []
        for tr_idx, val_idx in kf.split(X, y):
            X_tr, X_val = X.iloc[tr_idx][features_to_use], X.iloc[val_idx][features_to_use]
            y_tr, y_val = y[tr_idx], y[val_idx]
            m = LGBMRegressor(n_estimators=100, num_leaves=31, random_state=42, verbose=-1)
            m.fit(X_tr, y_tr)
            maes.append(mean_absolute_error(y_val, m.predict(X_val)))
        return np.mean(maes)

    mae_full = run_cv(full_features, "Full (405)")
    mae_elite = run_cv(elite_features, f"Elite ({len(elite_features)})")
    
    print(f"\nMAE Baseline: {mae_full:.4f}")
    print(f"MAE Elite:    {mae_elite:.4f}")
    print(f"Gap:          {mae_elite - mae_full:.4f}")
    
    if mae_elite <= mae_full + 0.003:
        print("VERDICT: COMPRESSION PROVEN SAFE (SIGNAL PRESERVED)")
    else:
        print("VERDICT: COMPRESSION FAILED (SIGNAL LOSS DETECTED)")

if __name__ == "__main__":
    validate_compression_safety_v2()
