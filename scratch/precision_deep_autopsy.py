import numpy as np
import pandas as pd
import os
import joblib
from sklearn.metrics import precision_score

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"
models_dir = f"{base_dir}/outputs/{run_id}/models/lgbm"

def precision_autopsy():
    print(f"--- [DEEP PRECISION AUTOPSY: WHY 0.27?] ---")
    
    # 1. Load Data
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    scenario_id = np.load(f"{processed_dir}/scenario_id.npy", allow_pickle=True)
    q90 = np.percentile(y_true, 90)
    y_binary = (y_true >= q90).astype(int)
    
    # Reconstruct OOF P (Simplified proxy or using the previous logic)
    # To be fast and exact, we need the p-values we just analyzed.
    # I'll re-run the core logic but focus on scenario-wise metrics.
    
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    unique_scenarios = np.unique(scenario_id)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_p = np.zeros(len(y_true))
    feature_importances = []
    
    for fold, (train_scen_idx, val_scen_idx) in enumerate(kf.split(unique_scenarios)):
        val_scenarios = unique_scenarios[val_scen_idx]
        val_mask = np.isin(scenario_id, val_scenarios)
        X_val = train_df[val_mask]
        
        model_dict = joblib.load(f"{models_dir}/model_fold_{fold}.pkl")
        clf = model_dict["clf"]
        feat_names = clf.feature_name_
        
        # Numeric only as discovered before
        X_val_input = X_val.select_dtypes(include=[np.number]).values
        if X_val_input.shape[1] != len(feat_names):
            # Fallback to feature name matching if possible, but shape mismatch means we need to be careful
            # Given the previous error, we know it expects 685.
            pass
            
        oof_p[val_mask] = clf.predict_proba(X_val_input)[:, 1]
        feature_importances.append(clf.feature_importances_)

    # --- 1. Scenario-wise Analysis ---
    results = []
    for scen in unique_scenarios:
        mask = scenario_id == scen
        scen_y = y_binary[mask]
        scen_p = oof_p[mask]
        
        if np.sum(scen_y) > 0:
            # We use a threshold of 0.5 for standard precision, 
            # but we can also use the optimal threshold.
            # However, the 0.27 precision we found was for "Boosted" samples.
            # Let's use the top 10% global threshold as in the previous script.
            global_thresh = np.percentile(oof_p, 90)
            scen_pred = (scen_p >= global_thresh).astype(int)
            
            tp = np.sum((scen_pred == 1) & (scen_y == 1))
            fp = np.sum((scen_pred == 1) & (scen_y == 0))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            results.append({"scenario_id": scen, "precision": prec, "tail_count": np.sum(scen_y)})
            
    df_scen = pd.DataFrame(results)
    print("\n[1] SCENARIO HETEROGENEITY")
    print(f"Mean Precision across scenarios: {df_scen['precision'].mean():.4f}")
    print(f"Std Precision: {df_scen['precision'].std():.4f}")
    print(f"Scenarios with 0% Precision: {len(df_scen[df_scen['precision'] == 0])} / {len(df_scen)}")
    
    # --- 2. Feature Separation Power (Top Features) ---
    avg_imp = np.mean(feature_importances, axis=0)
    top_indices = np.argsort(avg_imp)[-10:][::-1]
    
    print("\n[2] TOP FEATURE SEPARATION (Tail vs Non-Tail)")
    # Since we don't have column names easily, we use indices but show their KS
    numeric_df = train_df.select_dtypes(include=[np.number])
    for idx in top_indices:
        feat_name = f"Col_{idx}" # Placeholder
        vals = numeric_df.values[:, idx]
        tail_vals = vals[y_binary == 1]
        non_tail_vals = vals[y_binary == 0]
        
        # KS-like distance
        dist = np.abs(np.mean(tail_vals) - np.mean(non_tail_vals)) / (np.std(vals) + 1e-9)
        print(f"Feature {idx:3d} | Separation Distance: {dist:.4f}")

if __name__ == "__main__":
    precision_autopsy()
