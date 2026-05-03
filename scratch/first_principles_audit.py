import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold

# Add src to path
sys.path.append(os.getcwd())
from src.config import Config
from src.data_loader import add_time_series_features

def run_first_principles_audit_v2():
    RUN_ID = "run_20260430_231842"
    BASE_PATH = f"./outputs/{RUN_ID}"
    Config.rebuild_paths(RUN_ID)
    
    print(f"--- [FIRST-PRINCIPLES AUDIT V2: {RUN_ID}] ---")
    
    # Load Data
    y_all = np.load(f"{BASE_PATH}/processed/y_train.npy").astype(np.float64)
    scenarios = np.load(f"{BASE_PATH}/processed/scenario_id.npy", allow_pickle=True)
    with open(f"{BASE_PATH}/processed/train_base.pkl", "rb") as f:
        X_base_df = pickle.load(f)
    if Config.TARGET in X_base_df.columns:
        X_base_df = X_base_df.drop(columns=[Config.TARGET])
        
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(np.arange(len(y_all)), y_all, groups=scenarios))
    tr_idx, val_idx = splits[0]
    
    X_tr = add_time_series_features(X_base_df.iloc[tr_idx]).select_dtypes(include=[np.number])
    y_tr = y_all[tr_idx]
    y_tr_log = np.log1p(y_tr)
    
    q90 = np.quantile(y_tr, 0.90)
    y_tr_bin = (y_tr >= q90).astype(int)

    # 1. Information Loss Audit (Raw vs Log)
    print("\n[1. INFORMATION LOSS AUDIT]")
    # Use top 3 to avoid time explosion
    corrs = X_tr.apply(lambda x: np.abs(np.corrcoef(x, y_tr)[0, 1]) if np.std(x) > 0 else 0)
    top_cols = corrs.sort_values(ascending=False).head(3).index.tolist()
    
    for col in top_cols:
        mi_raw = mutual_info_regression(X_tr[[col]], y_tr, random_state=42)[0]
        mi_log = mutual_info_regression(X_tr[[col]], y_tr_log, random_state=42)[0]
        print(f"  Feature: {col:30} | MI(Raw): {mi_raw:.4f} | MI(Log): {mi_log:.4f} | Loss: {1 - mi_log/mi_raw:.2%}")

    # 2. Scenario Overlap Audit
    print("\n[2. SCENARIO OVERLAP AUDIT]")
    # Measure similarity between scenarios in Train and Validation
    # Select only numeric columns for mean
    numeric_df = X_base_df.select_dtypes(include=[np.number])
    scen_means = numeric_df.groupby(scenarios).mean()
    
    unique_scens = np.unique(scenarios)
    tr_scens = np.unique(scenarios[tr_idx])
    val_scens = np.unique(scenarios[val_idx])
    
    # Cosine similarity on mean scenario vectors
    # Handle possible NaN in means
    scen_profiles = scen_means.loc[unique_scens].fillna(0)
    sim_matrix = cosine_similarity(scen_profiles)
    
    cross_sims = []
    for vs in val_scens:
        v_idx = list(unique_scens).index(vs)
        t_indices = [list(unique_scens).index(ts) for ts in tr_scens]
        cross_sims.append(np.mean(sim_matrix[v_idx, t_indices]))
    
    print(f"  Mean Scenario Similarity (Train-Val): {np.mean(cross_sims):.4f}")
    print(f"  Max Cross-Similarity: {np.max(cross_sims):.4f}")

    # 3. Target Distribution Complexity
    print("\n[3. TARGET DISTRIBUTION COMPLEXITY]")
    print(f"  Overall Tail Ratio (Q90+): {np.mean(y_tr_bin):.2%}")
    # Per-scenario tail ratio in training
    scen_tr_ids = scenarios[tr_idx]
    tr_scen_tail_ratios = []
    for ts in tr_scens:
        mask = scen_tr_ids == ts
        tr_scen_tail_ratios.append(np.mean(y_tr_bin[mask]))
    
    print(f"  Scenario-wise Tail Ratio Variance: {np.var(tr_scen_tail_ratios):.6f}")
    print(f"  Scenario-wise Tail Ratio Range: [{np.min(tr_scen_tail_ratios):.4f}, {np.max(tr_scen_tail_ratios):.4f}]")

if __name__ == "__main__":
    run_first_principles_audit_v2()
