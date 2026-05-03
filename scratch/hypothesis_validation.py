import numpy as np
import pandas as pd
import sys
import os
import gc
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(os.getcwd())
from src.config import Config
from src.utils import load_npy
from src.trainer import Trainer

RUN_ID = "run_20260429_133848"
PROCESSED_PATH = f"outputs/{RUN_ID}/processed"

def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    std_ratio = np.std(y_pred) / (np.std(y_true) + 1e-9)
    p90_val = np.quantile(y_true, 0.90)
    tail_mask = y_true >= p90_val
    tail_mae = mean_absolute_error(y_true[tail_mask], y_pred[tail_mask])
    return mae, std_ratio, tail_mae, np.max(y_pred)

def get_leaf_stats(model, X_val):
    # Predict leaf indices
    leaf_preds = model.predict(X_val, pred_leaf=True)
    
    # Calculate unique leaves visited
    unique_leaves = len(np.unique(leaf_preds))
    
    # Routing entropy
    unique_counts = np.unique(leaf_preds, return_counts=True)[1]
    probs = unique_counts / np.sum(unique_counts)
    entropy = -np.sum(probs * np.log2(probs + 1e-9))
    
    # Extract tree structure for leaf value stats
    tree_df = model.booster_.trees_to_dataframe()
    if 'is_leaf' in tree_df.columns:
        leaf_df = tree_df[tree_df['is_leaf'] == True].copy()
    else:
        leaf_df = tree_df[tree_df['split_feature'].isna()].copy()
    if 'value' not in leaf_df.columns and 'leaf_value' in leaf_df.columns:
        leaf_df['value'] = leaf_df['leaf_value']
    
    max_leaf_log = leaf_df['value'].max()
    max_leaf_exp = np.expm1(max_leaf_log)
    leaf_std = leaf_df['value'].std()
    
    return unique_leaves, entropy, max_leaf_exp, leaf_std

def validate_hypotheses():
    print("\n--- [HYPOTHESIS VALIDATION] ---")
    
    # 1. Load Data
    train_base = pd.read_pickle(f"{PROCESSED_PATH}/train_base.pkl")
    y = load_npy(f"{PROCESSED_PATH}/y_train.npy")
    sid = load_npy(f"{PROCESSED_PATH}/scenario_id.npy", allow_pickle=True)
    
    # Ensure numerical columns only
    X_full = train_base.select_dtypes(include=[np.number]).fillna(0)
    
    # Top 10 features (from previous analysis)
    top10_features = ['avg_trip_distance_rolling_mean_5', 'sku_concentration_rolling_mean_3', 
                      'pack_utilization_rolling_mean_5', 'cold_chain_ratio_rolling_mean_5', 
                      'avg_items_per_order_rolling_mean_5', 'heavy_item_ratio_rolling_mean_5', 
                      'robot_utilization_rolling_mean_5', 'warehouse_temp_avg_rolling_mean_5', 
                      'manual_override_ratio_rolling_mean_5', 'humidity_pct_rolling_mean_5']
    
    top20_features = top10_features + ['pack_utilization_rolling_std_5', 'air_quality_idx_rolling_mean_5', 
                                       'battery_std_rolling_mean_5', 'urgent_order_ratio_rolling_mean_5', 
                                       'day_of_week_rolling_mean_5', 'sku_concentration', 
                                       'robot_idle_rolling_mean_3', 'robot_utilization_rolling_std_5', 
                                       'cold_chain_ratio_rolling_std_5', 'urgent_order_ratio_rolling_std_5']
    
    # Time split (first 80% scenarios vs last 20% scenarios)
    trainer = Trainer(None, y, None, sid, full_df=train_base)
    unique_scenarios = trainer.get_scenario_order(train_base)
    split_idx = int(len(unique_scenarios) * 0.8)
    train_scenarios = unique_scenarios[:split_idx]
    val_scenarios = unique_scenarios[split_idx:]
    
    tr_mask = train_base['scenario_id'].isin(train_scenarios)
    val_mask = train_base['scenario_id'].isin(val_scenarios)
    
    X_tr_time = X_full[tr_mask]
    y_tr_time = y[tr_mask]
    X_val_time = X_full[val_mask]
    y_val_time = y[val_mask]
    
    # IID split
    X_tr_iid, X_val_iid, y_tr_iid, y_val_iid = train_test_split(X_full, y, test_size=0.2, random_state=42)
    
    y_tr_time_log = np.log1p(y_tr_time)
    y_tr_iid_log = np.log1p(y_tr_iid)
    
    print("\n[H1 검증: Feature signal은 존재하지만 모델이 못 쓰는가?]")
    # Train restricted LGBM (Top 10)
    lgbm_h1 = LGBMRegressor(**Config.RAW_LGBM_PARAMS)
    if 'early_stopping_rounds' in lgbm_h1.get_params(): lgbm_h1.set_params(early_stopping_rounds=None)
    lgbm_h1.fit(X_tr_time[top10_features], y_tr_time_log)
    preds_lgbm = np.expm1(lgbm_h1.predict(X_val_time[top10_features]))
    mae_lgbm, std_lgbm, tail_lgbm, max_lgbm = get_metrics(y_val_time, preds_lgbm)
    
    # Train simple model (Ridge)
    ridge_h1 = Ridge(alpha=1.0)
    ridge_h1.fit(X_tr_time[top10_features], y_tr_time_log)
    preds_ridge = np.expm1(ridge_h1.predict(X_val_time[top10_features]))
    mae_ridge, std_ridge, tail_ridge, max_ridge = get_metrics(y_val_time, preds_ridge)
    
    # Train shallow LGBM (max_depth=2)
    lgbm_shallow = LGBMRegressor(**{**Config.RAW_LGBM_PARAMS, 'max_depth': 2, 'num_leaves': 4})
    if 'early_stopping_rounds' in lgbm_shallow.get_params(): lgbm_shallow.set_params(early_stopping_rounds=None)
    lgbm_shallow.fit(X_tr_time[top10_features], y_tr_time_log)
    preds_shallow = np.expm1(lgbm_shallow.predict(X_val_time[top10_features]))
    mae_shallow, std_shallow, tail_shallow, max_shallow = get_metrics(y_val_time, preds_shallow)
    
    print(f"LGBM (Complex) - Tail MAE: {tail_lgbm:.4f} | Max Pred: {max_lgbm:.4f} | Std Ratio: {std_lgbm:.4f}")
    print(f"LGBM (Shallow) - Tail MAE: {tail_shallow:.4f} | Max Pred: {max_shallow:.4f} | Std Ratio: {std_shallow:.4f}")
    print(f"Ridge (Linear) - Tail MAE: {tail_ridge:.4f} | Max Pred: {max_ridge:.4f} | Std Ratio: {std_ridge:.4f}")
    
    h1_true = tail_ridge < tail_lgbm or tail_shallow < tail_lgbm
    print(f"H1 TRUE/FALSE: {'TRUE' if h1_true else 'FALSE'}")
    
    print("\n[H2 검증: Leaf value가 작은 것은 feature 부족 때문이 아니다]")
    # Feature count variation
    feats_list = [('Top 10', top10_features), ('Top 20', top20_features), ('All', X_full.columns.tolist())]
    h2_results = []
    
    for name, feats in feats_list:
        m = LGBMRegressor(**Config.RAW_LGBM_PARAMS)
        if 'early_stopping_rounds' in m.get_params(): m.set_params(early_stopping_rounds=None)
        m.fit(X_tr_time[feats], y_tr_time_log)
        p = np.expm1(m.predict(X_val_time[feats]))
        _, _, max_leaf, leaf_std = get_leaf_stats(m, X_val_time[feats])
        h2_results.append((name, max_leaf, leaf_std, p.max()))
        
    for name, max_leaf, leaf_std, max_p in h2_results:
        print(f"Feats: {name} | Max Leaf(Exp): {max_leaf:.4f} | Leaf Std: {leaf_std:.4f} | Max Pred: {max_p:.4f}")
        
    # Variation across counts
    max_leaf_variation = np.std([x[1] for x in h2_results])
    print(f"Max Leaf Variation across feature counts: {max_leaf_variation:.4f}")
    h2_true = max_leaf_variation < 2.0  # If variation is small, leaf value isn't growing with features
    print(f"H2 TRUE/FALSE: {'TRUE' if h2_true else 'FALSE'}")
    
    print("\n[H3 검증: 현재 구조는 drift 상황에서 split 조건이 붕괴되는 구조다]")
    # IID vs Time Split
    m_iid = LGBMRegressor(**Config.RAW_LGBM_PARAMS)
    if 'early_stopping_rounds' in m_iid.get_params(): m_iid.set_params(early_stopping_rounds=None)
    m_iid.fit(X_tr_iid, y_tr_iid_log)
    
    m_time = LGBMRegressor(**Config.RAW_LGBM_PARAMS)
    if 'early_stopping_rounds' in m_time.get_params(): m_time.set_params(early_stopping_rounds=None)
    m_time.fit(X_tr_time, y_tr_time_log)
    
    # Test on validation sets
    u_iid, ent_iid, _, _ = get_leaf_stats(m_iid, X_val_iid)
    u_time, ent_time, _, _ = get_leaf_stats(m_time, X_val_time)
    
    print(f"IID  Split - Unique Leaves Visited: {u_iid} | Routing Entropy: {ent_iid:.4f}")
    print(f"TIME Split - Unique Leaves Visited: {u_time} | Routing Entropy: {ent_time:.4f}")
    
    h3_true = u_time < u_iid * 0.8 or ent_time < ent_iid * 0.9 # Significant drop
    print(f"H3 TRUE/FALSE: {'TRUE' if h3_true else 'FALSE'}")
    
    # To double check drift filtering impact, we didn't apply drift pruning here (we used all features).
    # The time split already shows the raw structural response.
    
    print("\n=== FINAL HYPOTHESIS SUMMARY ===")
    print(f"H1 (Signal exists but model can't use): {'TRUE' if h1_true else 'FALSE'}")
    print(f"H2 (Leaf value small != feature lack): {'TRUE' if h2_true else 'FALSE'}")
    print(f"H3 (Drift splits breakdown): {'TRUE' if h3_true else 'FALSE'}")
    
if __name__ == "__main__":
    validate_hypotheses()
