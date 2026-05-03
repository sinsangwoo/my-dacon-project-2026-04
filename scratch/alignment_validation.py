import numpy as np
import pandas as pd
import sys
import os
import gc
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.getcwd())
from src.config import Config
from src.utils import load_npy
from src.trainer import Trainer

RUN_ID = "run_20260429_133848"
PROCESSED_PATH = f"outputs/{RUN_ID}/processed"

def get_tree_contributions(model, X):
    n_trees = model.booster_.num_trees()
    preds = np.zeros((len(X), n_trees))
    
    # Base margin (initial prediction)
    # Actually, lightgbm predict with num_iteration=1 gives base + tree 1.
    # To get individual tree contributions, we can take differences.
    
    # Getting cumulative predictions
    cum_preds = []
    for k in range(1, n_trees + 1):
        cum_preds.append(model.predict(X, num_iteration=k))
        
    cum_preds = np.array(cum_preds).T # Shape: (n_samples, n_trees)
    
    # Contributions
    contribs = np.zeros_like(cum_preds)
    contribs[:, 0] = cum_preds[:, 0] # Base + Tree 0
    contribs[:, 1:] = cum_preds[:, 1:] - cum_preds[:, :-1]
    
    # Adjust base margin. The base is mean(y) or determined by objective.
    # Usually the first tree handles the base score, but let's just look at the variance from tree 1 onwards.
    # For alignment, the base score is a constant shift. The direction of trees is what matters.
    # We will ignore the first term if it contains the massive base score, or subtract it.
    base_score = contribs[:, 0].mean()
    contribs[:, 0] -= base_score
    
    return contribs, cum_preds

def alignment_audit():
    print("\n--- [TREE ALIGNMENT VALIDATION] ---")
    
    # Load Data
    train_base = pd.read_pickle(f"{PROCESSED_PATH}/train_base.pkl")
    y = load_npy(f"{PROCESSED_PATH}/y_train.npy")
    sid = load_npy(f"{PROCESSED_PATH}/scenario_id.npy", allow_pickle=True)
    
    X_full = train_base.select_dtypes(include=[np.number]).fillna(0)
    
    # Splits
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
    
    X_tr_iid, X_val_iid, y_tr_iid, y_val_iid = train_test_split(X_full, y, test_size=0.2, random_state=42)
    
    y_tr_time_log = np.log1p(y_tr_time)
    y_tr_iid_log = np.log1p(y_tr_iid)
    
    print("Training IID Model...")
    params = Config.RAW_LGBM_PARAMS.copy()
    params['n_estimators'] = 500
    m_iid = LGBMRegressor(**params) # Limit to 500 for speed
    if 'early_stopping_rounds' in m_iid.get_params(): m_iid.set_params(early_stopping_rounds=None)
    m_iid.fit(X_tr_iid, y_tr_iid_log)
    
    print("Training Time-Split Model...")
    params['n_estimators'] = 500
    m_time = LGBMRegressor(**params)
    if 'early_stopping_rounds' in m_time.get_params(): m_time.set_params(early_stopping_rounds=None)
    m_time.fit(X_tr_time, y_tr_time_log)
    
    print("\n--- [EXP 1: Tree Contribution Sign Alignment] ---")
    c_iid, cum_iid = get_tree_contributions(m_iid, X_val_iid)
    c_time, cum_time = get_tree_contributions(m_time, X_val_time)
    
    def calc_alignment(contribs):
        # alignment_score = |sum(contribs)| / sum(|contribs|)
        sum_c = np.abs(np.sum(contribs, axis=1))
        sum_abs_c = np.sum(np.abs(contribs), axis=1) + 1e-9
        return np.mean(sum_c / sum_abs_c)
        
    align_iid = calc_alignment(c_iid)
    align_time = calc_alignment(c_time)
    
    p90_time = np.quantile(y_val_time, 0.90)
    tail_mask_time = y_val_time >= p90_time
    align_time_tail = calc_alignment(c_time[tail_mask_time])
    
    print(f"IID Mean Alignment Score: {align_iid:.4f}")
    print(f"Time-Split Mean Alignment Score: {align_time:.4f}")
    print(f"Time-Split TAIL Alignment Score: {align_time_tail:.4f}")
    
    print("\n--- [EXP 2: Tree Direction Consistency] ---")
    # Evaluate m_time on its own train set
    c_time_tr, _ = get_tree_contributions(m_time, X_tr_time[:10000]) # sample for speed
    c_time_val = c_time[:10000] # already computed
    
    sign_tr = np.sign(np.mean(c_time_tr, axis=0))
    sign_val = np.sign(np.mean(c_time_val, axis=0))
    
    consistency = np.mean(sign_tr == sign_val)
    print(f"Sign Consistency Ratio (Train vs Test): {consistency:.4f}")
    
    print("\n--- [EXP 3: Cumulative Prediction Growth] ---")
    import sys
    
    k_points = [10, 50, 100, 200, 500]
    print(f"{'Trees':<10} | {'IID Mean':<12} | {'IID Std':<12} | {'Time Mean':<12} | {'Time Std':<12}")
    for k in k_points:
        if k > cum_iid.shape[1]: continue
        pred_k_iid = cum_iid[:, k-1]
        pred_k_time = cum_time[:, k-1]
        
        print(f"{k:<10} | {np.mean(pred_k_iid):<12.4f} | {np.std(pred_k_iid):<12.4f} | {np.mean(pred_k_time):<12.4f} | {np.std(pred_k_time):<12.4f}")

    # Final Verdict
    print("\n--- FINAL VERDICT ---")
    if align_time < align_iid * 0.8 or consistency < 0.8 or np.std(cum_time[:, -1]) < np.std(cum_time[:, 100]):
        print("ALIGNMENT COLLAPSE = TRUE")
    else:
        print("ALIGNMENT COLLAPSE = FALSE")

if __name__ == "__main__":
    alignment_audit()
