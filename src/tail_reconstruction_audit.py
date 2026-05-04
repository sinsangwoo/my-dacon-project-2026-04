import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import KFold
from scipy.stats import ks_2samp
import warnings
from . import utils
from .utils import downcast_df
warnings.filterwarnings('ignore')

def compute_risk_controller_stats(run_id):
    print("\n" + "="*50)
    print(f" 2. RISK CONTROLLER OVERLOAD VALIDATION [{run_id}]")
    print("="*50)
    
    fold_dir = f"outputs/{run_id}/models/reconstructors"
    all_gaps = []
    all_weights = []
    all_p = []
    damping_factors = []
    
    for fold in range(5):
        path = f"{fold_dir}/forensic_fold_{fold}.json"
        if not os.path.exists(path): continue
        with open(path, "r") as f: data = json.load(f)
        
        gaps = np.array(data["gap"])
        p = np.array(data["p_val"])
        weights = np.array(data["final_weight"])
        
        # Calculate expected sigmoid weights
        k, theta = 8.0, 0.55 # Config values
        expected_w = 1.0 / (1.0 + np.exp(-k * (p - theta)))
        
        # Damping factor applied = actual_weight / expected_weight
        # To avoid div by zero, add epsilon
        damping = weights / (expected_w + 1e-9)
        # Filter to only where damping was actually applied (< 0.99)
        applied_mask = damping < 0.99
        if np.any(applied_mask):
            damping_factors.extend(damping[applied_mask])
            
        all_gaps.extend(gaps)
        all_p.extend(p)
        
    all_gaps = np.array(all_gaps)
    damping_factors = np.array(damping_factors)
    
    activation_rate = len(damping_factors) / len(all_gaps) * 100 if len(all_gaps) > 0 else 0
    avg_damping = np.mean(damping_factors) if len(damping_factors) > 0 else 1.0
    high_risk_ratio = np.mean(all_gaps > 3.0) * 100 # GAP_THRESHOLD = 3.0
    
    print(f"Activation Rate:      {activation_rate:.2f}% (> 70% = Collapse)")
    print(f"Avg Damping Factor:   {avg_damping:.4f}")
    print(f"High-Risk Ratio (Gap>3): {high_risk_ratio:.2f}%")
    print(f"Pred Gap P90:         {np.percentile(all_gaps, 90):.4f}")
    
    if activation_rate > 50:
        print("Verdict: OVERLOADED - Controller is acting as core system, not fail-safe.")

def evaluate_models(X, y):
    print("\n" + "="*50)
    print(" 4. CLASSIFIER vs RANKER (Regressor) COMPARISON")
    print("="*50)
    
    q90_val = np.percentile(y, 90)
    y_bin = (y >= q90_val).astype(int)
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    # Model 1: Binary Classifier (Current config)
    clf = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, scale_pos_weight=2.0, random_state=42, verbose=-1, n_jobs=-1)
    
    # Model 2: Regressor (Ranking approach)
    reg = lgb.LGBMRegressor(n_estimators=100, num_leaves=31, objective='regression_l1', random_state=42, verbose=-1, n_jobs=-1)
    
    clf_prec, reg_prec = [], []
    clf_rec, reg_rec = [], []
    
    for tr_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        y_bin_tr, y_bin_val = y_bin[tr_idx], y_bin[val_idx]
        
        # Classifier
        utils.SAFE_FIT(clf, X_tr.values.astype(np.float32), y_bin_tr.astype(np.float32))
        p_val_clf = utils.SAFE_PREDICT_PROBA(clf, X_val.values.astype(np.float32))[:, 1]
        mask_clf = p_val_clf > 0.7
        clf_prec.append(np.mean(y_bin_val[mask_clf]) if np.any(mask_clf) else 0)
        clf_rec.append(np.sum(y_bin_val[mask_clf]) / np.sum(y_bin_val) if np.sum(y_bin_val)>0 else 0)
        
        # Regressor (Ranker)
        utils.SAFE_FIT(reg, X_tr.values.astype(np.float32), np.log1p(y_tr).astype(np.float32))
        p_val_reg = np.expm1(utils.SAFE_PREDICT(reg, X_val.values.astype(np.float32)))
        # Threshold for ranker: predict tail if prediction > q90_val * 0.8
        # We find threshold that matches recall of clf for fair precision comparison
        target_rec = clf_rec[-1]
        sorted_preds = np.sort(p_val_reg)[::-1] # descending
        # Select top N to match recall roughly
        top_n = max(1, int(target_rec * np.sum(y_bin_val)))
        # Actually, let's just use a fixed percentile of predictions
        thresh_reg = np.percentile(p_val_reg, 90) # top 10%
        mask_reg = p_val_reg > thresh_reg
        reg_prec.append(np.mean(y_bin_val[mask_reg]) if np.any(mask_reg) else 0)
        reg_rec.append(np.sum(y_bin_val[mask_reg]) / np.sum(y_bin_val) if np.sum(y_bin_val)>0 else 0)

    print(f"{'Model':<15} | {'Prec@HighConf':<15} | {'Recall':<10}")
    print("-" * 45)
    print(f"{'Binary Clf':<15} | {np.mean(clf_prec):.4f}         | {np.mean(clf_rec):.4f}")
    print(f"{'Regressor':<15} | {np.mean(reg_prec):.4f}         | {np.mean(reg_rec):.4f}")

def tail_feature_audit(X, y):
    print("\n" + "="*50)
    print(" 5. TAIL FEATURE FORENSIC AUDIT (Permutation Importance on Tail)")
    print("="*50)
    
    q90_val = np.percentile(y, 90)
    tail_mask = y >= q90_val
    
    X_tail = X[tail_mask]
    y_tail = y[tail_mask]
    
    # Train a fast regressor on full data to see what it learns
    model = lgb.LGBMRegressor(n_estimators=50, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
    utils.SAFE_FIT(model, X.values.astype(np.float32), np.log1p(y).astype(np.float32))
    
    base_tail_mae = mean_absolute_error(y_tail, np.expm1(utils.SAFE_PREDICT(model, X_tail.values.astype(np.float32))))
    
    # Permutation importance ONLY on tail samples
    np.random.seed(42)
    sample_cols = np.random.choice(X.columns, size=min(50, len(X.columns)), replace=False)
    
    imp_dict = {}
    for col in sample_cols:
        X_perm = X_tail.copy()
        X_perm[col] = np.random.permutation(X_perm[col])
        perm_mae = mean_absolute_error(y_tail, np.expm1(utils.SAFE_PREDICT(model, X_perm.values.astype(np.float32))))
        imp_dict[col] = perm_mae - base_tail_mae # Positive means feature is IMPORTANT
        
    sorted_imp = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 10 TAIL-CRITICAL Features (Features that prevent tail MAE collapse):")
    for f, imp in sorted_imp[:10]:
        print(f"{f:<35} | MAE impact: +{imp:.4f}")
        
    print("\nTop 5 TAIL-DESTRUCTIVE Features (Features that HURT tail prediction):")
    for f, imp in sorted_imp[-5:]:
        if imp < 0:
            print(f"{f:<35} | MAE impact: {imp:.4f}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default="run_20260503_160600")
    args = parser.parse_args()
    
    latest_run = args.run_id
    proc_dir = f"outputs/{latest_run}/processed"
    
    train_base = pd.read_pickle(f"{proc_dir}/train_base.pkl")
    y_train = np.load(f"{proc_dir}/y_train.npy")
    
    # Subset 10k for speed
    np.random.seed(42)
    idx = np.random.choice(len(y_train), min(10000, len(y_train)), replace=False)
    X = train_base.iloc[idx].select_dtypes(include=[np.number]).fillna(0)
    if 'target' in X.columns: X = X.drop(columns=['target'])
    if 'avg_delay_minutes_next_30m' in X.columns: X = X.drop(columns=['avg_delay_minutes_next_30m'])
    y = y_train[idx]
    
    compute_risk_controller_stats(latest_run)
    evaluate_models(X, y)
    tail_feature_audit(X, y)

if __name__ == "__main__":
    main()
