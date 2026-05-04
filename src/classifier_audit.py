import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold
import lightgbm as lgb
from scipy.stats import ks_2samp
from sklearn.feature_selection import mutual_info_classif
import warnings
from . import utils
from .utils import SAFE_FIT, SAFE_PREDICT, SAFE_PREDICT_PROBA

warnings.filterwarnings("ignore")

def run_classifier_audit():
    print("🔬 Starting Classifier Reconstruction Audit...")
    
    # Load data
    proc_dir = "outputs/run_20260503_150859/processed"
    if not os.path.exists(proc_dir):
        print(f"Directory not found: {proc_dir}")
        # Try to find the latest run
        outputs = os.listdir("outputs")
        latest_run = sorted([d for d in outputs if d.startswith("run_")])[-1]
        proc_dir = f"outputs/{latest_run}/processed"
        print(f"Using fallback directory: {proc_dir}")
        
    train_base = pd.read_pickle(f"{proc_dir}/train_base.pkl")
    y_train = np.load(f"{proc_dir}/y_train.npy")
    
    # Use a 20k subset for speed, similar to TRACE_ROWS
    np.random.seed(42)
    idx = np.random.choice(len(y_train), min(20000, len(y_train)), replace=False)
    X = train_base.iloc[idx].select_dtypes(include=[np.number]).fillna(0)
    
    # DROP LEAKAGE COLUMNS
    if 'target' in X.columns:
        X = X.drop(columns=['target'])
    if 'avg_delay_minutes_next_30m' in X.columns:
        X = X.drop(columns=['avg_delay_minutes_next_30m'])
        
    y = y_train[idx]
    
    print(f"\nData Shape: X={X.shape}, y={y.shape}")

    # --- 1. Threshold Sensitivity Scan ---
    print("\n" + "="*50)
    print(" 1. Threshold Sensitivity Scan")
    print("="*50)
    print(f"{'Threshold':<10} | {'Tail Ratio':<12} | {'Prec@0.7':<10} | {'Recall':<8} | {'AUC':<8}")
    print("-" * 60)
    
    thresholds = [90, 92, 95, 97, 99]
    best_thresh = None
    max_prec = 0
    
    for q in thresholds:
        q_val = np.percentile(y, q)
        y_bin = (y >= q_val).astype(int)
        
        # Train simple LGBM
        model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
        
        # Simple CV
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        prec_list, rec_list, auc_list = [], [], []
        
        for tr_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y_bin[tr_idx], y_bin[val_idx]
            
            if sum(y_tr) == 0: continue
            
            utils.SAFE_FIT(model, X_tr.values.astype(np.float32), y_tr.astype(np.float32))
            p_val = utils.SAFE_PREDICT_PROBA(model, X_val.values.astype(np.float32))[:, 1]
            
            high_conf = p_val > 0.7
            if sum(high_conf) > 0:
                prec_list.append(precision_score(y_val[high_conf], [1]*sum(high_conf), zero_division=0))
            else:
                prec_list.append(0)
                
            pred_bin = (p_val > 0.5).astype(int)
            rec_list.append(recall_score(y_val, pred_bin, zero_division=0))
            try:
                auc_list.append(roc_auc_score(y_val, p_val))
            except:
                pass
                
        prec = np.mean(prec_list)
        rec = np.mean(rec_list)
        auc = np.mean(auc_list) if auc_list else 0
        tail_ratio = sum(y_bin) / len(y_bin)
        
        print(f"q{q:<8} | {tail_ratio:.4f}     | {prec:.4f}     | {rec:.4f}   | {auc:.4f}")

    # --- 2. Class Imbalance Impact ---
    print("\n" + "="*50)
    print(" 2. Class Imbalance Impact (Using q95)")
    print("="*50)
    print(f"{'Weight Scheme':<15} | {'Prec@0.7':<10} | {'Recall':<8}")
    print("-" * 40)
    
    q_val = np.percentile(y, 95)
    y_bin = (y >= q_val).astype(int)
    
    weights = [
        ("baseline", None),
        ("scale_pos=2", 2),
        ("scale_pos=5", 5),
        ("scale_pos=10", 10),
        ("asymmetric (fp pen)", 0.2) # penalize false positives by downweighting pos? Actually LGBM scale_pos_weight > 1 helps recall, < 1 helps precision
    ]
    
    for name, w in weights:
        model = lgb.LGBMClassifier(n_estimators=100, random_state=42, scale_pos_weight=w if w else 1.0, verbose=-1, n_jobs=-1)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        prec_list, rec_list = [], []
        
        for tr_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y_bin[tr_idx], y_bin[val_idx]
            utils.SAFE_FIT(model, X_tr.values.astype(np.float32), y_tr.astype(np.float32))
            p_val = utils.SAFE_PREDICT_PROBA(model, X_val.values.astype(np.float32))[:, 1]
            high_conf = p_val > 0.7
            if sum(high_conf) > 0:
                prec_list.append(np.mean(y_val[high_conf]))
            else:
                prec_list.append(0)
            rec_list.append(recall_score(y_val, (p_val > 0.5).astype(int), zero_division=0))
            
        print(f"{name:<15} | {np.mean(prec_list):.4f}     | {np.mean(rec_list):.4f}")

    # --- 3. Feature Signal Audit ---
    print("\n" + "="*50)
    print(" 3. Feature Signal Audit (Top 15 Features by MI)")
    print("="*50)
    
    # MI
    mi_scores = mutual_info_classif(X, y_bin, random_state=42)
    
    # LGBM Importance
    model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
    utils.SAFE_FIT(model, X.values.astype(np.float32), y_bin.astype(np.float32))
    imp_scores = model.feature_importances_
    
    # KS Stat
    ks_scores = []
    for col in X.columns:
        tail_vals = X[col][y_bin == 1]
        nontail_vals = X[col][y_bin == 0]
        if len(tail_vals) > 0 and len(nontail_vals) > 0:
            ks, _ = ks_2samp(tail_vals, nontail_vals)
        else:
            ks = 0
        ks_scores.append(ks)
        
    audit_df = pd.DataFrame({
        "feature": X.columns,
        "KS": ks_scores,
        "MI": mi_scores,
        "importance": imp_scores
    }).sort_values("MI", ascending=False).head(15)
    
    print(audit_df.to_string(index=False))

if __name__ == "__main__":
    run_classifier_audit()
