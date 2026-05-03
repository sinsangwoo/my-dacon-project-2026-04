import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import KFold
import gc

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def gain_prediction_mission():
    print("--- [MISSION: GAIN PREDICTION SYSTEM DESIGN] ---")
    
    # 1. Load Data
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    q90_val = np.percentile(y_true, 90)
    y_binary = (y_true >= q90_val).astype(int)
    
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    X = train_df.select_dtypes(include=[np.number]).fillna(0)
    X = X.drop(columns=['avg_delay_minutes_next_30m', 'ID', 'scenario_id', 'layout_id'], errors='ignore').values
    
    # [OPTIMIZATION] 20% sampling for rapid validation
    sample_mask = np.random.choice([True, False], len(y_true), p=[0.2, 0.8])
    X = X[sample_mask]
    y_true = y_true[sample_mask]
    y_binary = y_binary[sample_mask]
    
    # 2. Stage 0: Generate Gain Labels via 3-Fold OOF
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    oof_base = np.zeros(len(y_true))
    oof_tail = np.zeros(len(y_true))
    oof_p = np.zeros(len(y_true))
    
    for tr, val in kf.split(X):
        m_base = LGBMRegressor(n_estimators=50).fit(X[tr][y_binary[tr]==0], np.log1p(y_true[tr][y_binary[tr]==0]))
        m_tail = LGBMRegressor(n_estimators=50).fit(X[tr][y_binary[tr]==1], np.log1p(y_true[tr][y_binary[tr]==1]))
        m_clf = LGBMClassifier(n_estimators=50).fit(X[tr], y_binary[tr])
        
        oof_base[val] = np.expm1(m_base.predict(X[val]))
        oof_tail[val] = np.expm1(m_tail.predict(X[val]))
        oof_p[val] = m_clf.predict_proba(X[val])[:, 1]
    
    gain = np.abs(y_true - oof_base) - np.abs(y_true - oof_tail)
    gain_class = (gain > 0).astype(int)
    
    # 3. [TASK 2: Gain Predictor Training]
    print("\n[TASK 2: GAIN PREDICTOR PERFORMANCE]")
    oof_gain_pred = np.zeros(len(y_true))
    oof_gain_class_p = np.zeros(len(y_true))
    
    for tr, val in kf.split(X):
        # A. Classification
        m_gain_clf = LGBMClassifier(n_estimators=50).fit(X[tr], gain_class[tr])
        oof_gain_class_p[val] = m_gain_clf.predict_proba(X[val])[:, 1]
        
        # B. Regression
        m_gain_reg = LGBMRegressor(n_estimators=50).fit(X[tr], gain[tr])
        oof_gain_pred[val] = m_gain_reg.predict(X[val])

    print(f"Gain Class ROC-AUC: {roc_auc_score(gain_class, oof_gain_class_p):.4f}")
    print(f"Gain Class PR-AUC:  {average_precision_score(gain_class, oof_gain_class_p):.4f}")
    print(f"Gain Reg Corr:      {np.corrcoef(gain, oof_gain_pred)[0,1]:.4f}")

    # 4. [TASK 6: FULL PIPELINE EVALUATION]
    print("\n[TASK 6: PIPELINE COMPARISON]")
    # Baseline: p-based (p > 0.5)
    y_p_based = np.where(oof_p > 0.5, oof_tail, oof_base)
    # Gain-based: predicted_gain > 0
    y_gain_based = np.where(oof_gain_pred > 0, oof_tail, oof_base)
    # Gain-Class-based: predicted_gain_class_p > 0.5
    y_gain_class_based = np.where(oof_gain_class_p > 0.5, oof_tail, oof_base)
    
    results = [
        {"Method": "No Tail", "MAE": mean_absolute_error(y_true, oof_base)},
        {"Method": "p-based (p>0.5)", "MAE": mean_absolute_error(y_true, y_p_based)},
        {"Method": "Gain-Reg (pred>0)", "MAE": mean_absolute_error(y_true, y_gain_based)},
        {"Method": "Gain-Clf (p>0.5)", "MAE": mean_absolute_error(y_true, y_gain_class_based)}
    ]
    print(pd.DataFrame(results))

    # 5. [CRITICAL VALIDATION]
    print("\n[CRITICAL VALIDATION]")
    print(f"p vs Gain corr: {np.corrcoef(oof_p, gain)[0,1]:.4f}")
    print(f"Gain-Pred vs Gain corr: {np.corrcoef(oof_gain_pred, gain)[0,1]:.4f}")

if __name__ == "__main__":
    gain_prediction_mission()
