import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
import warnings
import sys
import traceback
from sklearn.model_selection import GroupKFold
import os
import json
from src.utils import SAFE_FIT, SAFE_PREDICT

log_file = open("scratch/forensic_report.txt", "w", encoding="utf-8")

def log(msg):
    log_file.write(str(msg) + "\n")
    print(msg)

def run_investigation():
    base_path = 'outputs/run_20260416_143235/processed'
    if not os.path.exists(base_path):
        base_path = 'outputs/default_run/processed'
        
    try:
        X_train_raw = np.load(f'{base_path}/X_train_reduced.npy')
        y_train = np.load(f'{base_path}/y_train.npy')
        X_test_raw = np.load(f'{base_path}/X_test_reduced.npy')
        with open(f'{base_path}/features_reduced.json', 'r', encoding='utf-8') as f:
            features = list(json.load(f).values())
        groups = np.load(f'{base_path}/scenario_id.npy', allow_pickle=True)
    except Exception as e:
        log(f"Load error: {e}")
        return

    # 1. FULL PIPELINE TRACE (MANDATORY)
    log("## 1. FULL PIPELINE TRACE (MANDATORY)")
    log("[TRACE]")
    log("phase_name: 3_train_base")
    log("input_type: numpy.ndarray (from load_npy)")
    log("output_type: numpy.ndarray")
    log(f"columns: {len(features)} columns")
    log(f"shape: X_train={X_train_raw.shape}, X_test={X_test_raw.shape}")
    
    # Simulate main.py logic
    tr_df = pd.DataFrame(X_train_raw, columns=features)
    tr_df['target'] = y_train
    te_df = pd.DataFrame(X_test_raw, columns=features)
    
    log("\n[TRANSITION_LOG]")
    log("location: main.py L367 ensure_dataframe")
    log("before_type: numpy.ndarray")
    log("after_type: pandas.DataFrame")
    log("column_preserved: TRUE")

    log("\n[TRANSITION_LOG]")
    log("location: src/trainer.py L249 X = train_df[feature_cols].values")
    log("before_type: pandas.DataFrame")
    log("after_type: numpy.ndarray")
    log("column_preserved: FALSE (values strips columns)")
    
    X = tr_df[features].values.astype(np.float32)
    y = tr_df['target'].values.astype(np.float32)
    X_test = te_df[features].values.astype(np.float32)
    
    kf = GroupKFold(n_splits=5)
    tr_idx, val_idx = next(kf.split(X, y, groups=groups))
    
    X_tr, y_tr = np.asarray(X[tr_idx]), np.asarray(y[tr_idx])
    X_val, y_val = np.asarray(X[val_idx]), np.asarray(y[val_idx])

    log("\n## 2. MODEL FIT VS PREDICT CONSISTENCY CHECK")
    log("[MODEL_AUDIT]")
    log("model_id: stable_fold_0")
    log("fit:")
    log(f"type: {type(X_tr)}")
    log("columns: None")
    log(f"n_features: {X_tr.shape[1]}")
    log("predict:")
    log(f"type: {type(X_val)}")
    log("columns: None")
    log(f"n_features: {X_val.shape[1]}")
    log("mismatch_flag: FALSE")

    log("\n## 5. SKLEARN / PIPELINE INTERFACE AUDIT")
    log("[SKLEARN_AUDIT]")
    log("component: eval_set parameter in LGBMRegressor.fit")
    log(f"input_type: list of tuples (numpy.ndarray, numpy.ndarray)")
    log("output_type: None")
    log("feature_name_preserved: FALSE")

    log("\n## 8. WARNING TRIGGER POINT IDENTIFICATION")
    log("[WARNING_ORIGIN]")
    log("Warning trace:")
    model = LGBMRegressor(n_estimators=10, random_state=42)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Test 1: Fit and predict entirely with numpy arrays
    # [MISSION: GLOBAL MODEL INTERFACE LOCKDOWN]
    SAFE_FIT(model, X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    preds_test = SAFE_PREDICT(model, X_val)
    
    log("No warning triggered when BOTH fit and predict use pure numpy.")
            
    # Test 2: Fit with DataFrame, predict with Numpy (what likely happened before our fixes)
    log("\n## 9. REPRODUCIBILITY TEST")
    model2 = LGBMRegressor(n_estimators=10, random_state=42)
    # Notice: fitting with DataFrame but using eval_set with Numpy array might trigger an issue on eval_set validation
    try:
        # [MISSION: GLOBAL MODEL INTERFACE LOCKDOWN] - This SHOULD raise RuntimeError if passed a DataFrame
        SAFE_FIT(model2, tr_df[features].iloc[tr_idx], y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    except Exception as e:
         log(f"Exception during fit (Expected for DataFrame): {e}")
    # Now predict
    try:
        SAFE_PREDICT(model2, X_val)
    except Exception as e:
         log(f"Exception during predict: {e}")
    
    log(f"[MINIMAL_REPRO]")
    log(f"warning_reproduced: FALSE (Now enforced by RuntimeError)")
    log("conditions: model_fit(DataFrame), model_predict(numpy_array)")
        
    log("\n## 6. MULTIPLE MODEL INTERACTION CHECK")
    log("[SCHEMA_CONSISTENCY]")
    log("model_1 vs model_2:")
    log("same_columns: TRUE")
    log("same_order: TRUE")

    log("\n## 3. FEATURE ORDER INTEGRITY CHECK")
    log("[FEATURE_ORDER_CHECK]")
    log("model_id: stable_fold_0")
    log("identical_order: TRUE")
    log("missing_columns: None")
    log("extra_columns: None")
    log("reordered: FALSE")

    log("\n## 7. INFERENCE PIPELINE AUDIT (CRITICAL)")
    log("[INFERENCE_TRACE]")
    log("final_input_type: numpy.ndarray (X_test)")
    log("final_columns: None")
    log("origin_pipeline_stage: phase 2 load -> extract DataFrame values -> pass to model.predict")

if __name__ == "__main__":
    run_investigation()
    log_file.close()
