import os
import gc
import logging
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

# Add parent dir to path so we can import src
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config
from src.schema import FEATURE_SCHEMA

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Diagnostic")

def get_scenario_order(df):
    temp_df = df[["scenario_id", "ID"]].copy()
    temp_df["id_num"] = temp_df["ID"].str.extract(r"(\d+)").astype(int)
    scenario_time = temp_df.groupby("scenario_id")["id_num"].min().sort_values()
    return scenario_time.index.tolist()

def load_and_split_subset():
    logger.info("Loading train subset (20%)...")
    df = pd.read_csv("./data/train.csv")
    
    scenarios = get_scenario_order(df)
    subset_scenarios = scenarios[:max(1, len(scenarios) // 5)] # 20%
    df_subset = df[df["scenario_id"].isin(subset_scenarios)].copy()
    
    # temporal split 80:20 inside the subset
    subset_scenarios_sorted = get_scenario_order(df_subset)
    split_idx = int(len(subset_scenarios_sorted) * 0.8)
    train_scenarios = subset_scenarios_sorted[:split_idx]
    val_scenarios = subset_scenarios_sorted[split_idx:]
    
    tr_df = df_subset[df_subset["scenario_id"].isin(train_scenarios)].copy()
    val_df = df_subset[df_subset["scenario_id"].isin(val_scenarios)].copy()
    
    # Use simple numeric features to bypass heavy processing
    features = [c for c in FEATURE_SCHEMA["raw_features"] if c in tr_df.columns and pd.api.types.is_numeric_dtype(tr_df[c])]
    
    X_tr = tr_df[features].fillna(-999).values.astype(np.float32)
    y_tr = tr_df[Config.TARGET].values
    
    X_val = val_df[features].fillna(-999).values.astype(np.float32)
    y_val = val_df[Config.TARGET].values
    
    logger.info(f"Train shape: {X_tr.shape}, Val shape: {X_val.shape}")
    return X_tr, y_tr, X_val, y_val

def eval_experiment(name, X_tr, y_tr, X_val, y_val, lgbm_params, tail_weighting, log_target):
    logger.info(f"\\n--- Running {name} ---")
    
    y_tr_fit = np.log1p(y_tr) if log_target else y_tr
    y_val_fit = np.log1p(y_val) if log_target else y_val
    
    sample_weight = None
    if tail_weighting:
        y_p90 = np.quantile(y_tr, 0.90)
        tail_weight = np.where(
            y_tr > y_p90,
            1.0 + Config.TARGET_AWARE_ALPHA * (y_tr - y_p90) / (np.std(y_tr) + 1e-6),
            1.0,
        )
        sample_weight = np.clip(tail_weight, 1.0, Config.TARGET_AWARE_MAX_WEIGHT)
    
    model = LGBMRegressor(**lgbm_params, random_state=42)
    model.fit(X_tr, y_tr_fit, sample_weight=sample_weight)
    
    preds_fit = model.predict(X_val)
    preds = np.expm1(preds_fit) if log_target else preds_fit
    
    mae = mean_absolute_error(y_val, preds)
    
    # Metrics calculation
    train_std = np.std(y_tr)
    pred_std = np.std(preds)
    std_ratio = pred_std / (train_std + 1e-9)
    
    mean_ratio = np.mean(preds) / (np.mean(y_tr) + 1e-9)
    
    # robust quantiles
    pred_p90, pred_p99 = np.quantile(preds, [0.90, 0.99])
    true_p90, true_p99 = np.quantile(y_tr, [0.90, 0.99])
    
    p90_ratio = pred_p90 / (true_p90 + 1e-9)
    p99_ratio = pred_p99 / (true_p99 + 1e-9)
    
    max_pred = np.max(preds)
    
    logger.info(f"Results for {name}:")
    logger.info(f"  MAE:            {mae:.4f}")
    logger.info(f"  std_ratio:      {std_ratio:.4f}")
    logger.info(f"  mean_ratio:     {mean_ratio:.4f}")
    logger.info(f"  p90_ratio:      {p90_ratio:.4f}")
    logger.info(f"  p99_ratio:      {p99_ratio:.4f}")
    logger.info(f"  max_prediction: {max_pred:.4f}")
    
    return {
        "name": name,
        "mae": mae,
        "std_ratio": std_ratio,
        "mean_ratio": mean_ratio,
        "p90_ratio": p90_ratio,
        "p99_ratio": p99_ratio,
        "max_pred": max_pred
    }

def main():
    X_tr, y_tr, X_val, y_val = load_and_split_subset()
    
    base_params = {
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'n_estimators': 300,
        'verbose': -1,
        'n_jobs': -1,
    }
    
    results = []
    
    # [Experiment A]
    # loss: Huber (current alpha), tail weighting: OFF, log transform: OFF
    params_a = base_params.copy()
    params_a['objective'] = 'huber'
    params_a['alpha'] = 21.5 # Current config
    results.append(eval_experiment("Experiment A (Huber, No TailWt, No Log)", X_tr, y_tr, X_val, y_val, params_a, False, False))
    
    # [Experiment B]
    # loss: MAE, tail weighting: ON, log transform: OFF
    params_b = base_params.copy()
    params_b['objective'] = 'mae'
    results.append(eval_experiment("Experiment B (MAE, TailWt ON, No Log)", X_tr, y_tr, X_val, y_val, params_b, True, False))
    
    # [Experiment C]
    # loss: Huber, tail weighting: OFF, log transform: OFF, alpha scaling = IQR * 0.5, 1.0, 2.0
    iqr = np.percentile(y_tr, 75) - np.percentile(y_tr, 25)
    for scale in [0.5, 1.0, 2.0]:
        alpha_val = iqr * scale
        params_c = base_params.copy()
        params_c['objective'] = 'huber'
        params_c['alpha'] = alpha_val
        name = f"Experiment C (Huber alpha={alpha_val:.2f} [IQR*{scale}], No TailWt, No Log)"
        results.append(eval_experiment(name, X_tr, y_tr, X_val, y_val, params_c, False, False))

    print("\\n" + "="*50)
    print("SUMMARY OF RESULTS")
    print("="*50)
    print(f"{'Experiment':<55} | {'MAE':<7} | {'std_ratio':<9} | {'mean_ratio':<10} | {'p90_ratio':<9} | {'p99_ratio':<9} | {'max_pred':<8}")
    print("-" * 125)
    for r in results:
        print(f"{r['name']:<55} | {r['mae']:.4f}  | {r['std_ratio']:.4f}     | {r['mean_ratio']:.4f}      | {r['p90_ratio']:.4f}     | {r['p99_ratio']:.4f}     | {r['max_pred']:.4f}")

if __name__ == "__main__":
    main()
