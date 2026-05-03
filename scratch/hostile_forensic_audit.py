
import pandas as pd
import numpy as np
import os
import sys
import logging
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

# Add src to path
sys.path.append(os.getcwd())

from src.config import Config
from src.data_loader import load_data, add_time_series_features, add_extreme_detection_features
from src.trainer import Trainer

logging.basicConfig(level=logging.ERROR) # Quiet logging for forensic report
logger = logging.getLogger("HOSTILE_AUDITOR")

def mission_1_layout_audit(df_train, df_test):
    print("\n" + "="*60)
    print(" [MISSION 1] LAYOUT FALLBACK FORENSIC ")
    print("="*60)
    
    # Simulate Unknown Layouts: Pick 20% of layouts to hide
    all_lids = df_train['layout_id'].unique()
    np.random.seed(42)
    hidden_lids = np.random.choice(all_lids, size=int(len(all_lids)*0.2), replace=False)
    
    tr_mask = ~df_train['layout_id'].isin(hidden_lids)
    val_mask = df_train['layout_id'].isin(hidden_lids)
    
    tr_df = df_train[tr_mask].copy()
    val_df = df_train[val_mask].copy()
    
    # Target feature for bias check
    target_feat = 'order_inflow_15m'
    actual_target_mean = val_df[Config.TARGET].mean()
    
    # 1. Global Fallback
    global_mean = tr_df[target_feat].mean()
    
    # 2. Hierarchical Fallback
    type_means = tr_df.groupby('layout_type')[target_feat].mean().to_dict()
    val_df['hierarchical_val'] = val_df['layout_type'].map(type_means).fillna(global_mean)
    
    # Model comparison (Simple proxy)
    # We'll just measure the Bias of the fallback itself first
    bias_global = abs(global_mean - val_df[target_feat].mean())
    bias_hierarchical = abs(val_df['hierarchical_val'].mean() - val_df[target_feat].mean())
    
    # Full Model Test
    # Features (Raw only for speed)
    cols = [c for c in df_train.columns if pd.api.types.is_numeric_dtype(df_train[c]) and c not in Config.ID_COLS and c != Config.TARGET]
    
    def train_and_eval(fallback_type):
        v_df = val_df.copy()
        if fallback_type == 'global':
            v_df[f'{target_feat}_layout_mean'] = global_mean
        else:
            v_df[f'{target_feat}_layout_mean'] = v_df['hierarchical_val']
            
        # Training a small model on tr_df (which HAS layout_id stats)
        # Note: In real trainer, we compute these stats.
        model = LGBMRegressor(n_estimators=100, verbose=-1, random_state=42)
        model.fit(tr_df[cols], tr_df[Config.TARGET])
        preds = model.predict(v_df[cols])
        
        mae = mean_absolute_error(v_df[Config.TARGET], preds)
        bias = preds.mean() - v_df[Config.TARGET].mean()
        return mae, bias, preds.mean()

    mae_g, bias_g, pred_m_g = train_and_eval('global')
    mae_h, bias_h, pred_m_h = train_and_eval('hierarchical')
    
    print(f"\nTarget Mean (Unknown Layouts): {actual_target_mean:.4f}")
    print(f"--- Global Fallback ---")
    print(f"MAE: {mae_g:.4f} | Bias: {bias_g:.4f} | Pred Mean: {pred_m_g:.4f}")
    print(f"--- Hierarchical Fallback ---")
    print(f"MAE: {mae_h:.4f} | Bias: {bias_h:.4f} | Pred Mean: {pred_m_h:.4f}")
    
    status = "PASS" if (mae_h <= mae_g and abs(bias_h) < abs(bias_g)) else "FAIL"
    print(f"\n[MISSION 1 JUDGMENT] {status}")

def mission_2_weighting_audit(df_train, df_test):
    print("\n" + "="*60)
    print(" [MISSION 2] ADVERSARIAL WEIGHTING FORENSIC ")
    print("="*60)
    
    # Temporal Split (Deciles)
    df_train['id_num'] = df_train['ID'].str.extract(r'(\d+)').astype(int)
    df_train['decile'] = pd.qcut(df_train['id_num'], 10, labels=False)
    
    # 80/20 Temporal Split for audit
    split_idx = int(len(df_train) * 0.8)
    tr = df_train.iloc[:split_idx].copy()
    val = df_train.iloc[split_idx:].copy()
    
    cols = [c for c in tr.columns if pd.api.types.is_numeric_dtype(tr[c]) and c not in Config.ID_COLS and c != Config.TARGET and c != 'decile' and c != 'id_num']
    
    # Compute Weights
    trainer = Trainer(tr, tr[Config.TARGET], df_test)
    weights = trainer.compute_adversarial_weights(cols)
    
    # 1. No Weights
    m_off = LGBMRegressor(n_estimators=200, verbose=-1, random_state=42)
    m_off.fit(tr[cols], tr[Config.TARGET])
    val['preds_off'] = m_off.predict(val[cols])
    
    # 2. With Weights
    m_on = LGBMRegressor(n_estimators=200, verbose=-1, random_state=42)
    m_on.fit(tr[cols], tr[Config.TARGET], sample_weight=weights)
    val['preds_on'] = m_on.predict(val[cols])
    
    # Temporal Decile Analysis
    results = []
    for d in range(8, 10): # Last 2 deciles are our validation proxy
        mae_off = mean_absolute_error(val[val['decile']==d][Config.TARGET], val[val['decile']==d]['preds_off'])
        mae_on = mean_absolute_error(val[val['decile']==d][Config.TARGET], val[val['decile']==d]['preds_on'])
        results.append({'decile': d, 'MAE_OFF': mae_off, 'MAE_ON': mae_on})
        
    res_df = pd.DataFrame(results)
    print("\nTemporal Decile MAE (Tail Scenarios):")
    print(res_df)
    
    print(f"\nWeight Stats: Min={weights.min():.4f}, Max={weights.max():.4f}, Mean={weights.mean():.4f}, Std={weights.std():.4f}")
    
    # Check if tail (decile 9) improved
    d9_imp = res_df.loc[res_df['decile']==9, 'MAE_ON'].values[0] < res_df.loc[res_df['decile']==9, 'MAE_OFF'].values[0]
    status = "VALID" if d9_imp else "INVALID"
    print(f"\n[MISSION 2 JUDGMENT] {status}")

def mission_3_pruning_audit(df_train, df_test):
    print("\n" + "="*60)
    print(" [MISSION 3] PRUNING OVER-KILL AUDIT ")
    print("="*60)
    
    cols_all = [c for c in df_train.columns if pd.api.types.is_numeric_dtype(df_train[c]) and c not in Config.ID_COLS and c != Config.TARGET and c != 'decile' and c != 'id_num']
    
    # Pruning simulation (Prune 30% most drifted by KS for demo)
    from src.distribution import DomainShiftAudit
    audit = DomainShiftAudit()
    drift_df, _ = audit.calculate_drift(df_train.sample(10000), df_test.sample(10000), cols_all)
    pruned_cols = drift_df[drift_df['ks_stat'] > 0.15]['feature'].tolist()
    cols_stable = [c for c in cols_all if c not in pruned_cols]
    
    print(f"Initial Count: {len(cols_all)} | Pruned: {len(pruned_cols)} | Stable: {len(cols_stable)}")
    
    # 1. Full Model
    m_full = LGBMRegressor(n_estimators=200, verbose=-1, random_state=42)
    m_full.fit(df_train[cols_all], df_train[Config.TARGET])
    imp_full = pd.Series(m_full.feature_importances_, index=cols_all).sort_values(ascending=False)
    
    # 2. Stable Model
    m_stable = LGBMRegressor(n_estimators=200, verbose=-1, random_state=42)
    m_stable.fit(df_train[cols_stable], df_train[Config.TARGET])
    
    # Check if we killed high-importance features
    killed_important = [f for f in imp_full.head(20).index if f in pruned_cols]
    
    print("\nTop 10 Full Model Features:")
    print(imp_full.head(10))
    print(f"\nImportant Features Killed: {len(killed_important)}")
    if killed_important:
        print(f"Killed List: {killed_important}")
        
    # Check Tail MAE (P95)
    val_preds_full = m_full.predict(df_train[cols_all])
    val_preds_stable = m_stable.predict(df_train[cols_stable])
    
    mae_full = mean_absolute_error(df_train[Config.TARGET], val_preds_full)
    mae_stable = mean_absolute_error(df_train[Config.TARGET], val_preds_stable)
    
    # Tail MAE (samples with target > P95)
    p95 = np.percentile(df_train[Config.TARGET], 95)
    tail_mask = df_train[Config.TARGET] > p95
    tail_mae_full = mean_absolute_error(df_train[tail_mask][Config.TARGET], val_preds_full[tail_mask])
    tail_mae_stable = mean_absolute_error(df_train[tail_mask][Config.TARGET], val_preds_stable[tail_mask])
    
    print(f"\nOverall MAE: Full={mae_full:.4f}, Stable={mae_stable:.4f}")
    print(f"Tail (P95) MAE: Full={tail_mae_full:.4f}, Stable={tail_mae_stable:.4f}")
    
    status = "PASS" if (mae_stable <= mae_full * 1.05 and tail_mae_stable <= tail_mae_full * 1.05) else "FAIL"
    print(f"\n[MISSION 3 JUDGMENT] {status}")

if __name__ == "__main__":
    df_train, df_test = load_data()
    mission_1_layout_audit(df_train, df_test)
    mission_2_weighting_audit(df_train, df_test)
    mission_3_pruning_audit(df_train, df_test)
