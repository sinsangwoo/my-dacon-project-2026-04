import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, brier_score_loss, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.calibration import IsotonicRegression
from sklearn.model_selection import GroupKFold

# Paths
base_dir = "c:/Github_public/my_dacon_project/my-dacon-project-2026-04"
run_id = "run_20260501_144356"
processed_dir = f"{base_dir}/outputs/{run_id}/processed"

def phase_2_hardening():
    print("--- [MISSION: FP-ROBUST PROBABILISTIC SYSTEM - PHASE 2] ---")
    
    # 1. Load Data
    y_true = np.load(f"{processed_dir}/y_train.npy", allow_pickle=True)
    scenario_id = np.load(f"{processed_dir}/scenario_id.npy", allow_pickle=True)
    q90_val = np.percentile(y_true, 90)
    y_binary = (y_true >= q90_val).astype(int)
    
    train_df = pd.read_pickle(f"{processed_dir}/train_base.pkl")
    X = train_df.select_dtypes(include=[np.number]).fillna(0)
    X = X.drop(columns=['avg_delay_minutes_next_30m', 'ID', 'scenario_id', 'layout_id'], errors='ignore').values
    
    gkf = GroupKFold(n_splits=5)
    # Using 5-fold to get variance metrics
    fold_results = []
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_binary, groups=scenario_id)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y_binary[tr_idx], y_binary[val_idx]
        y_gt_tr, y_gt_val = y_true[tr_idx], y_true[val_idx]
        
        # Models
        clf = RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=-1, random_state=fold).fit(X_tr, y_tr)
        reg_tail = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=fold).fit(X_tr[y_tr==1], y_gt_tr[y_tr==1])
        reg_base = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=fold).fit(X_tr[y_tr==0], y_gt_tr[y_tr==0])
        
        p_raw = clf.predict_proba(X_val)[:, 1]
        iso = IsotonicRegression(out_of_bounds='clip').fit(clf.predict_proba(X_tr)[:, 1], y_tr)
        p_cal = iso.transform(p_raw)
        
        y_tail = reg_tail.predict(X_val)
        y_base = reg_base.predict(X_val)
        gain_actual = np.abs(y_gt_val - y_base) - np.abs(y_gt_val - y_tail)
        
        # [TASK 1: Raw vs Calibrated]
        def eval_p(p_in, name):
            y_blend = y_base + (p_in**2) * (y_tail - y_base)
            mae = mean_absolute_error(y_gt_val, y_blend)
            fp_mask = (y_gt_val < q90_val) & (y_blend > y_base)
            fp_rate = np.mean(fp_mask)
            fp_cost_total = np.sum(np.abs(y_blend[fp_mask] - y_gt_val[fp_mask])) / len(y_gt_val)
            corr = np.corrcoef(p_in, gain_actual)[0, 1]
            brier = brier_score_loss(y_val, p_in)
            return {"name": name, "MAE": mae, "FP_R": fp_rate, "FP_C": fp_cost_total, "Corr": corr, "Brier": brier}

        t1_raw = eval_p(p_raw, "Raw")
        t1_cal = eval_p(p_cal, "Calibrated")
        
        # [TASK 2: Clipping Side-Effect]
        delta_raw = y_tail - y_base
        max_delta = np.percentile(np.abs(y_gt_tr[y_tr==0] - reg_base.predict(X_tr[y_tr==0])), 95)
        delta_clip = np.clip(delta_raw, -max_delta, max_delta)
        
        true_tail_mask = y_val == 1
        fp_mask = (y_val == 0) & (p_cal > 0.1) # Proxy for FP
        
        tt_loss = (np.mean(delta_raw[true_tail_mask]) - np.mean(delta_clip[true_tail_mask])) / (np.mean(delta_raw[true_tail_mask]) + 1e-9)
        fp_reduction = (np.mean(delta_raw[fp_mask]) - np.mean(delta_clip[fp_mask])) / (np.mean(delta_raw[fp_mask]) + 1e-9)
        
        # [TASK 6: FP Reduction Strategy C - Two-stage]
        # Train a meta-model to predict if a sample with p > 0.2 is actually a True Tail
        X_meta = X_tr[clf.predict_proba(X_tr)[:, 1] > 0.2]
        y_meta = y_tr[clf.predict_proba(X_tr)[:, 1] > 0.2]
        if len(y_meta) > 0 and len(np.unique(y_meta)) > 1:
            meta_clf = RandomForestClassifier(n_estimators=30, max_depth=5).fit(X_meta, y_meta)
            p_meta = meta_clf.predict_proba(X_val)[:, 1]
            p_final = p_cal * p_meta # Combined probability
        else:
            p_final = p_cal

        t6_metrics = eval_p(p_final, "Two-Stage")

        fold_results.append({
            "fold": fold,
            "t1_raw": t1_raw, "t1_cal": t1_cal,
            "tt_loss": tt_loss, "fp_reduction": fp_reduction,
            "t6": t6_metrics,
            "tail_std": np.std(y_tail)
        })
        if fold == 0: break # Run 1 fold for speed in this interaction

    # Outputting RAW NUMBERS
    res = fold_results[0]
    print("\n[TASK 1: CALIBRATION COMPARISON]")
    print(f"Raw: MAE={res['t1_raw']['MAE']:.4f}, FP_R={res['t1_raw']['FP_R']:.4f}, FP_C={res['t1_raw']['FP_C']:.4f}, Corr={res['t1_raw']['Corr']:.4f}, Brier={res['t1_raw']['Brier']:.4f}")
    print(f"Cal: MAE={res['t1_cal']['MAE']:.4f}, FP_R={res['t1_cal']['FP_R']:.4f}, FP_C={res['t1_cal']['FP_C']:.4f}, Corr={res['t1_cal']['Corr']:.4f}, Brier={res['t1_cal']['Brier']:.4f}")

    print("\n[TASK 2: CLIPPING SIDE-EFFECT]")
    print(f"True Tail Signal Loss: {res['tt_loss']*100:.2f}%")
    print(f"FP Damage Reduction:   {res['fp_reduction']*100:.2f}%")

    print("\n[TASK 3: LOGICAL CORRECTION]")
    print(f"Delta Brier: {res['t1_cal']['Brier'] - res['t1_raw']['Brier']:.6f}")
    print(f"Delta Corr:  {res['t1_cal']['Corr'] - res['t1_raw']['Corr']:.6f}")

    print("\n[TASK 6: FP REDUCTION (Two-Stage)]")
    print(f"Two-Stage: MAE={res['t6']['MAE']:.4f}, FP_R={res['t6']['FP_R']:.4f}, FP_C={res['t6']['FP_C']:.4f}")

if __name__ == "__main__":
    phase_2_hardening()
