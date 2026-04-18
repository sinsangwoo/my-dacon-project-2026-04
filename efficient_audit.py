import os
import sys
import numpy as np
import pandas as pd
import logging
import time
import gc
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp

# Internal imports
from src.config import Config
from src.utils import get_logger, seed_everything, load_npy, save_npy, PhaseTracer, memory_guard, SAFE_FIT, SAFE_PREDICT, check_model_contract_compliance
from src.trainer import Trainer

# Configuration for Audit
AUDIT_SEEDS = [42, 2024, 777]
MAX_PHASES = 3
GATE_1_MAE_LIMIT = 9.0
FAIL_FAST_MAE = 9.5
FAIL_FAST_GAP = 1.5
FAIL_FAST_KS = 0.5  # KL-Divergence or KS statistic for distribution shift

GO_MAE_LIMIT = 8.5
GO_GAP_LIMIT = 0.8
GO_STD_LIMIT = 0.4

logger = get_logger()

def calculate_kl_divergence(p, q):
    """Simplified KL-Divergence check via histogram overlap (v1.0)."""
    p_hist, _ = np.histogram(p, bins=50, density=True)
    q_hist, _ = np.histogram(q, bins=50, density=True)
    # Add epsilon to avoid log(0)
    p_hist = p_hist + 1e-10
    q_hist = q_hist + 1e-10
    return np.sum(p_hist * np.log(p_hist / q_hist))

class EfficientAuditor:
    def __init__(self):
        self.X = None
        self.y = None
        self.groups = None
        self.features = None
        self.results = []

    def load_artifacts(self):
        logger.info("--- [AUDIT_INIT] Loading Preprocessed Artifacts ---")
        if not os.path.exists('outputs/processed/X_train_reduced.npy'):
            logger.warning("Missing artifacts! Running Phase 2 (Preprocessing) first...")
            import subprocess
            subprocess.run([sys.executable, 'main.py', '--phase', '2_preprocess', '--mode', 'full'], check=True)
            
        self.X = load_npy('outputs/processed/X_train_reduced.npy')
        self.y = load_npy('outputs/processed/y_train.npy')
        self.groups = load_npy('outputs/processed/scenario_id.npy', allow_pickle=True)
        self.features = pd.read_json('outputs/processed/features_reduced.json', typ='series').tolist()
        logger.info(f"Loaded {self.X.shape[0]} rows, {len(self.features)} features.")

    def run_strategy(self, X_train, y_train, X_val, y_val, groups_train, groups_val, seed):
        """Run CAT, LGB, Blend, Stack in one pass for efficiency."""
        # Initialize Trainer with numpy arrays
        trainer = Trainer(
            X=X_train.astype(np.float32),
            y=y_train.astype(np.float32),
            X_test=X_val.astype(np.float32),
            feature_cols=self.features,
            groups=groups_train
        )
        
        # 1. Train CV (OOFs) on 80% split
        trainer.train_kfolds(feature_cols=self.features, seeds=[seed])
        
        # 2. Get OOF MAEs on 80%
        oof_lgb_mae = mean_absolute_error(y_train, trainer.oof_lgb)
        oof_cat_mae = mean_absolute_error(y_train, trainer.oof_cat)
        
        # 3. Predict on 20% (Pseudo LB)
        # In Trainer, test_preds are already calculated during train_kfolds
        pred_lgb = trainer.test_preds_lgb
        pred_cat = trainer.test_preds_cat
        
        # 4. Conditional Blend
        # Find best threshold on 80% OOF
        best_threshold, _ = trainer.find_best_gating_threshold(y_train)
        pred_blend = np.where(pred_cat < best_threshold, pred_cat, pred_lgb)
        
        # 5. Mini Stacking (Local Implementation for Robust Audit)
        from sklearn.linear_model import Ridge
        X_stack_tr = np.c_[trainer.oof_cat, trainer.oof_lgb].astype(np.float32)
        X_stack_te = np.c_[pred_cat, pred_lgb].astype(np.float32)
        
        meta_model = Ridge(alpha=1.0)
        # [MISSION: GLOBAL MODEL INTERFACE LOCKDOWN]
        SAFE_FIT(meta_model, X_stack_tr, y_train.astype(np.float32))
        pred_stack = SAFE_PREDICT(meta_model, X_stack_te)
            
        return {
            'LGB': mean_absolute_error(y_val, pred_lgb),
            'CAT': mean_absolute_error(y_val, pred_cat),
            'Blend': mean_absolute_error(y_val, pred_blend),
            'Stack': mean_absolute_error(y_val, pred_stack),
            'OOF_LGB': oof_lgb_mae,
            'Pred_Dist': pred_stack
        }

    def run_phase(self, phase_id, n_seeds, n_splits):
        logger.info(f"\n{'#'*60}\n# STARTING PHASE {phase_id} (Seeds: {n_seeds}, Splits: {n_splits})\n{'#'*60}")
        
        phase_results = []
        unique_groups = np.unique(self.groups)
        
        for s_idx in range(n_seeds):
            seed = AUDIT_SEEDS[s_idx]
            seed_everything(seed)
            
            for split_idx in range(n_splits):
                logger.info(f"Phase {phase_id} | Seed {seed} | Split {split_idx+1}/{n_splits}...")
                
                # Scenario-based Split (80/20)
                train_gps, val_gps = train_test_split(unique_groups, test_size=0.2, random_state=seed+split_idx)
                
                mask_train = np.isin(self.groups, train_gps)
                mask_val = np.isin(self.groups, val_gps)
                
                metrics = self.run_strategy(
                    self.X[mask_train], self.y[mask_train],
                    self.X[mask_val], self.y[mask_val],
                    self.groups[mask_train], self.groups[mask_val],
                    seed
                )
                
                # Check for Early Exit
                if phase_id == 1 and metrics['Stack'] > GATE_1_MAE_LIMIT:
                    logger.error(f"!!! [GATE 1 FAILED] Pseudo LB MAE {metrics['Stack']:.4f} > {GATE_1_MAE_LIMIT} !!!")
                    return False, []

                if metrics['Stack'] >= FAIL_FAST_MAE:
                    logger.error(f"!!! [FAIL FAST] MAE {metrics['Stack']:.4f} >= {FAIL_FAST_MAE} !!!")
                    return False, []
                
                gap = metrics['Stack'] - metrics['OOF_LGB'] # Example gap check
                if gap >= FAIL_FAST_GAP:
                    logger.error(f"!!! [FAIL FAST] Gap {gap:.4f} >= {FAIL_FAST_GAP} !!!")
                    return False, []
                
                # Distribution Check (KS Test)
                ks_stat, _ = ks_2samp(self.y[mask_val], metrics['Pred_Dist'])
                if ks_stat >= FAIL_FAST_KS:
                    logger.error(f"!!! [FAIL FAST] Distribution Shift Detected (KS={ks_stat:.4f}) !!!")
                    return False, []
                
                phase_results.append(metrics)
                gc.collect()

        return True, phase_results

    def execute(self):
        self.load_artifacts()
        
        # PHASE 1
        ok, p1_res = self.run_phase(1, n_seeds=1, n_splits=2)
        if not ok: return "Phase 1 Failed"
        
        # PHASE 2
        ok, p2_res = self.run_phase(2, n_seeds=2, n_splits=5)
        if not ok: return "Phase 2 Failed"
        
        # PHASE 3
        ok, p3_res = self.run_phase(3, n_seeds=3, n_splits=5)
        if not ok: return "Phase 3 Failed"
        
        self.final_report(p3_res)
        return "Audit Complete"

    def final_report(self, results):
        df_res = pd.DataFrame(results)
        mean_mae = df_res['Stack'].mean()
        std_mae = df_res['Stack'].std()
        mean_oof = df_res['OOF_LGB'].mean()
        gap = mean_mae - mean_oof
        
        logger.info("\n" + "="*50)
        logger.info("  FINAL AUDIT REPORT (Efficient Robust Validation)")
        logger.info("="*50)
        logger.info(f"1. Pseudo LB Avg MAE: {mean_mae:.4f}")
        logger.info(f"2. Pseudo LB Std:     {std_mae:.4f}")
        logger.info(f"3. Avg OOF-LB Gap:    {gap:.4f}")
        
        # GO / NO-GO Logic
        is_go = (mean_mae <= GO_MAE_LIMIT) and (gap <= GO_GAP_LIMIT) and (std_mae <= GO_STD_LIMIT)
        status = "HIGH" if is_go else ("MEDIUM" if mean_mae < 9.0 else "LOW")
        
        logger.info(f"Stability Rating:     {status}")
        logger.info(f"FINAL DECISION:       {'[ GO ]' if is_go else '[ NO-GO ]'}")
        logger.info("="*50 + "\n")

if __name__ == "__main__":
    check_model_contract_compliance()
    auditor = EfficientAuditor()
    status = auditor.execute()
    logger.info(f"Final Execution Status: {status}")
