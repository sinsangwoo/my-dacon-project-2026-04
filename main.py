import pandas as pd
import numpy as np
import logging
import os
import json
import gc
import warnings
from pathlib import Path
from datetime import datetime
import argparse
import pickle
from src.trainer import (
    Trainer, 
    TailRiskController, 
    _sigmoid_gate, 
    _multi_signal_blend
)

# Suppress warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from src.config import Config
from src.schema import FEATURE_SCHEMA
from src import utils
from src.utils import (
    seed_everything, 
    get_logger, 
    save_npy,
    load_npy,
    log_forensic_snapshot,
    PhaseTracer,
    validate_submission,
    build_submission,
    calculate_std_ratio,
    build_metrics,
    calculate_risk_score,
    save_json,
    assert_artifact_exists,
    downcast_df
)
from src.data_loader import (
    load_data, 
    build_base_features, 
    build_features, 
    SuperchargedPCAReconstructor,
    get_protected_candidates,
    add_extreme_detection_features,
    add_time_series_features,
    add_scenario_context_features,
    apply_latent_features
)
from src.explosion_inference import ExplosionInference
from src.distribution import DomainShiftAudit, FeatureStabilityFilter
from src.intelligence import ExperimentIntelligence
from src.signal_validation import CollectiveDriftPruner

# ─────────────────────────────────────────────────────────────────────────────
# [PHASE 1: SSOT CONSISTENCY]
# ─────────────────────────────────────────────────────────────────────────────

VALID_PHASES = [
    "1_data_check",
    "2_build_base",
    "2.5_drift_audit", 
    "3_train_raw",
    "5_train_leakage_free",
    "6_calibrate",     
    "7_inference",
    "8_submission",
    "9_intelligence"
]

def initialize_run(phase):
    # [MISSION: ARTIFACT ISOLATION GUARD]
    # Only enforce isolation on the first phase to allow subsequent phases to add artifacts.
    is_start_phase = (phase in ["1_data_check", "2_build_base"])
    if is_start_phase and os.path.exists(Config.PROCESSED_PATH) and not Config.FORCE_OVERWRITE:
        if any(os.scandir(Config.PROCESSED_PATH)):
            msg = f"[ISOLATION_VIOLATION] Directory {Config.PROCESSED_PATH} already contains artifacts. "
            msg += "Use a unique RUN_ID or set FORCE_OVERWRITE=True to proceed."
            raise RuntimeError(msg)
    elif not is_start_phase:
        assert_artifact_exists(Config.PROCESSED_PATH, f"Run Directory for phase {phase}")
    dirs = [Config.OUTPUT_BASE, Config.PROCESSED_PATH, Config.MODELS_PATH, 
            Config.PREDICTIONS_PATH, Config.LOG_DIR, Config.SUMMARY_DIR,
            f"{Config.SUMMARY_DIR}/distribution"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    config_snap = {}
    for k, v in Config.__dict__.items():
        if k.startswith('__') or k == 'METRIC_SCHEMA': continue
        if isinstance(v, (int, float, str, bool, list, type(None))):
            config_snap[k] = v
        elif isinstance(v, dict):
            try:
                json.dumps(v)
                config_snap[k] = v
            except:
                continue
            
    save_json(config_snap, f"{Config.SUMMARY_DIR}/config_snapshot.json")

def run_phase(phase, mode, smoke_test=False):
    # [SSOT_FIX] All imports moved to top-level to prevent UnboundLocalError
    
    Config.MODE = mode
    Config.NFOLDS = Config.ADAPTIVE_FOLDS[mode]
    Config.SMOKE_TEST = smoke_test
    logger = get_logger(phase)
    
    if smoke_test:
        Config.NFOLDS = 2 # Force minimal folds
        logger.info("[SMOKE_TEST] Capping folds to 2.")
    seed_everything(42)
    
    with PhaseTracer(phase, logger) as tracer:
        initialize_run(phase)
        
        if phase == '1_data_check':
            train, test = load_data()
            if smoke_test:
                train = train.head(Config.SMOKE_ROWS); test = test.head(Config.SMOKE_ROWS)
            elif Config.MODE in ['trace', 'debug']:
                train = train.head(Config.TRACE_ROWS); test = test.head(Config.TRACE_ROWS)
            logger.info(f"[DATA] Train: {train.shape} | Test: {test.shape}")
            log_forensic_snapshot(train, "train_raw", logger)
            log_forensic_snapshot(test, "test_raw", logger)
            
        elif phase == '2_build_base':
            train, test = load_data()
            if smoke_test:
                train = train.head(Config.SMOKE_ROWS); test = test.head(Config.SMOKE_ROWS)
            elif Config.MODE in ['trace', 'debug']:
                train = train.head(Config.TRACE_ROWS); test = test.head(Config.TRACE_ROWS)
            
            # [WHY_THIS_CHANGE] Zero-Hardcode Pipeline Reconstruction
            # Problem: Phase 2 previously computed thresholds independently for train/test.
            #   This introduced data leakage as test distribution influenced test pruning.
            # Decision: Build manifest on train, apply to test. Record decisions in registry.
            # Why this approach: Ensures deterministic pruning logic across datasets.
            train_base, manifest, registry = build_base_features(train)
            test_base = build_base_features(test, pruning_manifest=manifest)

            # [WHY_THIS_CHANGE] Zero-Failure Contract
            # Ensure train and test features are perfectly synchronized before proceeding.
            # This is critical for downstream PCA and model inference consistency.
            train_cols = [c for c in train_base.columns if c not in Config.ID_COLS and c != Config.TARGET]
            test_cols = [c for c in test_base.columns if c not in Config.ID_COLS]
            if train_cols != test_cols:
                logger.error(f"[SCHEMA_MISMATCH] Train count: {len(train_cols)}, Test count: {len(test_cols)}")
                diff = set(train_cols) ^ set(test_cols)
                logger.error(f"[SCHEMA_MISMATCH] Symmetric difference: {list(diff)[:10]}")
                raise RuntimeError(f"Strict Schema Synchronization failed between Train and Test.")
            logger.info(f"[SCHEMA_SYNC] Perfect match: {len(train_cols)} features synchronized.")

            # Save audit artifacts for forensics and cross-phase consistency
            manifest.save(f'{Config.PROCESSED_PATH}/pruning_manifest.json')
            registry.save(f'{Config.SUMMARY_DIR}/feature_drop_registry.json')
            
            # Save dataframes for the fold loop
            train_base.to_pickle(f'{Config.PROCESSED_PATH}/train_base.pkl')
            test_base.to_pickle(f'{Config.PROCESSED_PATH}/test_base.pkl')
            save_npy(train[Config.TARGET].values, f'{Config.PROCESSED_PATH}/y_train.npy')
            save_npy(train['scenario_id'].values, f'{Config.PROCESSED_PATH}/scenario_id.npy')
            
            stats = {
                "mean": float(train[Config.TARGET].mean()), 
                "std": float(train[Config.TARGET].std()), 
                "p99": float(np.quantile(train[Config.TARGET], 0.99))
            }
            save_json(stats, f'{Config.PROCESSED_PATH}/train_stats.json')

        elif phase == '2.5_drift_audit':
            train_base = pd.read_pickle(f'{Config.PROCESSED_PATH}/train_base.pkl')
            test_base = pd.read_pickle(f'{Config.PROCESSED_PATH}/test_base.pkl')
            
            # [MISSION 1] SCA-Based Dynamic Drift Audit
            # [WHY] Replaces fixed 0.05/0.10 thresholds with Signal Curvature Analysis.
            # [FIX] Automatically detects inflection point where drift signal accelerates.
            audit = DomainShiftAudit()
            drift_df, dynamic_ks_threshold = audit.calculate_drift(train_base, test_base, FEATURE_SCHEMA['raw_features'])
            audit.save_report(drift_df, f"{Config.SUMMARY_DIR}/distribution/drift_audit_raw.csv")
            
            # [MISSION 2] Lineage-Based Conditional Protection
            # [WHY] Prevents drift amplification from derivatives of shifted base sensors.
            # [FIX] Prunes children if parents are drifty (using dynamic_ks_threshold).
            # [SSOT_FIX] Local import removed
            protected_cols = get_protected_candidates(train_base.columns, drift_df=drift_df, ks_threshold=dynamic_ks_threshold)
            
            # [MISSION 3] Collective Drift Iterative Pruning
            # [WHY] Individually stable features (KS < threshold) can collectively reach AUC 0.95.
            # [FIX] Prune top contributors until ADV AUC < 0.75.
            pruner = CollectiveDriftPruner(target_auc=Config.ADV_TARGET_AUC, max_iterations=10, prune_step=3)
            
            # Use only features that are not already drifted by KS
            initial_features = drift_df[drift_df['ks_stat'] <= dynamic_ks_threshold]['feature'].tolist()
            # Ensure we only use features actually in train_base
            initial_features = [f for f in initial_features if f in train_base.columns]
            
            stable_features, history = pruner.prune(train_base, test_base, initial_features, protected_cols=protected_cols)
            save_json(history, f'{Config.SUMMARY_DIR}/distribution/iterative_pruning_history.json')
            
            # Final Manifest
            # Features are stable if they survived iterative pruning OR if they are structural BASE_COLS
            final_stable = list(set(stable_features) | protected_cols)
            final_unstable = list(set(train_base.columns) - set(final_stable))
            
            stability_manifest = {
                "dynamic_ks_threshold": dynamic_ks_threshold,
                "stable_features": final_stable,
                "unstable_features": final_unstable
            }
            save_json(stability_manifest, f'{Config.PROCESSED_PATH}/stability_manifest.json')
            
            logger.info(f"[DRIFT_AUDIT] SCA Threshold: {dynamic_ks_threshold:.4f}")
            logger.info(f"[DRIFT_AUDIT] Collective Pruning: {len(initial_features)} -> {len(stable_features)} features.")
            logger.info(f"[DRIFT_AUDIT] Total Stable: {len(final_stable)} identified.")

        elif phase == '3_train_raw':
            train_base = pd.read_pickle(f'{Config.PROCESSED_PATH}/train_base.pkl')
            test_base = pd.read_pickle(f'{Config.PROCESSED_PATH}/test_base.pkl')
            y = load_npy(f'{Config.PROCESSED_PATH}/y_train.npy')
            sid = load_npy(f'{Config.PROCESSED_PATH}/scenario_id.npy', allow_pickle=True)
            
            trainer = Trainer(None, y, None, sid, full_df=train_base, test_df=test_base)
            mae, oof = trainer.fit_raw_model()
            
            save_npy(oof, f'{Config.PROCESSED_PATH}/oof_raw.npy')
            save_npy(y - oof, f'{Config.PROCESSED_PATH}/residuals_raw.npy')

        elif phase == '5_train_leakage_free':
            # [SSOT_FIX] Local import removed
            assert_artifact_exists(f'{Config.PROCESSED_PATH}/train_base.pkl', "Train Base Features")
            assert_artifact_exists(f'{Config.PROCESSED_PATH}/y_train.npy', "Train Labels")
            
            train_base = downcast_df(pd.read_pickle(f'{Config.PROCESSED_PATH}/train_base.pkl'))
            test_base = downcast_df(pd.read_pickle(f'{Config.PROCESSED_PATH}/test_base.pkl'))
            y = load_npy(f'{Config.PROCESSED_PATH}/y_train.npy')
            sid = load_npy(f'{Config.PROCESSED_PATH}/scenario_id.npy', allow_pickle=True)
            
            # [SSOT] Load dynamic drift metadata from Phase 2.5
            stability_manifest = json.load(open(f'{Config.PROCESSED_PATH}/stability_manifest.json'))
            dynamic_ks_threshold = stability_manifest.get("dynamic_ks_threshold", 0.15)
            
            try:
                drift_df = pd.read_csv(f"{Config.SUMMARY_DIR}/distribution/drift_audit_raw.csv")
            except:
                drift_df = None

            trainer = Trainer(None, y, None, sid, full_df=train_base, test_df=test_base, 
                              drift_df=drift_df, ks_threshold=dynamic_ks_threshold,
                              manifest=stability_manifest)
            
            # [ZERO_TOLERANCE] The unified leakage-free loop
            mae_final, oof_final = trainer.fit_leakage_free_model()
            
            # [ANTI-CV-ILLUSION AUDIT]
            logger.info("--- [ANTI-CV-ILLUSION AUDIT START] ---")
            adv_auc = trainer.perform_adversarial_audit()
            
            final_metrics = build_metrics(y, oof_final, y_base=trainer.oof_base)
            risk_results = calculate_risk_score(final_metrics, adv_auc, fold_maes=trainer.fold_scores)
            risk_score = risk_results['total']
            
            # [ARTIFACT_HANDOVER] Save raw results for Phase 6 (Calibration)
            save_npy(oof_final, f'{Config.PREDICTIONS_PATH}/oof_raw.npy')
            save_npy(trainer.oof_base, f'{Config.PREDICTIONS_PATH}/oof_base_raw.npy')
            save_npy(trainer.test_preds['final'], f'{Config.PREDICTIONS_PATH}/test_raw.npy')
            
            # Save logs and initial audit
            intelligence = ExperimentIntelligence()
            intelligence.log_experiment_audit(
                Config.RUN_ID, final_metrics, adv_auc, risk_results, trainer.fold_scores, metadata=trainer.metadata
            )
            
            logger.info("[PHASE_SUCCESS] 5_train_leakage_free - Raw artifacts saved for calibration.")
            
            # [MEMORY_OPTIMIZATION] Clean up heavy objects
            del trainer, oof_final, y, sid, train_base, test_base, drift_df, stability_manifest
            gc.collect()
            
        elif phase == '6_calibrate':
            # [DYNAMIC_BIAS_RECOVERY] 
            # Why: Separated from training to allow post-training recalibration.
            y = load_npy(f'{Config.PROCESSED_PATH}/y_train.npy')
            oof_raw = load_npy(f'{Config.PREDICTIONS_PATH}/oof_raw.npy')
            oof_base_raw = load_npy(f'{Config.PREDICTIONS_PATH}/oof_base_raw.npy')
            test_raw = load_npy(f'{Config.PREDICTIONS_PATH}/test_raw.npy')
            
            valid_mask = ~np.isnan(oof_raw)
            oof_mean = np.mean(oof_raw[valid_mask])
            true_mean = np.mean(y[valid_mask])
            oof_std = np.std(oof_raw[valid_mask])
            true_std = np.std(y[valid_mask])
            
            mean_scalar = float(true_mean / (oof_mean + 1e-9))
            std_scalar = float(true_std / (oof_std + 1e-9))
            
            # [CONTROLLED AGGRESSION] Limit variance recovery to prevent instability
            if Config.VARIANCE_RECOVERY_ENABLED:
                std_scalar = min(Config.STD_SCALAR_CAP, std_scalar)
            else:
                std_scalar = 1.0
                
            logger.info(f"[CALIBRATION] OOF Mean: {oof_mean:.4f} | True Mean: {true_mean:.4f}")
            logger.info(f"[CALIBRATION] OOF Std: {oof_std:.4f} | True Std: {true_std:.4f}")
            logger.info(f"[CALIBRATION] Mean Scalar: {mean_scalar:.4f} | Std Scalar: {std_scalar:.4f}")
            
            # Apply and save: oof_recovered = (oof - oof_mean) * std_scalar + true_mean
            oof_stable = (oof_raw - oof_mean) * std_scalar + true_mean
            test_stable = (test_raw - oof_mean) * std_scalar + true_mean
            
            save_npy(oof_stable, f'{Config.PREDICTIONS_PATH}/oof_stable.npy')
            save_npy(test_stable, f'{Config.PREDICTIONS_PATH}/test_stable.npy')
            
            # [MISSION 9: INTELLIGENCE_SYNC] Update registry with calibrated metrics
            # [FORENSIC FIX] Pass y_base to enable Gain Capture tracking
            stable_metrics = build_metrics(y, oof_stable, y_base=oof_base_raw)
            
            # Recalculate risk score using calibrated metrics and original adv_auc
            intelligence = ExperimentIntelligence()
            prev_run = next((r for r in intelligence.registry["runs"] if r["run_id"] == Config.RUN_ID), {})
            adv_auc = prev_run.get("adv_auc", 0.5)
            
            risk_results = calculate_risk_score(stable_metrics, adv_auc, fold_maes=prev_run.get("fold_stats", []))
            risk_score = risk_results['total']
            
            intelligence.log_experiment_audit(
                Config.RUN_ID, 
                stable_metrics, 
                adv_auc, 
                risk_results, 
                fold_stats=prev_run.get("fold_stats", []),
                metadata=prev_run.get("metadata", {})
            )
            logger.info(f"[SYNC] Registry updated with calibrated MAE: {stable_metrics['mae']:.4f}")
            # [FORENSIC #4] Phase 6 saves detailed calibration metadata for Phase 7
            save_json({
                "mean_scalar": mean_scalar,
                "std_scalar": std_scalar,
                "oof_mean": float(oof_mean),
                "true_mean": float(true_mean)
            }, f'{Config.LOG_DIR}/calibration.json')
            save_json(risk_results, f'{Config.LOG_DIR}/risk_breakdown.json')
            
            
            # Final distribution validation
            # [SSOT_FIX] Local import removed
            train_stats = json.load(open(f'{Config.PROCESSED_PATH}/train_stats.json'))
            temp_trainer = Trainer(None, y, None, None)
            temp_trainer.validate_distribution(test_stable, train_stats)
            
            logger.info("[PHASE_SUCCESS] 6_calibrate - Dynamic scaling complete.")

        elif phase == '7_inference':
            assert_artifact_exists(f'{Config.PROCESSED_PATH}/test_base.pkl', "Test Base Features")
            test_base = pd.read_pickle(f'{Config.PROCESSED_PATH}/test_base.pkl')
            
            n_test = len(test_base)
            test_preds = np.zeros(n_test)
            
            # [FORENSIC #3] Load calibration metadata — SOLE bias correction path
            try:
                calib_data = json.load(open(f'{Config.LOG_DIR}/calibration.json'))
                oof_mean_cal = calib_data.get("oof_mean", 0.0)
                true_mean_cal = calib_data.get("true_mean", 0.0)
                std_scalar_cal = calib_data.get("std_scalar", 1.0)
            except:
                logger.warning("[INFERENCE] Calibration metadata missing. Using defaults.")
                oof_mean_cal, true_mean_cal, std_scalar_cal = 0.0, 0.0, 1.0
            logger.info(f"[INFERENCE] Calibration: MeanOffset={true_mean_cal-oof_mean_cal:.4f}, StdScalar={std_scalar_cal:.4f}")
            
            for fold in range(Config.NFOLDS):
                logger.info(f"[INFERENCE] Processing Fold {fold}...")
                
                with open(f'{Config.MODELS_PATH}/reconstructors/recon_fold_{fold}.pkl', 'rb') as f: reconstructor = pickle.load(f)
                with open(f'{Config.MODELS_PATH}/reconstructors/scaler_fold_{fold}.pkl', 'rb') as f: scaler = pickle.load(f)
                with open(f'{Config.MODELS_PATH}/reconstructors/features_fold_{fold}.pkl', 'rb') as f: fold_features = pickle.load(f)
                with open(f'{Config.MODELS_PATH}/reconstructors/norm_scaler_fold_{fold}.pkl', 'rb') as f: norm_scaler = pickle.load(f)
                with open(f'{Config.MODELS_PATH}/reconstructors/layout_stats_fold_{fold}.pkl', 'rb') as f: layout_stats = pickle.load(f)
                with open(f'{Config.MODELS_PATH}/reconstructors/bucket_edges_fold_{fold}.pkl', 'rb') as f: bucket_edges = pickle.load(f)
                with open(f'{Config.MODELS_PATH}/reconstructors/risk_ctrl_fold_{fold}.pkl', 'rb') as f: risk_ctrl = pickle.load(f)
                with open(f'{Config.MODELS_PATH}/lgbm/model_fold_{fold}.pkl', 'rb') as f: model = pickle.load(f)
                
                # [FORENSIC #13] Load train-derived column means to prevent test distribution leakage
                try:
                    with open(f'{Config.MODELS_PATH}/reconstructors/train_col_means_fold_{fold}.pkl', 'rb') as f:
                        train_col_means = pickle.load(f)
                except FileNotFoundError:
                    logger.warning(f"[INFERENCE] train_col_means_fold_{fold}.pkl not found. Falling back to None (test leakage risk).")
                    train_col_means = None
                
                # [FORENSIC #14] Load train-derived extreme quantiles
                try:
                    with open(f'{Config.MODELS_PATH}/reconstructors/extreme_quantiles_fold_{fold}.pkl', 'rb') as f:
                        extreme_quantiles = pickle.load(f)
                except FileNotFoundError:
                    logger.warning(f"[INFERENCE] extreme_quantiles_fold_{fold}.pkl not found. Falling back to None.")
                    extreme_quantiles = None

                # Apply fold-specific transformations
                test_fold = test_base.copy()
                for feat, mapping in layout_stats.items():
                    if not (feat.endswith("_layout_mean") or feat.endswith("_layout_std")):
                        continue
                    
                    base_col = feat.replace("_layout_mean", "").replace("_layout_std", "")
                    stat_type = "mean" if "mean" in feat else "std"
                    
                    # [GLOBAL_FALLBACK] Handle new layouts (40% of test set)
                    fallback = layout_stats.get(f"{base_col}_global_{stat_type}", 0.0)
                    test_fold[feat] = test_fold['layout_id'].map(mapping).fillna(fallback)
                
                # [FORENSIC #13] Pass train_col_means to prevent test leakage
                test_fold = add_time_series_features(test_fold, train_col_means=train_col_means)
                test_fold = add_scenario_context_features(test_fold)
                # [FORENSIC #14] Pass extreme_quantiles to prevent test leakage
                test_fold = add_extreme_detection_features(test_fold, extreme_quantiles=extreme_quantiles, bucket_edges=bucket_edges)
                
                # Apply scaling
                raw_cols = list(norm_scaler.feature_names_in_)
                test_fold_drifted = scaler.transform(test_fold, raw_cols)
                test_fold_scaled = test_fold_drifted.copy()
                test_fold_scaled[raw_cols] = norm_scaler.transform(test_fold_drifted[raw_cols])
                
                test_df_full = apply_latent_features(test_fold_scaled, reconstructor, scaler=None, selected_features=fold_features, is_train=False)
                X_test_f = test_df_full[fold_features].values.astype(np.float32)
                
                if Config.USE_2STAGE_MODEL:
                    # [ARCHITECTURAL RECONSTRUCTION] Feature Set Synchronization
                    clf_features = model.get("clf_features", [f for f in fold_features if f not in getattr(Config, 'CLASSIFIER_POISON_FEATURES', [])])
                    tail_features = model.get("tail_features", [f for f in fold_features if f not in getattr(Config, 'TAIL_DESTRUCTIVE_FEATURES', [])])
                    
                    X_test_clf = test_df_full[clf_features].values.astype(np.float32)
                    X_test_tail = test_df_full[tail_features].values.astype(np.float32)
                    
                    p_te = utils.SAFE_PREDICT_PROBA(model["clf"], X_test_clf)[:, 1]
                    preds_t = np.expm1(utils.SAFE_PREDICT(model["tail"], X_test_tail))
                    preds_nt = np.expm1(utils.SAFE_PREDICT(model["non_tail"], X_test_f))
                    
                    # [ARCHITECTURAL RECONSTRUCTION: MULTI-SIGNAL GATING]
                    p_te_sharpened = _multi_signal_blend(p_te, preds_t, preds_nt)
                    
                    gap_te = risk_ctrl.compute_gap(preds_t, preds_nt)
                    p_te_sharpened = risk_ctrl.apply(p_te_sharpened, p_te, gap_te)
                    
                    # Combine
                    fold_preds = p_te_sharpened * preds_t + (1.0 - p_te_sharpened) * preds_nt
                    
                    # [FORENSIC #3] Apply calibration formula: (pred - oof_mean) * std_scalar + true_mean
                    test_preds += ((fold_preds - oof_mean_cal) * std_scalar_cal + true_mean_cal) / Config.NFOLDS
                else:
                    fold_preds_raw = np.expm1(utils.SAFE_PREDICT(model, X_test_f))
                    test_preds += ((fold_preds_raw - oof_mean_cal) * std_scalar_cal + true_mean_cal) / Config.NFOLDS
                
                del reconstructor, scaler, norm_scaler, model, test_df_full, test_fold, test_fold_drifted, test_fold_scaled, X_test_f
                gc.collect()
            
            save_npy(test_preds, f'{Config.PREDICTIONS_PATH}/final_submission.npy')
            
            # [MEMORY_OPTIMIZATION] Clean up heavy objects
            del test_preds
            gc.collect()

        elif phase == '8_submission':
            final_preds = load_npy(f'{Config.PREDICTIONS_PATH}/final_submission.npy')
            sample_sub = pd.read_csv(os.path.join(Config.DATA_PATH, 'sample_submission.csv'))
            sub, f_hash = build_submission(final_preds, sample_sub['ID'].values[:len(final_preds)])
            validate_submission(sub, sample_sub.iloc[:len(final_preds)], logger)
            logger.info(f"[DONE] Fingerprint: {f_hash}")

        elif phase == '9_intelligence':
            intelligence = ExperimentIntelligence()
            intelligence.generate_intelligence_summary(Config.RUN_ID)

def main():
    # [SSOT_FIX] Local import removed
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, required=True)
    parser.add_argument("--mode", type=str, default="full")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()
    
    Config.FORCE_OVERWRITE = args.force
    if args.run_id:
        Config.rebuild_paths(args.run_id)
    
    if args.phase == 'all':
        for p in VALID_PHASES:
            run_phase(p, args.mode, smoke_test=args.smoke_test)
    else:
        run_phase(args.phase, args.mode, smoke_test=args.smoke_test)

if __name__ == "__main__":
    main()
