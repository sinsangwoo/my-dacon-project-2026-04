import pandas as pd
import numpy as np
import logging
import os
import json
import gc
import warnings
from pathlib import Path
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from src.config import Config
from src.schema import FEATURE_SCHEMA
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
    calculate_risk_score
)
from src.intelligence import ExperimentIntelligence
from src.data_loader import (
    load_data, 
    build_base_features,
    build_features,
    SuperchargedPCAReconstructor
)
from src.trainer import Trainer
from src.explosion_inference import ExplosionInference
from src.distribution import DomainShiftAudit, FeatureStabilityFilter

# ─────────────────────────────────────────────────────────────────────────────
# [PHASE 1: SSOT CONSISTENCY]
# ─────────────────────────────────────────────────────────────────────────────

VALID_PHASES = [
    "1_data_check",
    "2_build_base",
    "2.5_drift_audit", # NEW: Mandatory Distribution Visibility
    "3_train_raw",
    "5_train_leakage_free",
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
        from src.utils import assert_artifact_exists
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
            
    with open(f"{Config.SUMMARY_DIR}/config_snapshot.json", "w") as f:
        json.dump(config_snap, f, indent=2)

def run_phase(phase, mode, smoke_test=False):
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
            
        elif phase in ['2_build_base', '2_build_raw']:
            train, test = load_data()
            if smoke_test:
                train = train.head(Config.SMOKE_ROWS); test = test.head(Config.SMOKE_ROWS)
            elif Config.MODE in ['trace', 'debug']:
                train = train.head(Config.TRACE_ROWS); test = test.head(Config.TRACE_ROWS)
            
            # [ZERO_TOLERANCE] Build isolated base features ONLY
            train_base = build_base_features(train)
            test_base = build_base_features(test)
            
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
            with open(f'{Config.PROCESSED_PATH}/train_stats.json', 'w') as f:
                json.dump(stats, f, indent=2)

        elif phase == '2.5_drift_audit':
            train_base = pd.read_pickle(f'{Config.PROCESSED_PATH}/train_base.pkl')
            test_base = pd.read_pickle(f'{Config.PROCESSED_PATH}/test_base.pkl')
            
            # Audit based on raw features (Base for all others)
            audit = DomainShiftAudit()
            drift_df = audit.calculate_drift(train_base, test_base, FEATURE_SCHEMA['raw_features'])
            audit.save_report(drift_df, f"{Config.SUMMARY_DIR}/distribution/drift_audit_raw.csv")
            
            # Filter unstable features
            filter = FeatureStabilityFilter(threshold=Config.STABILITY_THRESHOLD)
            filter.fit(drift_df)
            
            # Save stability mask
            stability_manifest = {
                "stable_features": filter.stable_features,
                "unstable_features": filter.unstable_features
            }
            with open(f'{Config.PROCESSED_PATH}/stability_manifest.json', 'w') as f:
                json.dump(stability_manifest, f, indent=2)
            
            logger.info(f"[DRIFT_AUDIT] {len(filter.stable_features)} stable features identified.")

        elif phase == '3_train_raw':
            train_base = pd.read_pickle(f'{Config.PROCESSED_PATH}/train_base.pkl')
            test_base = pd.read_pickle(f'{Config.PROCESSED_PATH}/test_base.pkl')
            y = load_npy(f'{Config.PROCESSED_PATH}/y_train.npy')
            sid = load_npy(f'{Config.PROCESSED_PATH}/scenario_id.npy', allow_pickle=True)
            
            trainer = Trainer(None, y, None, sid, full_df=train_base, test_df=test_base)
            mae, oof = trainer.fit_raw_model()
            
            save_npy(oof, f'{Config.PROCESSED_PATH}/oof_raw.npy')
            save_npy(y - oof, f'{Config.PROCESSED_PATH}/residuals_raw.npy')

        elif phase in ['5_train_leakage_free', '5_train_final']:
            from src.utils import assert_artifact_exists
            assert_artifact_exists(f'{Config.PROCESSED_PATH}/train_base.pkl', "Train Base Features")
            assert_artifact_exists(f'{Config.PROCESSED_PATH}/y_train.npy', "Train Labels")
            
            train_base = pd.read_pickle(f'{Config.PROCESSED_PATH}/train_base.pkl')
            test_base = pd.read_pickle(f'{Config.PROCESSED_PATH}/test_base.pkl')
            y = load_npy(f'{Config.PROCESSED_PATH}/y_train.npy')
            sid = load_npy(f'{Config.PROCESSED_PATH}/scenario_id.npy', allow_pickle=True)
            

            trainer = Trainer(None, y, None, sid, full_df=train_base, test_df=test_base)
            
            # [ZERO_TOLERANCE] The unified leakage-free loop
            mae_final, oof_final = trainer.fit_leakage_free_model()
            
            # [ANTI-CV-ILLUSION AUDIT]
            logger.info("--- [ANTI-CV-ILLUSION AUDIT START] ---")
            adv_auc = trainer.perform_adversarial_audit()
            corr, div = trainer.analyze_model_divergence()
            
            final_metrics = build_metrics(y, oof_final)
            risk_score = calculate_risk_score(final_metrics, adv_auc)
            
            # Save results
            train_stats = json.load(open(f'{Config.PROCESSED_PATH}/train_stats.json'))
            save_npy(oof_final, f'{Config.PREDICTIONS_PATH}/oof_stable.npy')
            save_npy(trainer.test_preds['final'], f'{Config.PREDICTIONS_PATH}/test_stable.npy')
            
            intelligence = ExperimentIntelligence()
            intelligence.log_experiment_audit(
                Config.RUN_ID, final_metrics, adv_auc, risk_score, trainer.fold_stats
            )
            
            # [DIST GUARD]
            trainer.validate_distribution(trainer.test_preds['final'], train_stats)

        elif phase == '7_inference':
            from src.utils import assert_artifact_exists
            import pickle
            
            assert_artifact_exists(f'{Config.PROCESSED_PATH}/test_base.pkl', "Test Base Features")
            test_base = pd.read_pickle(f'{Config.PROCESSED_PATH}/test_base.pkl')
            
            n_test = len(test_base)
            test_preds = np.zeros(n_test)
            
            for fold in range(Config.NFOLDS):
                logger.info(f"[INFERENCE] Processing Fold {fold}...")
                
                with open(f'{Config.MODELS_PATH}/reconstructors/recon_fold_{fold}.pkl', 'rb') as f:
                    reconstructor = pickle.load(f)
                with open(f'{Config.MODELS_PATH}/reconstructors/scaler_fold_{fold}.pkl', 'rb') as f:
                    scaler = pickle.load(f)
                with open(f'{Config.MODELS_PATH}/reconstructors/features_fold_{fold}.pkl', 'rb') as f:
                    fold_features = pickle.load(f)
                with open(f'{Config.MODELS_PATH}/lgbm/model_fold_{fold}.pkl', 'rb') as f:
                    model = pickle.load(f)
                
                # Apply features strictly via pre-computed cache
                from src.data_loader import apply_latent_features
                # [PHASE 2: UNIFIED SCALING] Scale test data before latent population
                test_base_scaled = scaler.transform(test_base, FEATURE_SCHEMA['raw_features'])
                test_df_full = apply_latent_features(test_base_scaled, reconstructor, scaler=None)
                X_test_f = test_df_full[fold_features].values.astype(np.float32)
                
                from src.utils import SAFE_PREDICT
                test_preds += SAFE_PREDICT(model, X_test_f) / Config.NFOLDS
                
                del reconstructor, scaler, model, test_df_full
                gc.collect()
            
            save_npy(test_preds, f'{Config.PREDICTIONS_PATH}/final_submission.npy')

        elif phase == '8_submission':
            final_preds = load_npy(f'{Config.PREDICTIONS_PATH}/final_submission.npy')
            sample_sub = pd.read_csv(Config.DATA_PATH + 'sample_submission.csv')
            sub, f_hash = build_submission(final_preds, sample_sub['ID'].values[:len(final_preds)])
            validate_submission(sub, sample_sub.iloc[:len(final_preds)], logger)
            logger.info(f"[DONE] Fingerprint: {f_hash}")

        elif phase == '9_intelligence':
            intelligence = ExperimentIntelligence()
            intelligence.generate_intelligence_summary(Config.RUN_ID)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, required=True)
    parser.add_argument("--mode", type=str, default="full")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    Config.FORCE_OVERWRITE = args.force
    
    if args.phase == 'all':
        for p in VALID_PHASES:
            run_phase(p, args.mode, smoke_test=args.smoke_test)
    else:
        run_phase(args.phase, args.mode, smoke_test=args.smoke_test)

if __name__ == "__main__":
    main()
