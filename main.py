import pandas as pd
import numpy as np
import logging
import argparse
import os
import json
import sys
import gc
import warnings
import glob
import time
import hashlib
from pathlib import Path

# Suppress PerformanceWarning after fixing the root cause (Batch Concat)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
from src.forensic_logger import ForensicLogger
from src.config import Config
from src.utils import (
    seed_everything, 
    get_logger, 
    save_pkl, 
    load_pkl, 
    save_npy,
    load_npy,
    log_forensic_snapshot,
    log_memory_usage,
    get_process_memory,
    PhaseTracer,
    memory_guard,
    GLOBAL_PEAK_MEMORY,
    inspect_columns,
    ensure_dataframe,
    validate_submission,
    generate_submission_fingerprint,
    save_submission_trace,
    preserve_fail_case,
    build_submission,
    SAFE_FIT,
    SAFE_PREDICT,
    check_model_contract_compliance
)
from src.explosion_inference import ExplosionInference, run_explosion_inference
from src.data_loader import (
    load_data, 
    select_top_ts_features, 
    build_causal_feature_manifest,
    get_removed_leaky_feature_patterns,
    add_time_series_features, 
    add_extreme_detection_features, 
    add_advanced_predictive_features,
    add_scenario_summary_features,
    add_scenario_sequence_compressed,
    add_sequence_trajectory_features,
    compress_sequence_features,
    add_binary_thresholding,
    add_nan_flags,
    handle_engineered_nans,
    get_features,
    align_features,
    prune_collinear_features,
    prune_drift_features,
    add_hybrid_latent_features,
    generate_supercharged_latent_features
)
from src.trainer import Trainer
from src.intelligence import ExperimentIntelligence
from sklearn.metrics import mean_absolute_error

# --- [PHASE 1: PHASE ENUM EXTRACTION (SSOT)] ---
# Purpose: Define the canonical execution graph for the pipeline
# Failure: Strictly enforced at argparse and runtime entry
VALID_PHASES = [
    "1_data_check",
    "2_build_raw",
    "3_train_raw",
    "4_build_full",
    "5_train_final",
    "6_retrain",
    "7_inference",
    "8_submission",
    "9_intelligence",
    "validate_artifacts",
    "verify_paths"
]

LEGACY_PHASES = ["2_preprocess", "3_train_base", "4_stacking", "5_pseudo_labeling"]

def log_nan_stats(df, stage, logger):
    # PURPOSE: Log missing value statistics for forensic tracking
    # INPUT: df (DataFrame), stage (str), logger (logging.Logger)
    # OUTPUT: total_nans (int)
    # FAILURE: Returns 0 if df is empty
    nan_counts = df.isna().sum()
    total_nans = nan_counts.sum()
    logger.info(f"[NaN Trace] Stage: {stage} | Total NaNs: {total_nans:,}")
    if total_nans > 0:
        top_nans = nan_counts[nan_counts > 0].sort_values(ascending=False).head(10)
        logger.info(f"[NaN Trace] Top problematic columns:\n{top_nans}")
    return total_nans

def validate_consistency(logger):
    # PURPOSE: Ensure RUN_ID and output paths are synchronized
    # INPUT: logger (logging.Logger)
    # OUTPUT: None
    # FAILURE: Raises ValueError if RUN_ID mismatch detected
    if not Config.RUN_ID:
        raise ValueError("[FATAL] RUN_ID is empty or not set.")
    expected_base = f"./outputs/{Config.RUN_ID}"
    if Config.OUTPUT_BASE != expected_base:
        raise ValueError(f"[FATAL] Config path mismatch! RUN_ID={Config.RUN_ID} but OUTPUT_BASE={Config.OUTPUT_BASE}")
    logger.info(f"[VALIDATE] Consistency Check Passed: {Config.RUN_ID} aligned with filesystem.")

def verify_path_permissions(logger):
    # PURPOSE: Pre-flight check for directory write permissions
    # INPUT: logger (logging.Logger)
    # OUTPUT: bool (True if all paths writable)
    # FAILURE: Returns False if any path is access denied
    paths_to_check = [Config.OUTPUT_BASE, Config.LOG_DIR, Config.SUMMARY_DIR]
    for p in paths_to_check:
        parent = os.path.dirname(p.rstrip('/'))
        if not os.path.exists(p):
            if os.path.exists(parent):
                if os.access(parent, os.W_OK):
                    logger.info(f"[DRY_RUN] Path {p} is writable (parent exists & writable)")
                else:
                    logger.error(f"[DRY_RUN] Path {p} is NOT writable (parent {parent} access denied)")
                    return False
            else:
                logger.warning(f"[DRY_RUN] Parent {parent} does not exist, assuming creation will work if base is ./outputs")
        else:
            if os.access(p, os.W_OK):
                logger.info(f"[DRY_RUN] Path {p} is writable")
            else:
                logger.error(f"[DRY_RUN] Path {p} access denied")
                return False
    return True

class ArtifactValidator:
    """Upgraded Artifact Validator (v10.0 - Health Checks)."""
    @staticmethod
    def check_phase_outputs(phase, logger):
        # PURPOSE: Verify physical existence and health of phase artifacts
        # INPUT: phase (str), logger (logging.Logger)
        # OUTPUT: bool (True if artifacts healthy)
        # FAILURE: Returns False if CRITICAL artifact missing or corrupt
        if Config.DRY_RUN:
            logger.info(f"[DRY_RUN] Skipping physical artifact check for {phase}. Structure validated.")
            return True
        if phase not in Config.ARTIFACT_MANIFEST:
            logger.info(f"[VALIDATE] Phase {phase} has no registered artifact contract.")
            return True
        manifest = Config.ARTIFACT_MANIFEST[phase]
        success = True
        search_paths = [Config.PROCESSED_PATH, Config.PREDICTIONS_PATH, Config.OUTPUT_BASE, Config.SUMMARY_DIR]
        for item in manifest:
            pattern = item["pattern"]; severity = item["severity"]; desc = item["desc"]
            found = False
            for base in search_paths:
                full_pattern = os.path.join(base, pattern)
                matches = glob.glob(full_pattern)
                for f in matches:
                    found = True
                    # NEW: Health Checks (Size & JSON)
                    size = os.path.getsize(f)
                    if size == 0:
                        logger.error(f"[HEALTH_FAILED] {f} exists but is EMPTY (0 bytes)")
                        if severity == "CRITICAL": success = False
                        continue
                    
                    if f.endswith('.json'):
                        try:
                            with open(f, 'r') as jf: json.load(jf)
                        except Exception as e:
                            logger.error(f"[HEALTH_FAILED] {f} is CORRUPT (JSON parse error): {str(e)}")
                            if severity == "CRITICAL": success = False
                            continue

                    logger.info(f"[VALIDATE_OK] {severity} artifact: {os.path.basename(f)} ({size/1024:.1f} KB)")
                if found: break
                    
            if not found:
                msg = f"Missing {severity} artifact: {pattern} ({desc}) after {phase}"
                if severity == "CRITICAL":
                    logger.error(f"[FATAL] {msg}"); success = False
                else: logger.warning(f"[WARNING] {msg}")
        return success

def validate_shapes(train, test, logger):
    # PURPOSE: Ensure feature symmetry between train and test datasets
    # INPUT: train (DataFrame), test (DataFrame), logger (logging.Logger)
    # OUTPUT: None
    # FAILURE: Raises RuntimeError if test missing features required by train
    train_cols = [c for c in train.columns if c not in [Config.TARGET] + Config.ID_COLS]
    test_cols = [c for c in test.columns if c not in Config.ID_COLS]
    missing_in_test = set(train_cols) - set(test_cols)
    extra_in_test = set(test_cols) - set(train_cols)
    if missing_in_test: raise RuntimeError(f"[SHAPE_FATAL] Test data missing features: {list(missing_in_test)[:10]}...")
    if extra_in_test: logger.warning(f"[SHAPE_WARNING] Test data has extra features that will be ignored: {list(extra_in_test)[:10]}...")
    logger.info(f"[SHAPE_CHECK] Symmetry Verified: {len(train_cols)} features aligned.")

def validate_output_len(preds, expected_len, label):
    # PURPOSE: Verify prediction array length matches expectations
    # INPUT: preds (ndarray), expected_len (int), label (str)
    # OUTPUT: None
    # FAILURE: Raises RuntimeError if length mismatch detected
    if len(preds) != expected_len: raise RuntimeError(f"[INFERENCE_FATAL] {label} shape mismatch! Expected {expected_len}, got {len(preds)}")

def export_metric(mae, filename="current_mae.txt"):
    # PURPOSE: Save performance metrics to disk for cross-phase access
    # INPUT: mae (float), filename (str)
    # OUTPUT: None
    # FAILURE: Creates log directory if missing
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    with open(f"{Config.LOG_DIR}/{filename}", "w") as f: f.write(f"{mae:.6f}")

def initialize_run_directories():
    # PURPOSE: Create canonical directory structure for a new run
    # INPUT: None
    # OUTPUT: None
    # FAILURE: Logs directory creation in dry_run mode
    dirs = [Config.OUTPUT_BASE, Config.PROCESSED_PATH, Config.MODELS_PATH, Config.PREDICTIONS_PATH, Config.LOG_DIR, Config.SUMMARY_DIR, Config.DONE_DIR]
    for d in dirs:
        if Config.DRY_RUN: logging.getLogger(__name__).info(f"[DRY_RUN] Would create directory: {d}")
        else: os.makedirs(d, exist_ok=True)
    
    # Filter for JSON serialization (v13.1: skip classmethods/methods/types)
    config_dict = {}
    for k, v in Config.__dict__.items():
        if k.startswith('__'): continue
        if k == 'METRIC_SCHEMA': continue # Skip the schema dictionary containing types
        if isinstance(v, (int, float, str, bool, list, dict, type(None))):
            config_dict[k] = v
            
    save_json(config_dict, f"{Config.SUMMARY_DIR}/config_snapshot.json")

def save_json(data, path):
    # PURPOSE: Atomically save dictionary to JSON file
    # INPUT: data (dict), path (str)
    # OUTPUT: None
    # FAILURE: Ensures parent directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)

def read_optional_metric(path):
    # PURPOSE: Read floating point metric from file if it exists
    # INPUT: path (str)
    # OUTPUT: float or None
    # FAILURE: Returns None if file missing or empty
    if not os.path.exists(path): return None
    with open(path, "r", encoding="utf-8") as f: raw = f.read().strip()
    return float(raw) if raw else None

def enforce_variance_policy(summary, label):
    # PURPOSE: Reject models that collapse to zero-variance predictions
    # INPUT: summary (dict), label (str)
    # OUTPUT: None
    # FAILURE: Raises RuntimeError if variance ratio below rejection threshold
    ratio = summary.get("variance_ratio")
    if ratio is None: return
    if ratio < Config.PRED_VARIANCE_RATIO_REJECT: raise RuntimeError(f"[MODEL_REJECT] {label} variance ratio {ratio:.4f} < {Config.PRED_VARIANCE_RATIO_REJECT:.2f}")
    if ratio < Config.PRED_VARIANCE_RATIO_TARGET_MIN: logging.getLogger(__name__).warning(f"[VARIANCE_WARNING] {label} variance ratio {ratio:.4f} < target {Config.PRED_VARIANCE_RATIO_TARGET_MIN:.2f}")

def validate_artifacts(phase):
    # PURPOSE: [PHASE 3] Hard validation of required input artifacts for each phase
    # INPUT: phase (str)
    # OUTPUT: None
    # FAILURE: Raises FileNotFoundError or RuntimeError if contract violated
    """Contract Enforcement: Verify all required artifacts exist for the phase (v13.2)."""
    # Helper to clean paths
    def p(path): return os.path.normpath(path)
    
    artifacts = {
        '2_build_raw': [
            (p(f"{Config.DATA_PATH}/train.csv"), "Phase 1 (Data)"),
            (p(f"{Config.DATA_PATH}/test.csv"), "Phase 1 (Data)")
        ],
        '3_train_raw': [
            (p(f"{Config.PROCESSED_PATH}/X_train_raw.npy"), "Phase 2"),
            (p(f"{Config.PROCESSED_PATH}/X_test_raw.npy"), "Phase 2"),
            (p(f"{Config.PROCESSED_PATH}/y_train.npy"), "Phase 2"),
            (p(f"{Config.PROCESSED_PATH}/scenario_id.npy"), "Phase 2")
        ],
        '4_build_full': [
            (p(f"{Config.PROCESSED_PATH}/residuals_raw.npy"), "Phase 3"),
            (p(f"{Config.PROCESSED_PATH}/oof_raw.npy"), "Phase 3")
        ],
        '5_train_final': [
            (p(f"{Config.PROCESSED_PATH}/X_train_full.npy"), "Phase 4"),
            (p(f"{Config.PROCESSED_PATH}/X_test_full.npy"), "Phase 4"),
            (p(f"{Config.PROCESSED_PATH}/regime_proxy_tr.npy"), "Phase 4"),
            (p(f"{Config.PROCESSED_PATH}/regime_proxy_te.npy"), "Phase 4")
        ],
        '7_inference': [
            (p(f"{Config.PREDICTIONS_PATH}/oof_stable.npy"), "Phase 5"),
            (p(f"{Config.PREDICTIONS_PATH}/test_stable.npy"), "Phase 5")
        ],
        '8_submission': [
            (p(f"{Config.PREDICTIONS_PATH}/final_submission.npy"), "Phase 7")
        ]
    }
    
    if phase in artifacts:
        for path, source in artifacts[phase]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"[CONTRACT_VIOLATION] Phase {phase} requires {path} from {source}, but it is missing!")
            # NEW: Size Check for NPY files
            if path.endswith('.npy'):
                if os.path.getsize(path) < 100: # Bare minimum for header
                    raise RuntimeError(f"[CONTRACT_VIOLATION] {path} is suspiciously small. Possible corruption.")

def run_phase(phase, mode, smoke_test=False):
    # PURPOSE: Execute a specific pipeline phase with strict mode and contract enforcement
    # INPUT: phase (str), mode (str), smoke_test (bool)
    # OUTPUT: None
    # FAILURE: Raises RuntimeError if phase fails or contract violated
    """Standardized Phase Execution (v8.1)."""
    
    # [PHASE 7: BACKWARD COMPATIBILITY BLOCK]
    if phase in LEGACY_PHASES:
        print(f"\n[FATAL] Legacy phase '{phase}' detected. Use new pipeline phases.")
        sys.exit(1)
        
    if phase not in VALID_PHASES:
        print(f"\n[FATAL] Invalid phase '{phase}' blocked. VALID_PHASES: {VALID_PHASES}")
        sys.exit(1)

    Config.MODE = mode
    Config.NFOLDS = Config.ADAPTIVE_FOLDS[mode]
    Config.SEEDS = [42, 43, 44, 45, 46] if mode == 'full' else [42]
    
    validate_artifacts(phase)

    
    # Standardize Logger Fix (v10.0)
    forensic_mode = 'debug' if mode in ['debug', 'trace'] else 'lite'
    forensic = ForensicLogger(phase, mode=forensic_mode)
    logger = get_logger(phase, level=Config.LOG_LEVEL)
    
    if mode == 'trace' or smoke_test: 
        Config.TRACE_MODE = True
        Config.NFOLDS = 2
        Config.SEEDS = [42]
        Config.DEBUG_MINIMAL = True
        Config.DEBUG_MINIMAL_ROWS = 20 if smoke_test else Config.TRACE_ROWS
        logger.info(f"[SMOKE_TEST] Activated: ROWS={Config.DEBUG_MINIMAL_ROWS}, FOLDS={Config.NFOLDS}")

    seed_everything(42)
    with PhaseTracer(phase, logger) as tracer:
        initialize_run_directories()
        if phase not in ['verify_paths', 'validate_artifacts']: validate_consistency(logger)

        if phase == '1_data_check':
            tracer.checkpoint("loading_raw")
            train, test = load_data()
            if Config.TRACE_MODE or Config.DEBUG_MINIMAL:
                rows = Config.TRACE_ROWS if Config.TRACE_MODE else Config.DEBUG_MINIMAL_ROWS
                train = train.head(rows); test = test.head(rows)
            log_forensic_snapshot(train, "train_raw", logger); log_forensic_snapshot(test, "test_raw", logger)
            save_npy(train.head(100).values, f'{Config.LOG_DIR}/train_raw.npy') # For artifact check
            del train, test; gc.collect()

        elif phase == '2_build_raw':
            tracer.checkpoint("load_data")
            train, test = load_data()
            if Config.TRACE_MODE or Config.DEBUG_MINIMAL:
                rows = Config.TRACE_ROWS if Config.TRACE_MODE else Config.DEBUG_MINIMAL_ROWS
                train = train.head(rows); test = test.head(rows)
            validate_shapes(train, test, logger)
            
            # [PHASE 2: DETERMINISTIC FEATURE BUILDER]
            schema = Config.get_feature_schema()
            from src.data_loader import build_features
            
            X_train_raw, train = build_features(train, schema, mode='raw')
            X_test_raw, test = build_features(test, schema, mode='raw')
            
            # Save Raw Feature Matrices
            save_npy(X_train_raw.values, f'{Config.PROCESSED_PATH}/X_train_raw.npy')
            save_npy(X_test_raw.values, f'{Config.PROCESSED_PATH}/X_test_raw.npy')
            save_npy(train[Config.TARGET].values, f'{Config.PROCESSED_PATH}/y_train.npy')
            save_npy(train['scenario_id'].values, f'{Config.PROCESSED_PATH}/scenario_id.npy')
            
            # Save stats for later
            save_json({
                "mean": float(train[Config.TARGET].mean()), 
                "std": float(train[Config.TARGET].std()), 
                "p99": float(np.quantile(train[Config.TARGET], 0.99))
            }, f'{Config.PROCESSED_PATH}/train_stats.json')
            
            del train, test; gc.collect()

        elif phase == '3_train_raw':
            # [PHASE 3: STAGE A - RAW TRAINING]
            schema = Config.get_feature_schema()
            X_train_raw = load_npy(f'{Config.PROCESSED_PATH}/X_train_raw.npy')
            X_test_raw = load_npy(f'{Config.PROCESSED_PATH}/X_test_raw.npy')
            y_train = load_npy(f'{Config.PROCESSED_PATH}/y_train.npy')
            scenario_id = load_npy(f'{Config.PROCESSED_PATH}/scenario_id.npy', allow_pickle=True)
            
            trainer = Trainer(X_train_raw, y_train, X_test_raw, schema, scenario_id)
            
            # Train Raw Model
            raw_mae, oof_raw = trainer.fit_raw_model()
            
            # Generate Residuals
            residuals_raw = y_train - oof_raw
            
            # Save Stage A Artifacts
            save_npy(oof_raw, f'{Config.PROCESSED_PATH}/oof_raw.npy')
            save_npy(residuals_raw, f'{Config.PROCESSED_PATH}/residuals_raw.npy')
            save_npy(trainer.test_preds['raw'], f'{Config.PREDICTIONS_PATH}/test_raw_preds.npy')
            
            export_metric(raw_mae, "raw_mae.txt")

        elif phase == '4_build_full':
            # [PHASE 5: STAGE B - EMBED BUILDING]
            tracer.checkpoint("load_data")
            train, test = load_data()
            if Config.TRACE_MODE or Config.DEBUG_MINIMAL:
                rows = Config.TRACE_ROWS if Config.TRACE_MODE else Config.DEBUG_MINIMAL_ROWS
                train = train.head(rows); test = test.head(rows)
            
            schema = Config.get_feature_schema()
            residuals = load_npy(f'{Config.PROCESSED_PATH}/residuals_raw.npy')
            oof_raw = load_npy(f'{Config.PROCESSED_PATH}/oof_raw.npy')
            
            # Initial hybrid features to get reconstructor
            train, test, _, reconstructor = add_hybrid_latent_features(train, test)
            
            # Supercharged PCA using residuals
            train, test, _ = generate_supercharged_latent_features(
                train, test, reconstructor, residuals, oof_raw
            )
            
            # [PHASE 6: IMMUTABILITY LOCK]
            # Final build and contract enforcement
            from src.data_loader import build_features
            X_train_full, train = build_features(train, schema, mode='full')
            X_test_full, test = build_features(test, schema, mode='full')
            
            # Save Full Feature Matrices
            save_npy(X_train_full.values, f'{Config.PROCESSED_PATH}/X_train_full.npy')
            save_npy(X_test_full.values, f'{Config.PROCESSED_PATH}/X_test_full.npy')
            
            # Also save regime_proxy for meta model
            save_npy(train['regime_proxy'].values, f'{Config.PROCESSED_PATH}/regime_proxy_tr.npy')
            save_npy(test['regime_proxy'].values, f'{Config.PROCESSED_PATH}/regime_proxy_te.npy')

        elif phase == '5_train_final':
            # [PHASE 4: TRAINER ISOLATION - FINAL]
            schema = Config.get_feature_schema()
            X_train_full = load_npy(f'{Config.PROCESSED_PATH}/X_train_full.npy')
            X_test_full = load_npy(f'{Config.PROCESSED_PATH}/X_test_full.npy')
            y_train = load_npy(f'{Config.PROCESSED_PATH}/y_train.npy')
            scenario_id = load_npy(f'{Config.PROCESSED_PATH}/scenario_id.npy', allow_pickle=True)
            
            trainer = Trainer(X_train_full, y_train, X_test_full, schema, scenario_id)
            
            # 1. We need oof_raw for the meta model
            oof_raw = load_npy(f'{Config.PROCESSED_PATH}/oof_raw.npy')
            trainer.oof_preds['raw'] = oof_raw
            trainer.test_preds['raw'] = load_npy(f'{Config.PREDICTIONS_PATH}/test_raw_preds.npy')
            
            # 2. Train Embed Model
            embed_mae, oof_embed = trainer.fit_embed_model()
            
            # 3. Meta Stacking
            regime_proxy_tr = load_npy(f'{Config.PROCESSED_PATH}/regime_proxy_tr.npy')
            regime_proxy_te = load_npy(f'{Config.PROCESSED_PATH}/regime_proxy_te.npy')
            
            X_meta_tr = np.column_stack([oof_raw, oof_embed, regime_proxy_tr]).astype(np.float32)
            X_meta_te = np.column_stack([trainer.test_preds['raw'], trainer.test_preds['embed'], regime_proxy_te]).astype(np.float32)
            
            meta_mae, oof_meta = trainer.fit_meta_model(X_meta_tr, X_meta_te)
            
            # Save Final Predictions
            save_npy(oof_meta, f"{Config.PREDICTIONS_PATH}/oof_stable.npy")
            save_npy(trainer.test_preds['meta'], f"{Config.PREDICTIONS_PATH}/test_stable.npy")
            
            export_metric(meta_mae)
            forensic.log_model_performance(y_train, oof_meta)
            forensic.save_all()

        elif phase == '6_retrain':
            # Pseudo Labeling on Full Schema
            schema = Config.get_feature_schema()
            X_train_full = load_npy(f'{Config.PROCESSED_PATH}/X_train_full.npy')
            X_test_full = load_npy(f'{Config.PROCESSED_PATH}/X_test_full.npy')
            y_train = load_npy(f'{Config.PROCESSED_PATH}/y_train.npy')
            scenario_id = load_npy(f'{Config.PROCESSED_PATH}/scenario_id.npy', allow_pickle=True)
            test_stable = load_npy(f'{Config.PREDICTIONS_PATH}/test_stable.npy')
            
            # Simple top 10% high confidence pseudo labeling
            n_pseudo = int(len(test_stable) * 0.1)
            high_conf_idx = np.argsort(test_stable)[-n_pseudo:]
            
            X_combined = np.concatenate([X_train_full, X_test_full[high_conf_idx]]).astype(np.float32)
            y_combined = np.concatenate([y_train, test_stable[high_conf_idx]]).astype(np.float32)
            groups_combined = np.concatenate([scenario_id, np.array(['PSEUDO'] * n_pseudo)])
            
            trainer = Trainer(X_combined, y_combined, X_test_full, schema, groups_combined)
            
            _, cat_mae = trainer.train_kfolds(X_subset=X_combined, y=y_combined, result_key='cat', custom_params=Config.CAT_PARAMS)
            save_npy(trainer.test_preds['cat'], f'{Config.PREDICTIONS_PATH}/test_cat.npy')

        elif phase == '7_inference':
            logger.info("[EXPLOSION] Starting Leaderboard Explosion Inference Pipeline")
            final_preds = run_explosion_inference()
            save_npy(final_preds, f'{Config.PREDICTIONS_PATH}/final_submission.npy')
            logger.info(f"[EXPLOSION] Final predictions saved | mean={final_preds.mean():.4f} | std={final_preds.std():.4f}")

        elif phase == '8_submission':
            final_preds = load_npy(f'{Config.PREDICTIONS_PATH}/final_submission.npy')
            sample_sub = pd.read_csv(Config.DATA_PATH + 'sample_submission.csv')
            if Config.TRACE_MODE or Config.DEBUG_MINIMAL:
                rows = Config.TRACE_ROWS if Config.TRACE_MODE else Config.DEBUG_MINIMAL_ROWS
                sample_sub = sample_sub.head(rows)

            validate_output_len(final_preds, len(sample_sub), "Submission")
            
            train_stats = json.load(open(f'{Config.PROCESSED_PATH}/train_stats.json'))
            pred_std = float(np.std(final_preds))
            std_ratio = pred_std / (train_stats['std'] + 1e-9)
            p99_ratio = float(np.quantile(final_preds, 0.99)) / (train_stats['p99'] + 1e-9)
            
            print("\n" + "="*50)
            print("[DISTRIBUTION_STATUS]")
            print(f"* std_ratio: {std_ratio:.4f}")
            print(f"* p99_ratio: {p99_ratio:.4f}")
            
            if std_ratio < 0.7: logger.warning("[COLLAPSE_DETECTED] std_ratio < 0.7!")
            if p99_ratio < 0.6: logger.warning("[COLLAPSE_DETECTED] p99_ratio < 0.6!")
            
            sample_sub[Config.TARGET] = final_preds
            sample_sub.to_csv(Config.SUBMISSION_PATH, index=False)
            logger.info(f"[SUBMISSION_COMPLETE] ID: {Config.RUN_ID}")
            
        elif phase == '9_intelligence':
            # [PHASE 3: ASSERTION FIREWALL]
            # Ensure the intelligence module receives a complete metric contract.
            intel = ExperimentIntelligence()
            
            # Load OOF and targets to build the full metric suite
            y_train = load_npy(f'{Config.PROCESSED_PATH}/y_train.npy')
            oof_stable = load_npy(f'{Config.PREDICTIONS_PATH}/oof_stable.npy')
            
            from src.utils import build_metrics
            metrics = build_metrics(y_train, oof_stable)
            
            # Additional metadata for registry
            metrics["features"] = [] # Placeholder for future feature tracking
            
            intel.run_risk_focused_pipeline(Config.RUN_ID, metrics)

        elif phase == 'validate_artifacts':
            target = getattr(Config, 'VALIDATE_TARGET', None)
            if target: ArtifactValidator.check_phase_outputs(target, logger)
        elif phase == 'verify_paths':
            verify_path_permissions(logger)
            validate_consistency(logger)

def inject_temporal_continuity(train, test, K=20):
    # PURPOSE: [DEPRECATED] Subsumed by HybridTemporalReconstructor
    # INPUT: train (DataFrame), test (DataFrame), K (int)
    # OUTPUT: DataFrame, List[int]
    # FAILURE: Returns original test if not implemented
    return test, list(range(len(test)))

def main():
    # PURPOSE: Entry point for the pipeline execution engine
    # INPUT: CLI arguments (--phase, --mode, --dry-run, --smoke-test)
    # OUTPUT: None
    # FAILURE: Exits with code 1 if phase or contract violation detected
    
    # [MISSION: GLOBAL MODEL INTERFACE LOCKDOWN — ELIMINATE ALL CONTRACT VIOLATIONS]
    # Scan entire codebase for direct model.fit / predict calls.
    check_model_contract_compliance()

    parser = argparse.ArgumentParser(description="DACON Pipeline Execution Engine")
    parser.add_argument('--phase', type=str, required=True, choices=VALID_PHASES, help="Pipeline phase to execute")
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'debug', 'trace'], help="Execution mode")
    parser.add_argument('--dry-run', action='store_true', help="Validate paths without writing")
    parser.add_argument('--smoke-test', action='store_true', help="Run minimal rows/folds for testing")
    parser.add_argument('--validate-target', type=str, help="Target phase for artifact validation")
    
    # [PHASE 5: DEBUG TRACE]
    args = parser.parse_args()
    
    print(f"\n[PIPELINE_PHASE_LIST] {VALID_PHASES}")
    print(f"[PHASE_EXECUTION_START] {args.phase} (Mode: {args.mode})")
    
    if args.dry_run: Config.DRY_RUN = True
    if args.validate_target: Config.VALIDATE_TARGET = args.validate_target
    
    try:
        run_phase(args.phase, args.mode, smoke_test=args.smoke_test)
        print(f"[PHASE_EXECUTION_SUCCESS] {args.phase}")
    except Exception as e:
        print(f"\n[PHASE_EXECUTION_FAILED] {args.phase}: {str(e)}")
        sys.exit(1)

if __name__ == "__main__": main()
