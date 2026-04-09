import pandas as pd
import numpy as np
import logging
import argparse
import os
import sys
import gc
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
    PhaseTracer,
    memory_guard,
    GLOBAL_PEAK_MEMORY
)
from src.data_loader import (
    load_data, 
    select_top_ts_features, 
    add_time_series_features, 
    add_advanced_predictive_features,
    add_scenario_summary_features,
    add_sequence_trajectory_features,
    add_binary_thresholding,
    add_nan_flags,
    handle_engineered_nans,
    get_features,
    align_features
)
from src.trainer import Trainer
from sklearn.metrics import mean_absolute_error

def log_nan_stats(df, stage, logger):
    """Trace NaN counts and problematic columns for telemetry."""
    nan_counts = df.isna().sum()
    total_nans = nan_counts.sum()
    logger.info(f"[NaN Trace] Stage: {stage} | Total NaNs: {total_nans:,}")
    if total_nans > 0:
        top_nans = nan_counts[nan_counts > 0].sort_values(ascending=False).head(10)
        logger.info(f"[NaN Trace] Top problematic columns:\n{top_nans}")
    return total_nans

def export_metric(mae, filename="current_mae.txt"):
    """Export MAE to a file for pipeline.sh to read."""
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/{filename}", "w") as f:
        f.write(f"{mae:.6f}")

def validate_submission(submission_path, test_len):
    """Strict production-level submission validation (fail-safe)."""
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"[SUBMISSION_FAIL] File not found: {submission_path}")
    
    df = pd.read_csv(submission_path)
    target_col = df.columns[1]
    
    # 1. Row Count Validation (Hardcoded 50000)
    EXPECTED_ROWS = 50000
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"[SUBMISSION_FAIL] Row count mismatch! Expected {EXPECTED_ROWS}, got {len(df)}")
    
    # 2. ID Sequential Order Validation
    expected_ids = [f"TEST_{i:06d}" for i in range(EXPECTED_ROWS)]
    actual_ids = df['ID'].tolist()
    if actual_ids != expected_ids:
        for i, (exp, act) in enumerate(zip(expected_ids, actual_ids)):
            if exp != act:
                raise ValueError(f"[SUBMISSION_FAIL] ID mismatch at row {i}! Expected '{exp}', got '{act}'")
        if len(actual_ids) != len(expected_ids):
            raise ValueError(f"[SUBMISSION_FAIL] ID count mismatch! Expected {len(expected_ids)}, got {len(actual_ids)}")
    
    # 3. NaN Check
    nan_count = df[target_col].isnull().sum()
    if nan_count > 0:
        raise ValueError(f"[SUBMISSION_FAIL] Contains {nan_count} NaN values!")
    
    # 4. Inf Check
    inf_count = (~np.isfinite(df[target_col].values)).sum()
    if inf_count > 0:
        raise ValueError(f"[SUBMISSION_FAIL] Contains {inf_count} infinite values!")
    
    # 5. Negative Value Check
    neg_count = (df[target_col] < 0).sum()
    if neg_count > 0:
        raise ValueError(f"[SUBMISSION_FAIL] Contains {neg_count} negative delay values!")
    
    # 6. Extreme Value Sanity Check (Warning only)
    p99 = df[target_col].quantile(0.99)
    p01 = df[target_col].quantile(0.01)
    val_max = df[target_col].max()
    val_min = df[target_col].min()
    val_mean = df[target_col].mean()
    
    print(f"✓ Submission Validation PASSED: {submission_path}")
    print(f"  Rows: {len(df)} | Range: [{val_min:.4f}, {val_max:.4f}] | Mean: {val_mean:.4f}")
    print(f"  P01: {p01:.4f} | P99: {p99:.4f}")
    
    if p99 > 1000:
        print(f"  ⚠ WARNING: 99th percentile is very high ({p99:.2f}). Check for outliers.")

def run_phase(phase, mode):
    """Execute a specific phase of the pipeline with Forensic Tracing (v5.2)."""
    logger = get_logger()
    Config.MODE = mode
    
    # Trace Mode Overrides
    if mode == 'trace' or Config.TRACE_MODE:
        logger.info("[TRACE_MODE] Activating high-speed forensic trace overrides.")
        Config.TRACE_MODE = True
        Config.NFOLDS = 2
        Config.SEEDS = [42]
        Config.DEBUG_MINIMAL_ROWS = Config.TRACE_ROWS
        
    seed_everything(42)
    
    with PhaseTracer(phase, logger) as tracer:
        # Ensure directories exist
        for d in ['outputs/processed', 'outputs/models', 'outputs/predictions', 'logs', '.done']:
            os.makedirs(d, exist_ok=True)

        if phase == '1_data_check':
            tracer.checkpoint("loading_raw")
            train, test = load_data()
            
            if Config.TRACE_MODE or Config.DEBUG_MINIMAL:
                rows = Config.TRACE_ROWS if Config.TRACE_MODE else Config.DEBUG_MINIMAL_ROWS
                logger.info(f"[TRACE/DEBUG] Slicing data to {rows} rows.")
                train = train.head(rows)
                test = test.head(rows)
                
            log_forensic_snapshot(train, "train_raw", logger)
            log_forensic_snapshot(test, "test_raw", logger)
            del train, test; gc.collect()

        elif phase == '2_preprocess':
            tracer.checkpoint("load_data")
            train, test = load_data()
            
            if Config.TRACE_MODE or Config.DEBUG_MINIMAL:
                rows = Config.TRACE_ROWS if Config.TRACE_MODE else Config.DEBUG_MINIMAL_ROWS
                logger.info(f"[TRACE/DEBUG] Slicing data to {rows} rows.")
                train = train.head(rows)
                test = test.head(rows)
                
            pc = log_forensic_snapshot(train, "load_data", logger) # INITIAL SNAPSHOT
            
            # 1. Feature Engineering with NaN Tracing
            primary_ts = select_top_ts_features(train)
            memory_guard("select_top_ts", logger)
            
            # TS Features
            tracer.checkpoint("ts_features")
            train = add_time_series_features(train, primary_ts)
            test = add_time_series_features(test, primary_ts)
            memory_guard("ts_features", logger)
            pc = log_forensic_snapshot(train, "ts_features", logger, prev_cols=pc)
            
            # Interaction Generation
            tracer.checkpoint("interactions")
            current_features = get_features(train, test)
            groups = train['scenario_id']
            base_trainer = Trainer(train, test, current_features, groups)
            
            from lightgbm import LGBMRegressor
            probe_model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
            probe_model.fit(train[current_features].fillna(0), train[Config.TARGET])
            top_20_imp = pd.Series(probe_model.feature_importances_, index=current_features).sort_values(ascending=False).head(20).index.tolist()
            del probe_model; gc.collect()
            memory_guard("interaction_probe", logger)
            
            train, interaction_cols = base_trainer.generate_interactions(train, top_20_imp)
            test, _ = base_trainer.generate_interactions(test, top_20_imp)
            top_50_interactions = base_trainer.prune_interactions(train.fillna(0), interaction_cols, groups)
            memory_guard("interactions", logger)
            pc = log_forensic_snapshot(train, "interactions", logger, prev_cols=pc)
            
            # Advanced FE
            tracer.checkpoint("full_fe")
            global_stds = train[primary_ts].std().to_dict()
            train = add_advanced_predictive_features(train, primary_ts, global_stds)
            test = add_advanced_predictive_features(test, primary_ts, global_stds)
            train = add_scenario_summary_features(train, primary_ts)
            test = add_scenario_summary_features(test, primary_ts)
            memory_guard("scenario_summary", logger)
            
            # Sequence Trajectory Features (must be after scenario_summary)
            tracer.checkpoint("sequence_fe")
            train = add_sequence_trajectory_features(train, primary_ts)
            test = add_sequence_trajectory_features(test, primary_ts)
            memory_guard("sequence_trajectory", logger)
            
            train = add_binary_thresholding(train, primary_ts)
            test = add_binary_thresholding(test, primary_ts)
            memory_guard("binary_thresholding", logger)
            pc = log_forensic_snapshot(train, "full_fe", logger, prev_cols=pc)
            
            # 2. NaN Intelligence Policy (v4.3)
            # Create Flags BEFORE filling
            flag_cols = [c for c in train.columns if '_lag1' in c or '_diff' in c]
            train = add_nan_flags(train, flag_cols)
            test = add_nan_flags(test, flag_cols)
            memory_guard("nan_flags", logger)
            
            # Apply Imputation
            tracer.checkpoint("imputation")
            all_cols = get_features(train, test)
            train = handle_engineered_nans(train, all_cols)
            test = handle_engineered_nans(test, all_cols)
            memory_guard("imputation", logger)
            pc = log_forensic_snapshot(train, "final_imputation", logger, prev_cols=pc)
            
            # 3. Importance-Based Pruning (95% Gain)
            tracer.checkpoint("importance_pruning")
            all_candidate_features = get_features(train, test)
            all_candidate_features = [f for f in all_candidate_features if 'inter_' not in f or f in top_50_interactions]
            reduced_features, feature_importances = base_trainer.importance_pruning(train, all_candidate_features)
            
            # --- [3] EARLY FEATURE REDUCTION (Top 400 only) ---
            TARGET_FEAT_COUNT = 400
            if len(reduced_features) > TARGET_FEAT_COUNT:
                logger.info(f"[EARLY_PRUNE] Reducing from {len(reduced_features)} to Top {TARGET_FEAT_COUNT} features.")
                reduced_features = reduced_features[:TARGET_FEAT_COUNT]
                feature_importances = feature_importances.loc[reduced_features]
            
            memory_guard("importance_pruning", logger)
            
            # --- v5.1 SSOT: Phase 2 = Final Authority ---
            tracer.checkpoint("ssot_alignment")
            pd.Series(all_candidate_features).to_json('outputs/processed/features_full.json')
            pd.Series(reduced_features).to_json('outputs/processed/features_reduced.json')
            feature_importances.to_json('outputs/processed/feature_importances.json')
            logger.info(f"[SSOT] Saved {len(feature_importances)} feature importances for stacking")
            
            # CANONICALIZE train and test to reduced schema (Column-wise Inplace)
            train_aligned = align_features(train, reduced_features, logger)
            test_aligned = align_features(test, reduced_features, logger)
            memory_guard("feature_alignment", logger)
            
            # Save ONLY canonicalized data using forensic wrappers
            save_npy(train_aligned.values, 'outputs/processed/X_train_reduced.npy')
            save_npy(train[Config.TARGET].values, 'outputs/processed/y_train.npy')
            save_npy(test_aligned.values, 'outputs/processed/X_test_reduced.npy')
            save_npy(train['scenario_id'].values, 'outputs/processed/groups.npy')
            
            # --- [5] DELETE-AS-YOU-GO 강화: Save full DFs before final BRIDGE check to free RAM ---
            save_pkl(train, 'outputs/processed/train_full.pkl')
            save_pkl(test, 'outputs/processed/test_full.pkl')
            
            logger.info("[DELETE-AS-YOU-GO] Deleting raw train/test DataFrames to free memory.")
            del train, test; gc.collect()
            memory_guard("raw_cleanup", logger)
            
            # Save Schema Hash
            import hashlib
            schema_str = ','.join(reduced_features)
            schema_hash = hashlib.md5(schema_str.encode()).hexdigest()
            with open('outputs/processed/features_reduced_hash.txt', 'w') as f:
                f.write(schema_hash)
            logger.info(f"[SSOT] Schema hash saved: {schema_hash}")
            
            # --- BRIDGE VALIDATION (v5.1) ---
            tracer.checkpoint("bridge_validation")
            reloaded_X = load_npy('outputs/processed/X_train_reduced.npy')
            reloaded_test_X = load_npy('outputs/processed/X_test_reduced.npy')
            reloaded_features = pd.read_json('outputs/processed/features_reduced.json', typ='series').tolist()
            
            log_forensic_snapshot(reloaded_X, "reloaded_train_X", logger)
            
            assert reloaded_X.shape[1] == len(reloaded_features)
            assert reloaded_test_X.shape[1] == len(reloaded_features)
            
            logger.info(f"✓ Bridge Validation SUCCESS. Train: {reloaded_X.shape}, Test: {reloaded_test_X.shape}")
            
            logger.info(f"Preprocessing Complete. Full features: {len(all_candidate_features)}, Reduced: {len(reduced_features)}")
            del train_aligned, test_aligned, reloaded_X, reloaded_test_X; gc.collect()
            memory_guard("final_cleanup", logger)

        elif phase == '3_train_base':
            tracer.checkpoint("data_load")
            X_train = load_npy('outputs/processed/X_train_reduced.npy')
            y_train = load_npy('outputs/processed/y_train.npy')
            X_test = load_npy('outputs/processed/X_test_reduced.npy')
            features = pd.read_json('outputs/processed/features_reduced.json', typ='series').tolist()
            groups = load_npy('outputs/processed/groups.npy', allow_pickle=True)
            
            log_forensic_snapshot(X_train, "X_train_base_raw", logger)
            
            # --- v5.1 SSOT: Schema Hash Verification ---
            tracer.checkpoint("schema_verify")
            expected_hash_path = 'outputs/processed/features_reduced_hash.txt'
            assert os.path.exists(expected_hash_path), "Schema hash file missing!"
            with open(expected_hash_path, 'r') as f:
                saved_hash = f.read().strip()
            import hashlib
            live_hash = hashlib.md5(','.join(features).encode()).hexdigest()
            assert saved_hash == live_hash, f"SCHEMA DRIFT! Saved: {saved_hash}, Live: {live_hash}"
            
            # --- v5.1 SSOT: ASSERT-ONLY Guards ---
            tracer.checkpoint("shape_assert")
            assert X_train.shape[1] == len(features), "X_train width mismatch!"
            assert X_test.shape[1] == len(features), "X_test width mismatch!"
            assert X_train.dtype != object, "X_train contains object dtype!"
            assert not np.isnan(X_train).any(), "X_train has NaNs!"
            
            # --- DEBUG/TRACE OVERRIDES (Slicing Only) ---
            if Config.TRACE_MODE or Config.DEBUG_MINIMAL:
                rows = Config.TRACE_ROWS if Config.TRACE_MODE else Config.DEBUG_MINIMAL_ROWS
                logger.info(f"[TRACE/DEBUG] Slicing to {rows} rows.")
                X_train = X_train[:rows]
                y_train = y_train[:rows]
                groups = groups[:rows]

                # --- DEBUG: Group-Aware Sampling ---
                if Config.DEBUG_PHASE3 and not Config.DEBUG_MINIMAL:
                    import random
                    unique_scenarios = np.unique(groups)
                    k = max(2, len(unique_scenarios) // 100)
                    sampled_scenarios = random.sample(list(unique_scenarios), k=k)
                    subset_idx = np.isin(groups, sampled_scenarios)
                    X_train = X_train[subset_idx]
                    y_train = y_train[subset_idx]
                    groups = groups[subset_idx]
                    logger.info(f"[DEBUG] Group-aware sampling: {k} scenarios. Shape: {X_train.shape}")

            # --- HARD FAIL ON EMPTY TRAINING ---
            assert X_train.shape[0] > 0, "No training samples available!"
            assert X_train.shape[1] > 0, "No features available!"

            # 2. Step: validation
            current_step = "validation"
            tr_df = pd.DataFrame(X_train, columns=features)
            tr_df[Config.TARGET] = y_train
            te_df = pd.DataFrame(X_test, columns=features)
            
            trainer = Trainer(tr_df, te_df, features, groups)
            trainer.validate_data(X_train, y_train, groups)
            
            # 3. Step: train (LightGBM + CatBoost in single CV loop)
            tracer.checkpoint("train_models")
            logger.info("[TRAINING_START]")
            logger.info(f"Folds: {Config.NFOLDS} | Samples: {X_train.shape[0]} | Features: {len(features)}")
            logger.info(f"--- Training LightGBM + CatBoost ---")
            lgb_mae, cat_mae = trainer.train_kfolds(features)
            
            if (Config.DEBUG_PHASE3 or Config.DEBUG_MINIMAL or Config.TRACE_MODE) and lgb_mae > 0:
                logger.info(f"[DONE] Phase 3 Sanity Check OK. MAE: {lgb_mae:.6f}")
            
            save_npy(trainer.oof_lgb, 'outputs/predictions/oof_lgb.npy')
            save_npy(trainer.test_preds_lgb, 'outputs/predictions/test_lgb.npy')
            save_npy(np.array(trainer.test_preds_lgb_folds), 'outputs/predictions/test_lgb_folds.npy')
            
            if not Config.DEBUG_PHASE3 and not Config.DEBUG_MINIMAL and not Config.TRACE_MODE:
                save_npy(trainer.oof_cat, 'outputs/predictions/oof_cat.npy')
                save_npy(trainer.test_preds_cat, 'outputs/predictions/test_cat.npy')
                
                ensemble_mae = (lgb_mae + cat_mae) / 2.0
                logger.info(f"Base Ensemble MAE (Approx): {ensemble_mae:.6f}")
                export_metric(ensemble_mae)
            else:
                export_metric(lgb_mae)
            
            logger.info("[TRAINING_END]")
            
            # --- EXECUTION GUARANTEES ---
            if not os.path.exists("outputs/predictions/oof_lgb.npy"):
                raise RuntimeError("Phase 3 failed: oof_lgb.npy was not created. Training did not execute.")
            if not os.path.exists("outputs/predictions/test_lgb.npy"):
                raise RuntimeError("Phase 3 failed: test_lgb.npy was not created. Training did not execute.")
                
            del tr_df, te_df, X_train, X_test, y_train, trainer; gc.collect()

        elif phase == '4_stacking':
            tracer.checkpoint("data_load")
            train = load_pkl('outputs/processed/train_full.pkl')
            test = load_pkl('outputs/processed/test_full.pkl')
            
            if Config.TRACE_MODE or Config.DEBUG_MINIMAL:
                rows = Config.TRACE_ROWS if Config.TRACE_MODE else Config.DEBUG_MINIMAL_ROWS
                logger.info(f"[TRACE/DEBUG] Slicing data to {rows} rows.")
                train = train.head(rows)
                test = test.head(rows)
                
            features = pd.read_json('outputs/processed/features_full.json', typ='series').tolist()
            groups = train['scenario_id']
            
            log_forensic_snapshot(train, "train_stack_raw", logger)
            
            tracer.checkpoint("meta_prep")
            trainer = Trainer(train, test, features, groups)
            trainer.oof_lgb = load_npy('outputs/predictions/oof_lgb.npy')
            
            # v5.2 TRACE-SAFE: Only load OOF Cat if shape matches trace rows
            if os.path.exists('outputs/predictions/oof_cat.npy'):
                trainer.oof_cat = load_npy('outputs/predictions/oof_cat.npy')
                if len(trainer.oof_cat) != len(train):
                    logger.warning(f"[TRACE] Shape mismatch in oof_cat ({len(trainer.oof_cat)} vs {len(train)}). Forcing zeros.")
                    trainer.oof_cat = np.zeros(len(train))
            else:
                trainer.oof_cat = np.zeros(len(train))
                
            trainer.test_preds_lgb = load_npy('outputs/predictions/test_lgb.npy')
            
            if os.path.exists('outputs/predictions/test_cat.npy'):
                trainer.test_preds_cat = load_npy('outputs/predictions/test_cat.npy')
                if len(trainer.test_preds_cat) != len(test):
                    logger.warning(f"[TRACE] Shape mismatch in test_cat ({len(trainer.test_preds_cat)} vs {len(test)}). Forcing zeros.")
                    trainer.test_preds_cat = np.zeros(len(test))
            else:
                trainer.test_preds_cat = np.zeros(len(test))
            
            # --- NEW: Load top features for Advanced Stacking ---
            tracer.checkpoint("top_features")
            top_features_train = None
            top_features_test = None
            
            if os.path.exists('outputs/processed/feature_importances.json'):
                imp_series = pd.read_json('outputs/processed/feature_importances.json', typ='series')
                reduced_features = pd.read_json('outputs/processed/features_reduced.json', typ='series').tolist()
                
                top_k = min(Config.STACKING_TOP_K, len(imp_series))
                top_feature_names = imp_series.sort_values(ascending=False).head(top_k).index.tolist()
                
                # Filter to only features that exist in reduced set
                valid_top = [f for f in top_feature_names if f in reduced_features]
                
                if valid_top:
                    top_indices = [reduced_features.index(f) for f in valid_top]
                    
                    X_train_reduced = load_npy('outputs/processed/X_train_reduced.npy')
                    X_test_reduced = load_npy('outputs/processed/X_test_reduced.npy')
                    
                    if Config.TRACE_MODE or Config.DEBUG_MINIMAL:
                        X_train_reduced = X_train_reduced[:rows]
                    
                    top_features_train = pd.DataFrame(
                        X_train_reduced[:, top_indices],
                        columns=valid_top
                    )
                    top_features_test = pd.DataFrame(
                        X_test_reduced[:, top_indices],
                        columns=valid_top
                    )
                    logger.info(f"[STACKING] Loaded {len(valid_top)} top features for meta-model")
                    del X_train_reduced, X_test_reduced; gc.collect()
            else:
                logger.warning("[STACKING] feature_importances.json not found. Using OOF-only stacking.")
            
            # --- Correlation Check ---
            tracer.checkpoint("correlation_check")
            if not (trainer.oof_cat == 0).all():
                corr = np.corrcoef(trainer.oof_lgb, trainer.oof_cat)[0, 1]
                logger.info(f"[CORRELATION] OOF_LGB vs OOF_CAT: {corr:.4f}")
                if corr > 0.9:
                    logger.warning(f"[CORRELATION_WARNING] Very high correlation ({corr:.4f}). Stacking benefit may be limited.")
            
            tracer.checkpoint("train_stack")
            oof_stack, test_stack, stack_mae, skipped = trainer.train_stacking(
                features, top_features_train, top_features_test
            )
            
            if skipped:
                logger.info(f"[STACKING_RESULT] Stacking skipped. Using base blending MAE: {stack_mae:.6f}")
            else:
                logger.info(f"[STACKING_RESULT] Stacking improved! MAE: {stack_mae:.6f}")
            
            export_metric(stack_mae)
            save_npy(test_stack, 'outputs/predictions/test_stack.npy')
            del train, test, trainer, top_features_train, top_features_test; gc.collect()

        elif phase == '5_pseudo_labeling':
            tracer.checkpoint("init")
            if Config.MODE != 'full' and not Config.TRACE_MODE:
                logger.info("Skipping Phase 5 in Debug Mode.")
                return
                
            tracer.checkpoint("data_load")
            # --- REDESIGN: Use Reduced Feature Space (Numpy) ---
            X_train_orig = load_npy('outputs/processed/X_train_reduced.npy')
            y_train_orig = load_npy('outputs/processed/y_train.npy')
            X_test_reduced = load_npy('outputs/processed/X_test_reduced.npy')
            groups_orig = load_npy('outputs/processed/groups.npy', allow_pickle=True)
            
            if Config.TRACE_MODE or Config.DEBUG_MINIMAL:
                rows = Config.TRACE_ROWS if Config.TRACE_MODE else Config.DEBUG_MINIMAL_ROWS
                logger.info(f"[TRACE/DEBUG] Slicing numpy arrays to {rows} rows.")
                X_train_orig = X_train_orig[:rows]
                y_train_orig = y_train_orig[:rows]
                groups_orig = groups_orig[:rows]
                
            features = pd.read_json('outputs/processed/features_reduced.json', typ='series').tolist()
            
            # Load prediction components for variance + agreement calculation
            test_preds_lgb = load_npy('outputs/predictions/test_lgb.npy')
            test_lgb_folds = load_npy('outputs/predictions/test_lgb_folds.npy')
            
            if os.path.exists('outputs/predictions/test_cat.npy'):
                test_preds_cat = load_npy('outputs/predictions/test_cat.npy')
            else:
                test_preds_cat = test_preds_lgb.copy() # Fallback if catboost not trained
            
            # --- v5.4 SAFETY: Align Auxiliary Arrays with Primary Test Data (Fix IndexError) ---
            n_samples = len(X_test_reduced)
            if len(test_preds_lgb) != n_samples:
                logger.warning(f"[SAFETY] Aligning predictions shape ({len(test_preds_lgb)}) to match test features ({n_samples})")
                test_preds_lgb = test_preds_lgb[:n_samples]
                test_preds_cat = test_preds_cat[:n_samples]
            if test_lgb_folds.shape[1] != n_samples:
                test_lgb_folds = test_lgb_folds[:, :n_samples]
                
            # Memory Telemetry Setup
            from src.utils import get_process_memory, get_system_ram
            total_ram = get_system_ram()
            limit_ram = total_ram * 0.8
            logger.info(f"[TELEMETRY] System RAM: {total_ram:.0f} MB | Fail-Safe Limit: {limit_ram:.0f} MB")

            # --- ENHANCED CONFIDENCE SELECTION (Fold Variance + Model Agreement) ---
            tracer.checkpoint("confidence_selection")
            test_lgb_folds_arr = np.asanyarray(test_lgb_folds)
            
            # 1. Fold Variance Score (lower variance = more reliable)
            fold_variance = np.std(test_lgb_folds_arr, axis=0)
            fold_var_rank = np.argsort(np.argsort(fold_variance))  # rank: 0=lowest variance
            
            # 2. Model Agreement Score (lower difference = more agreement)
            model_disagreement = np.abs(test_preds_lgb - test_preds_cat)
            agree_rank = np.argsort(np.argsort(model_disagreement))  # rank: 0=most agreement
            
            # 3. Combined Confidence Score (lower = better)
            if np.allclose(test_preds_cat, test_preds_lgb):
                # CatBoost not trained or identical - use only fold variance
                combined_score = fold_var_rank.astype(float)
                logger.info(f"[PSEUDO_DIAG] Using fold variance only (no CatBoost available)")
            else:
                combined_score = fold_var_rank.astype(float) + agree_rank.astype(float)
                logger.info(f"[PSEUDO_DIAG] Using combined fold variance + model agreement")
            
            n_pseudo = min(5000, len(X_test_reduced))
            confidence_idx = np.argsort(combined_score)[:n_pseudo].astype(int)
            
            X_pseudo = X_test_reduced[confidence_idx]
            y_pseudo = (test_preds_lgb[confidence_idx] + test_preds_cat[confidence_idx]) / 2.0
            
            logger.info(f"[PSEUDO_DIAG] Selected {len(X_pseudo)} pseudo-labeled samples")
            logger.info(f"[PSEUDO_DIAG] Fold Variance range: [{fold_variance[confidence_idx].min():.4f}, {fold_variance[confidence_idx].max():.4f}]")
            logger.info(f"[PSEUDO_DIAG] Model Disagreement range: [{model_disagreement[confidence_idx].min():.4f}, {model_disagreement[confidence_idx].max():.4f}]")
            log_memory_usage(X_pseudo, "X_pseudo", logger)

            # --- DISTRIBUTION CHECK ---
            tracer.checkpoint("distribution_check")
            train_mean, train_std = y_train_orig.mean(), y_train_orig.std()
            pseudo_mean, pseudo_std = y_pseudo.mean(), y_pseudo.std()
            
            logger.info(f"[PSEUDO_DIAG] Train target  | mean={train_mean:.4f} std={train_std:.4f}")
            logger.info(f"[PSEUDO_DIAG] Pseudo target  | mean={pseudo_mean:.4f} std={pseudo_std:.4f}")
            
            mean_shift = abs(pseudo_mean - train_mean)
            std_ratio = pseudo_std / (train_std + 1e-9)
            
            if mean_shift > 0.5 * train_std:
                logger.warning(f"[PSEUDO_ABORT] Mean shift too large: {mean_shift:.4f} > {0.5*train_std:.4f}")
                logger.warning(f"[PSEUDO_ABORT] Pseudo labeling aborted due to distribution mismatch!")
                del X_train_orig, X_pseudo, X_test_reduced; gc.collect()
                return
            
            if std_ratio > 2.0 or std_ratio < 0.5:
                logger.warning(f"[PSEUDO_ABORT] Std ratio out of range: {std_ratio:.4f} (acceptable: 0.5~2.0)")
                logger.warning(f"[PSEUDO_ABORT] Pseudo labeling aborted due to distribution mismatch!")
                del X_train_orig, X_pseudo, X_test_reduced; gc.collect()
                return
            
            logger.info(f"[PSEUDO_DIAG] Distribution check PASSED (mean_shift={mean_shift:.4f}, std_ratio={std_ratio:.4f})")

            # --- BASELINE EXPERIMENT (no pseudo, for comparison) ---
            tracer.checkpoint("baseline_experiment")
            tr_meta_base = pd.DataFrame({
                Config.TARGET: y_train_orig,
                'scenario_id': groups_orig
            })
            trainer_base = Trainer(tr_meta_base, X_test_reduced, features, groups_orig)
            base_mae, _ = trainer_base.train_kfolds(features, train_df=X_train_orig, seeds=Config.SEEDS[:1])
            logger.info(f"[PSEUDO_BASELINE] No-pseudo MAE (1 seed): {base_mae:.6f}")
            del tr_meta_base, trainer_base; gc.collect()

            # --- EXPERIMENT LOOP WITH SAMPLE WEIGHTING ---
            best_mae = float('inf')
            best_config = {'count': 1000, 'weight': 1.0}
            
            experiment_counts = [1000, 3000, 5000] if Config.MODE == 'full' else [1000]
            pseudo_weights = Config.PSEUDO_WEIGHTS if Config.MODE == 'full' else [0.5]
            
            stop_experiments = False
            for count in experiment_counts:
                if stop_experiments:
                    break
                for weight in pseudo_weights:
                    mem_now = get_process_memory()
                    logger.info(f"[TELEMETRY] Experiment (count={count}, weight={weight}) | Process: {mem_now:.1f} MB")
                    
                    if mem_now > limit_ram:
                        logger.warning(f"[FAIL-SAFE] Memory usage ({mem_now:.1f} MB) exceeds limit. Stopping experiments.")
                        stop_experiments = True
                        break
                        
                    tracer.checkpoint(f"experiment_{count}_w{weight}")
                    X_combined = np.vstack([X_train_orig, X_pseudo[:count]])
                    y_combined = np.concatenate([y_train_orig, y_pseudo[:count]])
                    
                    # Build sample weights
                    sw = np.concatenate([
                        np.ones(len(X_train_orig)),
                        np.full(count, weight)
                    ])
                    
                    pseudo_groups = np.array(['PSEUDO'] * count, dtype=object)
                    tr_meta = pd.DataFrame({
                        Config.TARGET: y_combined,
                        'scenario_id': np.concatenate([groups_orig, pseudo_groups])
                    })
                    
                    trainer = Trainer(tr_meta, X_test_reduced, features, tr_meta['scenario_id'].values)
                    lgb_mae, _ = trainer.train_kfolds(features, train_df=X_combined, seeds=Config.SEEDS[:1], sample_weight=sw)
                    
                    logger.info(f"[PSEUDO_EXPERIMENT] Count={count} | Weight={weight} | MAE: {lgb_mae:.6f}")
                    if lgb_mae < best_mae:
                        best_mae = lgb_mae
                        best_config = {'count': count, 'weight': weight}
                        
                    del X_combined, y_combined, sw, tr_meta, trainer; gc.collect()
                    logger.info(f"[TELEMETRY] Iteration End | Process: {get_process_memory():.1f} MB")

            logger.info(f"[PSEUDO_BEST] Best Config: count={best_config['count']}, weight={best_config['weight']} | MAE: {best_mae:.6f}")
            logger.info(f"[PSEUDO_COMPARE] Baseline (no pseudo): {base_mae:.6f} | Best pseudo: {best_mae:.6f} | Delta: {base_mae - best_mae:.6f}")
            
            if best_mae >= base_mae:
                logger.warning(f"[PSEUDO_SKIP] Pseudo labeling did NOT improve performance. Skipping.")
                export_metric(base_mae)
                del X_train_orig, X_pseudo, X_test_reduced; gc.collect()
                return
            
            export_metric(best_mae)
            
            tracer.checkpoint("save_best")
            best_count = best_config['count']
            pseudo_labels = pd.DataFrame({
                'idx': confidence_idx[:best_count],
                Config.TARGET: y_pseudo[:best_count],
                'weight': best_config['weight']
            })
            save_pkl(pseudo_labels, 'outputs/processed/pseudo_labels_index.pkl')
            
            del X_train_orig, X_pseudo, X_test_reduced; gc.collect()

        elif phase == '6_retrain':
            tracer.checkpoint("data_load")
            X_train_orig = load_npy('outputs/processed/X_train_reduced.npy')
            y_train_orig = load_npy('outputs/processed/y_train.npy')
            X_test_reduced = load_npy('outputs/processed/X_test_reduced.npy')
            groups_orig = load_npy('outputs/processed/groups.npy', allow_pickle=True)
            
            if Config.TRACE_MODE or Config.DEBUG_MINIMAL:
                rows = Config.TRACE_ROWS if Config.TRACE_MODE else Config.DEBUG_MINIMAL_ROWS
                X_train_orig = X_train_orig[:rows]
                y_train_orig = y_train_orig[:rows]
                groups_orig = groups_orig[:rows]
                
            features = pd.read_json('outputs/processed/features_reduced.json', typ='series').tolist()
            
            sw = None  # sample weights
            if os.path.exists('outputs/processed/pseudo_labels_index.pkl'):
                pseudo_info = load_pkl('outputs/processed/pseudo_labels_index.pkl')
                idx = pseudo_info['idx'].values
                target = pseudo_info[Config.TARGET].values
                pseudo_weight = float(pseudo_info['weight'].iloc[0]) if 'weight' in pseudo_info.columns else 1.0
                
                X_combined = np.vstack([X_train_orig, X_test_reduced[idx]])
                y_combined = np.concatenate([y_train_orig, target])
                
                # Build sample weights
                sw = np.concatenate([
                    np.ones(len(X_train_orig)),
                    np.full(len(idx), pseudo_weight)
                ])
                
                pseudo_groups = np.array(['PSEUDO'] * len(idx), dtype=object)
                groups_comb = np.concatenate([groups_orig, pseudo_groups])
                logger.info(f"Retraining with {len(idx)} pseudo-labeled samples (weight={pseudo_weight}).")
            else:
                X_combined = X_train_orig
                y_combined = y_train_orig
                groups_comb = groups_orig
                logger.warning("No pseudo labels found. Standard retraining.")

            tr_meta = pd.DataFrame({
                Config.TARGET: y_combined,
                'scenario_id': groups_comb
            })
            
            tracer.checkpoint("train_kfolds")
            trainer = Trainer(tr_meta, X_test_reduced, features, groups_comb)
            lgb_mae, cat_mae = trainer.train_kfolds(features, train_df=X_combined, sample_weight=sw)
            
            tracer.checkpoint("blending")
            best_w, final_mae = trainer.find_best_weight()
            final_preds = best_w * trainer.test_preds_lgb + (1 - best_w) * trainer.test_preds_cat
            
            logger.info(f"Retrained Final MAE: {final_mae:.6f}")
            export_metric(final_mae)
            save_npy(final_preds, 'outputs/predictions/final_preds.npy')
            del X_combined, y_combined, tr_meta, trainer; gc.collect()

        elif phase == '7_inference':
            tracer.checkpoint("compile")
            preds_path = None
            for path in ['outputs/predictions/test_stack.npy', 'outputs/predictions/final_preds.npy', 'outputs/predictions/test_lgb.npy']:
                if os.path.exists(path):
                    preds_path = path
                    break
            
            if preds_path is None:
                raise FileNotFoundError("No prediction files found.")
                
            logger.info(f"Using predictions from: {preds_path}")
            final_preds = load_npy(preds_path)
            save_npy(final_preds, 'outputs/predictions/submission_ready.npy')

        elif phase == '8_submission':
            tracer.checkpoint("schema_load")
            if Config.TRACE_MODE:
                raise RuntimeError("TRACE_MODE output detected. Full dataset required for submission.")
                
            sample_path = os.path.join(Config.DATA_PATH, 'sample_submission.csv')
            sample_sub = pd.read_csv(sample_path)
            test_meta = load_pkl('outputs/processed/test_full.pkl')
            
            tracer.checkpoint("pred_parsing")
            final_preds = load_npy('outputs/predictions/submission_ready.npy')
            pred_df = pd.DataFrame({'ID': test_meta['ID'].values, 'prediction': final_preds})
            
            tracer.checkpoint("id_merge")
            target_col = sample_sub.columns[1]
            submission = sample_sub[['ID']].merge(pred_df, on='ID', how='left', validate='one_to_one')
            submission[target_col] = submission['prediction']
            
            tracer.checkpoint("hard_validation")
            assert submission.shape[0] == sample_sub.shape[0]
            assert submission['ID'].equals(sample_sub['ID'])
            assert submission[target_col].isnull().sum() == 0
            assert np.isfinite(submission[target_col]).all()
            
            submission = submission[['ID', target_col]]
            submission.to_csv(Config.SUBMISSION_PATH, index=False, encoding='utf-8')
            
            tracer.checkpoint("validation")
            validate_submission(Config.SUBMISSION_PATH, len(submission))
            logger.info(f"✓ ID-Deterministic Submission exported: {Config.SUBMISSION_PATH}")
            del test_meta, sample_sub, submission, pred_df; gc.collect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, required=True, choices=[
        '1_data_check', '2_preprocess', '3_train_base', '4_stacking', 
        '5_pseudo_labeling', '6_retrain', '7_inference', '8_submission'
    ])
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'debug', 'trace'])
    parser.add_argument('--debug-minimal', action='store_true', help='Force 100 rows and 1 fold for sanity check')
    args = parser.parse_args()
    
    if args.mode == 'trace':
        Config.TRACE_MODE = True
        
    if args.debug_minimal:
        Config.DEBUG_MINIMAL = True
        
    run_phase(args.phase, args.mode)

if __name__ == "__main__":
    main()
