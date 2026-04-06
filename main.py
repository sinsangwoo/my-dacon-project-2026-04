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
    log_memory_usage
)
from src.data_loader import (
    load_data, 
    select_top_ts_features, 
    add_time_series_features, 
    add_advanced_predictive_features,
    add_scenario_summary_features,
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
    """Production-level validation for final submission."""
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file {submission_path} not found.")
    
    df = pd.read_csv(submission_path)
    
    # 1. Row Count Validation
    if len(df) != test_len:
        raise ValueError(f"Row count mismatch! Expected {test_len}, got {len(df)}")
    
    # 2. NaN Check
    if df.isnull().any().any():
        raise ValueError("Submission contains NaN values!")
    
    # 3. Value Range Check (e.g., negative delays are impossible)
    if (df[Config.TARGET] < 0).any():
        raise ValueError("Submission contains negative delay values!")
    
    print(f"✓ Submission Validation Passed: {submission_path}")

def run_phase(phase, mode):
    """Execute a specific phase of the pipeline."""
    logger = get_logger()
    Config.MODE = mode
    seed_everything(42)
    
    # Ensure directories exist
    for d in ['outputs/processed', 'outputs/models', 'outputs/predictions', 'logs', '.done']:
        os.makedirs(d, exist_ok=True)

    if phase == '1_data_check':
        logger.info("Phase 1: Validating Raw Data Integrity...")
        log_memory_usage("Start Data Check")
        train, test = load_data()
        
        if Config.DEBUG_MINIMAL:
            logger.info(f"[DEBUG_MINIMAL] Slicing data to {Config.DEBUG_MINIMAL_ROWS} rows.")
            train = train.head(Config.DEBUG_MINIMAL_ROWS)
            test = test.head(Config.DEBUG_MINIMAL_ROWS)
            
        logger.info(f"Train: {train.shape}, Test: {test.shape}")
        
        # Phase 1: Robust Data Readiness Check
        total_nans = log_nan_stats(train, "raw_data", logger)
        if total_nans > 0:
            logger.warning(f"Caution: Train data contains {total_nans} NaNs!")
        del train, test; gc.collect()

    elif phase == '2_preprocess':
        logger.info("Phase 2: Intelligent Feature Engineering & Zero-NaN Policy...")
        log_memory_usage("Preprocess Start")
        train, test = load_data()
        
        if Config.DEBUG_MINIMAL:
            logger.info(f"[DEBUG_MINIMAL] Slicing data to {Config.DEBUG_MINIMAL_ROWS} rows.")
            train = train.head(Config.DEBUG_MINIMAL_ROWS)
            test = test.head(Config.DEBUG_MINIMAL_ROWS)
            
        # 1. Feature Engineering with NaN Tracing
        log_nan_stats(train, "load_data", logger)
        
        primary_ts = select_top_ts_features(train)
        
        # TS Features
        train = add_time_series_features(train, primary_ts)
        test = add_time_series_features(test, primary_ts)
        log_nan_stats(train, "ts_features", logger)
        
        # Interaction Generation
        current_features = get_features(train, test)
        groups = train['scenario_id']
        base_trainer = Trainer(train, test, current_features, groups)
        
        from lightgbm import LGBMRegressor
        probe_model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
        probe_model.fit(train[current_features].fillna(0), train[Config.TARGET])
        top_20_imp = pd.Series(probe_model.feature_importances_, index=current_features).sort_values(ascending=False).head(20).index.tolist()
        
        train, interaction_cols = base_trainer.generate_interactions(train, top_20_imp)
        test, _ = base_trainer.generate_interactions(test, top_20_imp)
        top_50_interactions = base_trainer.prune_interactions(train.fillna(0), interaction_cols, groups)
        log_nan_stats(train, "interactions", logger)
        
        # Advanced FE
        global_stds = train[primary_ts].std().to_dict()
        train = add_advanced_predictive_features(train, primary_ts, global_stds)
        test = add_advanced_predictive_features(test, primary_ts, global_stds)
        train = add_scenario_summary_features(train, primary_ts)
        test = add_scenario_summary_features(test, primary_ts)
        train = add_binary_thresholding(train, primary_ts)
        test = add_binary_thresholding(test, primary_ts)
        log_nan_stats(train, "full_fe", logger)
        
        # 2. NaN Intelligence Policy (v4.3)
        # Create Flags BEFORE filling
        flag_cols = [c for c in train.columns if '_lag1' in c or '_diff' in c]
        train = add_nan_flags(train, flag_cols)
        test = add_nan_flags(test, flag_cols)
        
        # Apply Imputation
        all_cols = get_features(train, test)
        train = handle_engineered_nans(train, all_cols)
        test = handle_engineered_nans(test, all_cols)
        log_nan_stats(train, "final_imputation", logger)
        
        # 3. Importance-Based Pruning (95% Gain)
        all_candidate_features = get_features(train, test)
        all_candidate_features = [f for f in all_candidate_features if 'inter_' not in f or f in top_50_interactions]
        reduced_features = base_trainer.importance_pruning(train, all_candidate_features)
        
        # --- v5.1 SSOT: Phase 2 = Final Authority ---
        # Apply canonical alignment BEFORE saving anything
        logger.info("[SSOT] Phase 2: Applying canonical alignment (Feature Factory)...")
        pd.Series(all_candidate_features).to_json('outputs/processed/features_full.json')
        pd.Series(reduced_features).to_json('outputs/processed/features_reduced.json')
        
        # CANONICALIZE train and test to reduced schema
        train_aligned = align_features(train, reduced_features, logger)
        test_aligned = align_features(test, reduced_features, logger)
        
        # Save ONLY canonicalized data
        np.save('outputs/processed/X_train_reduced.npy', train_aligned.values)
        np.save('outputs/processed/y_train.npy', train[Config.TARGET].values)
        np.save('outputs/processed/X_test_reduced.npy', test_aligned.values)
        np.save('outputs/processed/groups.npy', train['scenario_id'].values)
        
        # Save Schema Hash (MD5 of feature list)
        import hashlib
        schema_str = ','.join(reduced_features)
        schema_hash = hashlib.md5(schema_str.encode()).hexdigest()
        with open('outputs/processed/features_reduced_hash.txt', 'w') as f:
            f.write(schema_hash)
        logger.info(f"[SSOT] Schema hash saved: {schema_hash}")
        
        # --- BRIDGE VALIDATION (v5.1) ---
        logger.info("[VERIFY] Running Phase 2 → 3 Bridge Validation...")
        reloaded_X = np.load('outputs/processed/X_train_reduced.npy')
        reloaded_test_X = np.load('outputs/processed/X_test_reduced.npy')
        reloaded_features = pd.read_json('outputs/processed/features_reduced.json', typ='series').tolist()
        
        assert reloaded_X.shape[1] == len(reloaded_features), \
            f"Bridge Failure: Train array width {reloaded_X.shape[1]} != Feature count {len(reloaded_features)}"
        assert reloaded_test_X.shape[1] == len(reloaded_features), \
            f"Bridge Failure: Test array width {reloaded_test_X.shape[1]} != Feature count {len(reloaded_features)}"
        assert reloaded_X.dtype != object, "Bridge Failure: Train array contains object types!"
        assert reloaded_test_X.dtype != object, "Bridge Failure: Test array contains object types!"
        assert not np.isnan(reloaded_X).any(), \
            f"Bridge Failure: Train NaN count = {np.isnan(reloaded_X).sum()}"
        assert not np.isnan(reloaded_test_X).any(), \
            f"Bridge Failure: Test NaN count = {np.isnan(reloaded_test_X).sum()}"
        logger.info(f"✓ Bridge Validation SUCCESS. Train: {reloaded_X.shape}, Test: {reloaded_test_X.shape}")
        
        # Save Full DF as Pickle (with all original columns for Phase 4+)
        save_pkl(train, 'outputs/processed/train_full.pkl')
        save_pkl(test, 'outputs/processed/test_full.pkl')
        
        logger.info(f"Preprocessing Complete. Full features: {len(all_candidate_features)}, Reduced: {len(reduced_features)}")
        del train, test, train_aligned, test_aligned, reloaded_X, reloaded_test_X; gc.collect()

    elif phase == '3_train_base':
        logger.info(f"Phase 3: Base Model Training (Mode: {'DEBUG' if Config.DEBUG_PHASE3 else 'FULL'})")
        current_step = "initializing"
        try:
            # 1. Step: data_load (Pure Consumer — NO alignment)
            current_step = "data_load"
            log_memory_usage(f"[STEP: {current_step}] Start")
            
            X_train = np.load('outputs/processed/X_train_reduced.npy')
            y_train = np.load('outputs/processed/y_train.npy')
            X_test = np.load('outputs/processed/X_test_reduced.npy')
            features = pd.read_json('outputs/processed/features_reduced.json', typ='series').tolist()
            groups = np.load('outputs/processed/groups.npy', allow_pickle=True)
            
            # --- v5.1 SSOT: Schema Hash Verification ---
            import hashlib
            current_step = "schema_verify"
            expected_hash_path = 'outputs/processed/features_reduced_hash.txt'
            assert os.path.exists(expected_hash_path), \
                "Schema hash file missing! Phase 2 must run first."
            with open(expected_hash_path, 'r') as f:
                saved_hash = f.read().strip()
            live_hash = hashlib.md5(','.join(features).encode()).hexdigest()
            assert saved_hash == live_hash, \
                f"SCHEMA DRIFT DETECTED! Saved hash: {saved_hash}, Live hash: {live_hash}"
            logger.info(f"[SSOT] Schema hash verified: {live_hash}")
            
            # --- v5.1 SSOT: ASSERT-ONLY Guards (NO alignment) ---
            current_step = "shape_assert"
            assert X_train.shape[1] == len(features), \
                f"HARD FAIL: X_train width {X_train.shape[1]} != feature count {len(features)}"
            assert X_test.shape[1] == len(features), \
                f"HARD FAIL: X_test width {X_test.shape[1]} != feature count {len(features)}"
            assert X_train.dtype != object, "HARD FAIL: X_train contains object dtype!"
            assert X_test.dtype != object, "HARD FAIL: X_test contains object dtype!"
            assert not np.isnan(X_train).any(), \
                f"HARD FAIL: X_train has {np.isnan(X_train).sum()} NaNs!"
            assert not np.isnan(X_test).any(), \
                f"HARD FAIL: X_test has {np.isnan(X_test).sum()} NaNs!"
            logger.info(f"[SSOT] Assert-only guard passed. Train: {X_train.shape}, Test: {X_test.shape}")
            
            # --- DEBUG_MINIMAL OVERRIDES ---
            if Config.DEBUG_MINIMAL:
                logger.info("[DEBUG_MINIMAL] Forcing minimal training config.")
                Config.NFOLDS = 2
                Config.SEEDS = [42]
                X_train = X_train[:Config.DEBUG_MINIMAL_ROWS]
                y_train = y_train[:Config.DEBUG_MINIMAL_ROWS]
                groups = groups[:Config.DEBUG_MINIMAL_ROWS]

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

            # 2. Step: validation
            current_step = "validation"
            tr_df = pd.DataFrame(X_train, columns=features)
            tr_df[Config.TARGET] = y_train
            te_df = pd.DataFrame(X_test, columns=features)
            
            trainer = Trainer(tr_df, te_df, features, groups)
            trainer.validate_data(X_train, y_train, groups)
            
            # 3. Step: train (LGBM)
            current_step = "train_lgbm"
            logger.info(f"--- 1/2 Training LightGBM ---")
            lgb_mae, _ = trainer.train_kfolds(features)
            
            if (Config.DEBUG_PHASE3 or Config.DEBUG_MINIMAL) and lgb_mae > 0:
                logger.info(f"[DONE] Phase 3 Sanity Check OK. MAE: {lgb_mae:.6f}")
            
            np.save('outputs/predictions/oof_lgb.npy', trainer.oof_lgb)
            np.save('outputs/predictions/test_lgb.npy', trainer.test_preds_lgb)
            np.save('outputs/predictions/test_lgb_folds.npy', np.array(trainer.test_preds_lgb_folds))
            
            # 4. Step: train (CatBoost)
            if not Config.DEBUG_PHASE3 and not Config.DEBUG_MINIMAL:
                current_step = "train_catboost"
                logger.info(f"--- 2/2 Training CatBoost ---")
                _, cat_mae = trainer.train_kfolds(features)
                np.save('outputs/predictions/oof_cat.npy', trainer.oof_cat)
                np.save('outputs/predictions/test_cat.npy', trainer.test_preds_cat)
                
                ensemble_mae = (lgb_mae + cat_mae) / 2.0
                logger.info(f"Base Ensemble MAE (Approx): {ensemble_mae:.6f}")
                export_metric(ensemble_mae)
            else:
                export_metric(lgb_mae)
            
            del tr_df, te_df, X_train, X_test, y_train, trainer; gc.collect()
            log_memory_usage("Phase 3 End")
            
        except Exception as e:
            error_msg = f"\n[ERROR]\n"
            error_msg += f"- step: {current_step}\n"
            error_msg += f"- reason: {str(e)}\n"
            error_msg += f"- shape:\n"
            error_msg += f"  - X: {X_train.shape if 'X_train' in locals() else 'N/A'}\n"
            error_msg += f"  - y: {y_train.shape if 'y_train' in locals() else 'N/A'}\n"
            error_msg += f"  - groups: {groups.shape if 'groups' in locals() else 'N/A'}\n"
            error_msg += f"- memory: N/A (Tracking Disabled)\n"
            error_msg += f"- file: outputs/processed/X_train_reduced.npy\n"
            logger.error(error_msg)
            raise e

    elif phase == '4_stacking':
        logger.info("Phase 4: Stacking Meta-model (Using Full Features)...")
        log_memory_usage("Stacking Start")
        train = load_pkl('outputs/processed/train_full.pkl')
        test = load_pkl('outputs/processed/test_full.pkl')
        features = pd.read_json('outputs/processed/features_full.json', typ='series').tolist()
        groups = train['scenario_id']
        
        trainer = Trainer(train, test, features, groups)
        trainer.oof_lgb = np.load('outputs/predictions/oof_lgb.npy')
        trainer.oof_cat = np.load('outputs/predictions/oof_cat.npy')
        trainer.test_preds_lgb = np.load('outputs/predictions/test_lgb.npy')
        trainer.test_preds_cat = np.load('outputs/predictions/test_cat.npy')
        
        oof_stack, test_stack, stack_mae = trainer.train_stacking(features)
        logger.info(f"Stacking MAE: {stack_mae:.6f}")
        export_metric(stack_mae)
        
        np.save('outputs/predictions/test_stack.npy', test_stack)
        del train, test, trainer; gc.collect()

    elif phase == '5_pseudo_labeling':
        logger.info("Phase 5: Pseudo Labeling Evaluation (Using Full Features)...")
        log_memory_usage("Pseudo Start")
        if Config.MODE != 'full':
            logger.info("Skipping Phase 5 in Debug Mode.")
            return
            
        train = load_pkl('outputs/processed/train_full.pkl')
        test = load_pkl('outputs/processed/test_full.pkl')
        features = pd.read_json('outputs/processed/features_full.json', typ='series').tolist()
        groups = train['scenario_id']
        
        trainer = Trainer(train, test, features, groups)
        trainer.test_preds_lgb = np.load('outputs/predictions/test_lgb.npy')
        trainer.test_preds_cat = np.load('outputs/predictions/test_cat.npy')
        trainer.test_preds_lgb_folds = np.load('outputs/predictions/test_lgb_folds.npy').tolist()
        
        best_ratio, pseudo_mae, _ = trainer.run_pseudo_labeling_experiments(features, test)
        logger.info(f"Best Pseudo Ratio: {best_ratio} (MAE: {pseudo_mae:.6f})")
        export_metric(pseudo_mae)
        
        pseudo_df = trainer.generate_pseudo_labels(test, best_ratio)
        save_pkl(pseudo_df, 'outputs/processed/pseudo_labeled_data_full.pkl')
        del train, test, trainer, pseudo_df; gc.collect()

    elif phase == '6_retrain':
        logger.info("Phase 6: Final Retraining (Using Full Features)...")
        log_memory_usage("Retrain Start")
        train = load_pkl('outputs/processed/train_full.pkl')
        test = load_pkl('outputs/processed/test_full.pkl')
        features = pd.read_json('outputs/processed/features_full.json', typ='series').tolist()
        
        if os.path.exists('outputs/processed/pseudo_labeled_data_full.pkl'):
            pseudo_df = load_pkl('outputs/processed/pseudo_labeled_data_full.pkl')
            combined_train = pd.concat([train, pseudo_df], axis=0).reset_index(drop=True)
            logger.info(f"Retraining with augmented data: {combined_train.shape}")
        else:
            combined_train = train
            logger.warning("No pseudo labels found, retraining on base data.")
            
        groups = combined_train['scenario_id']
        trainer = Trainer(combined_train, test, features, groups)
        lgb_mae, cat_mae = trainer.train_kfolds(features)
        
        best_w, final_mae = trainer.find_best_weight()
        final_preds = best_w * trainer.test_preds_lgb + (1 - best_w) * trainer.test_preds_cat
        
        logger.info(f"Retrained Final MAE: {final_mae:.6f}")
        export_metric(final_mae)
        np.save('outputs/predictions/final_preds.npy', final_preds)
        del train, test, trainer; gc.collect()

    elif phase == '7_inference':
        logger.info("Phase 7: Compiling Final Preds...")
        # In this simplistic pipeline, we use retrained preds if available, else stacking
        if os.path.exists('outputs/predictions/final_preds.npy'):
            final_preds = np.load('outputs/predictions/final_preds.npy')
        else:
            final_preds = np.load('outputs/predictions/test_stack.npy')
        np.save('outputs/predictions/submission_ready.npy', final_preds)

    elif phase == '8_submission':
        logger.info("Phase 8: Generating & Validating Submission...")
        test = load_pkl('outputs/processed/test_full.pkl')
        final_preds = np.load('outputs/predictions/submission_ready.npy')
        
        submission = pd.DataFrame({'ID': test['ID'], Config.TARGET: final_preds})
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        
        validate_submission(Config.SUBMISSION_PATH, len(test))
        logger.info(f"Result saved to {Config.SUBMISSION_PATH}")
        del test, submission; gc.collect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, required=True, choices=[
        '1_data_check', '2_preprocess', '3_train_base', '4_stacking', 
        '5_pseudo_labeling', '6_retrain', '7_inference', '8_submission'
    ])
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'debug'])
    parser.add_argument('--debug-minimal', action='store_true', help='Force 100 rows and 1 fold for sanity check')
    args = parser.parse_args()
    
    if args.debug_minimal:
        Config.DEBUG_MINIMAL = True
        
    run_phase(args.phase, args.mode)

if __name__ == "__main__":
    main()
