import pandas as pd
import numpy as np
import logging
import argparse
import os
import sys
from src.config import Config
from src.utils import seed_everything, get_logger
from src.data_loader import (
    load_data, 
    select_top_ts_features, 
    add_time_series_features, 
    add_advanced_predictive_features,
    add_scenario_summary_features,
    add_binary_thresholding,
    get_features
)
from src.trainer import Trainer
from sklearn.metrics import mean_absolute_error

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
        train, test = load_data()
        logger.info(f"Train: {train.shape}, Test: {test.shape}")
        if train.isnull().any().any():
            logger.warning("Caution: Train data contains NaNs!")

    elif phase == '2_preprocess':
        logger.info("Phase 2: Feature Engineering & Caching...")
        train, test = load_data()
        top_ts_cols = select_top_ts_features(train)
        train = add_time_series_features(train, top_ts_cols)
        test = add_time_series_features(test, top_ts_cols)
        
        # Interaction Generation
        current_features = get_features(train, test)
        groups = train['scenario_id']
        base_trainer = Trainer(train, test, current_features, groups)
        
        from lightgbm import LGBMRegressor
        probe_model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        probe_model.fit(train[current_features], train[Config.TARGET])
        top_20_imp = pd.Series(probe_model.feature_importances_, index=current_features).sort_values(ascending=False).head(20).index.tolist()
        
        train, interaction_cols = base_trainer.generate_interactions(train, top_20_imp)
        test, _ = base_trainer.generate_interactions(test, top_20_imp)
        top_50_interactions = base_trainer.prune_interactions(train, interaction_cols, groups)
        
        # Advanced FE
        global_stds = train[top_ts_cols].std().to_dict()
        train = add_advanced_predictive_features(train, top_ts_cols, global_stds)
        test = add_advanced_predictive_features(test, top_ts_cols, global_stds)
        train = add_scenario_summary_features(train, top_ts_cols)
        test = add_scenario_summary_features(test, top_ts_cols)
        train = add_binary_thresholding(train, top_ts_cols)
        test = add_binary_thresholding(test, top_ts_cols)
        
        # Final Selection
        all_final_features = get_features(train, test)
        all_final_features = [f for f in all_final_features if 'inter_' not in f or f in top_50_interactions]
        
        # Save Processed Data
        train.to_parquet('outputs/processed/train_preprocessed.parquet')
        test.to_parquet('outputs/processed/test_preprocessed.parquet')
        pd.Series(all_final_features).to_json('outputs/processed/features.json')
        logger.info("Preprocessed data saved to outputs/processed/")

    elif phase == '3_train_base':
        logger.info("Phase 3: Base Model Training...")
        train = pd.read_parquet('outputs/processed/train_preprocessed.parquet')
        test = pd.read_parquet('outputs/processed/test_preprocessed.parquet')
        features = pd.read_json('outputs/processed/features.json', typ='series').tolist()
        groups = train['scenario_id']
        
        trainer = Trainer(train, test, features, groups)
        lgb_mae, cat_mae = trainer.train_kfolds(features)
        _, ensemble_mae = trainer.find_best_weight()
        
        logger.info(f"Base Ensemble MAE: {ensemble_mae:.6f}")
        export_metric(ensemble_mae)
        
        # Save OOF & Test Preds for Stacking
        np.save('outputs/predictions/oof_lgb.npy', trainer.oof_lgb)
        np.save('outputs/predictions/oof_cat.npy', trainer.oof_cat)
        np.save('outputs/predictions/test_lgb.npy', trainer.test_preds_lgb)
        np.save('outputs/predictions/test_cat.npy', trainer.test_preds_cat)
        # Store folds for pseudo-labeling confidence
        np.save('outputs/predictions/test_lgb_folds.npy', np.array(trainer.test_preds_lgb_folds))

    elif phase == '4_stacking':
        logger.info("Phase 4: Stacking Meta-model...")
        train = pd.read_parquet('outputs/processed/train_preprocessed.parquet')
        test = pd.read_parquet('outputs/processed/test_preprocessed.parquet')
        features = pd.read_json('outputs/processed/features.json', typ='series').tolist()
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

    elif phase == '5_pseudo_labeling':
        logger.info("Phase 5: Pseudo Labeling Evaluation...")
        if Config.MODE != 'full':
            logger.info("Skipping Phase 5 in Debug Mode.")
            return
            
        train = pd.read_parquet('outputs/processed/train_preprocessed.parquet')
        test = pd.read_parquet('outputs/processed/test_preprocessed.parquet')
        features = pd.read_json('outputs/processed/features.json', typ='series').tolist()
        groups = train['scenario_id']
        
        trainer = Trainer(train, test, features, groups)
        trainer.test_preds_lgb = np.load('outputs/predictions/test_lgb.npy')
        trainer.test_preds_cat = np.load('outputs/predictions/test_cat.npy')
        trainer.test_preds_lgb_folds = np.load('outputs/predictions/test_lgb_folds.npy').tolist()
        
        best_ratio, pseudo_mae, _ = trainer.run_pseudo_labeling_experiments(features, test)
        logger.info(f"Best Pseudo Ratio: {best_ratio} (MAE: {pseudo_mae:.6f})")
        export_metric(pseudo_mae)
        
        # Save best pseudo labels
        pseudo_df = trainer.generate_pseudo_labels(test, best_ratio)
        pseudo_df.to_parquet('outputs/processed/pseudo_labeled_data.parquet')

    elif phase == '6_retrain':
        logger.info("Phase 6: Final Retraining (Train + Pseudo)...")
        train = pd.read_parquet('outputs/processed/train_preprocessed.parquet')
        test = pd.read_parquet('outputs/processed/test_preprocessed.parquet')
        features = pd.read_json('outputs/processed/features.json', typ='series').tolist()
        
        if os.path.exists('outputs/processed/pseudo_labeled_data.parquet'):
            pseudo_df = pd.read_parquet('outputs/processed/pseudo_labeled_data.parquet')
            combined_train = pd.concat([train, pseudo_df], axis=0).reset_index(drop=True)
            logger.info(f"Retraining with augmented data: {combined_train.shape}")
        else:
            combined_train = train
            logger.warning("No pseudo labels found, retraining on base data.")
            
        groups = combined_train['scenario_id']
        trainer = Trainer(combined_train, test, features, groups)
        lgb_mae, cat_mae = trainer.train_kfolds(features)
        
        # Final weights
        best_w, final_mae = trainer.find_best_weight()
        final_preds = best_w * trainer.test_preds_lgb + (1 - best_w) * trainer.test_preds_cat
        
        logger.info(f"Retrained Final MAE: {final_mae:.6f}")
        export_metric(final_mae)
        np.save('outputs/predictions/final_preds.npy', final_preds)

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
        test = pd.read_parquet('outputs/processed/test_preprocessed.parquet')
        final_preds = np.load('outputs/predictions/submission_ready.npy')
        
        submission = pd.DataFrame({'ID': test['ID'], Config.TARGET: final_preds})
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        
        validate_submission(Config.SUBMISSION_PATH, len(test))
        logger.info(f"Result saved to {Config.SUBMISSION_PATH}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, required=True, choices=[
        '1_data_check', '2_preprocess', '3_train_base', '4_stacking', 
        '5_pseudo_labeling', '6_retrain', '7_inference', '8_submission'
    ])
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'debug'])
    args = parser.parse_args()
    
    run_phase(args.phase, args.mode)

if __name__ == "__main__":
    main()
