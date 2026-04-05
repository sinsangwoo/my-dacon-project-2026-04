import os

class Config:
    # Execution Mode
    MODE = 'full' # 'full' or 'debug'
    
    # Paths
    DATA_PATH = './data/'
    LAYOUT_PATH = './data/layout_info.csv'
    SUBMISSION_PATH = './outputs/submission.csv'
    
    # Columns
    TARGET = 'avg_delay_minutes_next_30m'
    ID_COLS = ['ID', 'layout_id', 'scenario_id']
    
    # Feature Engineering
    TS_TOP_K = 30           # Number of features for basic TS
    INT_TOP_K = 20          # Number of features for interaction pair candidates
    INT_PRUNE_K = 50        # Number of interactions to keep after pruning
    WINDOWS = [3, 5]
    VARIANCE_THRESHOLD = 1e-6
    VELOCITY_CLIP_PERCENTILE = (1, 99)
    
    # Validation & Stability
    SPLIT_STRATEGY = 'GroupKFold' # 'GroupKFold' or 'KFold'
    NFOLDS = 5 if MODE == 'full' else 2
    SEEDS = [42, 43, 44] if MODE == 'full' else [42]
    STABILITY_VAR_THRESHOLD = 0.5
    OVERFITTING_GAP_THRESHOLD = 0.2
    
    # Stacking
    META_MODEL = 'ridge' # 'ridge' or 'lgbm'
    
    # Pseudo Labeling
    PSEUDO_RATIOS = [0.1, 0.2, 0.3] if MODE == 'full' else []
    MAX_PSEUDO_RATIO = 0.3
    
    # Model Hyperparameters - LightGBM
    LGBM_PARAMS = {
        'n_estimators': 2000,
        'learning_rate': 0.03,
        'max_depth': 8,
        'num_leaves': 127,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }
    
    # Model Hyperparameters - CatBoost
    CAT_PARAMS = {
        'n_estimators': 2000,
        'learning_rate': 0.03,
        'max_depth': 8,
        'random_state': 42,
        'verbose': False,
        'early_stopping_rounds': 100,
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'allow_writing_files': False,
        'thread_count': -1
    }
    
    # Training
    EARLY_STOPPING_ROUNDS = 100
    LOG_EVALUATION_STEPS = 200
