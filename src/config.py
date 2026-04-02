import os

class Config:
    # Paths
    DATA_PATH = './data/'
    SUBMISSION_PATH = './submission.csv'
    
    # Columns
    TARGET = 'avg_delay_minutes_next_30m'
    ID_COLS = ['ID', 'layout_id', 'scenario_id']
    
    # Model Hyperparameters
    LGBM_PARAMS = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 7,
        'num_leaves': 63,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'verbose': -1,
    }
    
    # Training
    NFOLDS = 5
    SEED = 42
    EARLY_STOPPING_ROUNDS = 50
    LOG_EVALUATION_STEPS = 100
