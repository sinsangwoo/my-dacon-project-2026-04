import os
import numpy as np

class Config:
    # --- Existing Path Contracts ---
    DATA_PATH = "./data/"
    PROCESSED_PATH = "./outputs/full_restoration_v2/processed/"
    MODEL_PATH = "./outputs/full_restoration_v2/models/"
    SUBMISSION_PATH = "./outputs/full_restoration_v2/submission/"
    LOG_PATH = "./logs/full_restoration_v2/"
    LAYOUT_PATH = "./data/layout.csv"
    
    # --- Operational Parameters ---
    SEED = 42
    N_FOLDS = 5
    SMOKE_ROWS = 500
    ADAPTIVE_FOLDS = {
        'raw': 3,
        'full': 5,
        'extreme': 3
    }
    
    # --- Feature Engineering Parameters ---
    ADVERSARIAL_THRESHOLD = 0.7  # Drop features with AUC > 0.7
    ZERO_IMPORTANCE_DROP = True
    
    # --- Model Hyperparameters ---
    LGBM_PARAMS = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'seed': SEED,
        'verbose': -1,
        'n_jobs': -1,
    }
    
    # [EMBEDDING_CONTRACT]
    from .schema import BASE_COLS, EMBED_DIM, MULTI_K
    EMBED_BASE_COLS = [
        'order_inflow_15m', 'unique_sku_15m', 'avg_items_per_order', 'robot_active', 
        'robot_utilization', 'avg_trip_distance', 'charge_queue_length', 'avg_charge_wait', 
        'congestion_score', 'near_collision_15m', 'manual_override_ratio', 
        'warehouse_temp_avg', 'humidity_pct', 'day_of_week', 'air_quality_idx'
    ]
    
    # [MISSION: PCA FEATURE SANITIZATION]
    # Optimal 15-feature set that captures >0.8 variance in 8 components
    PCA_INPUT_COLS = EMBED_BASE_COLS
    
    MULTI_K = MULTI_K
    EMBED_DIM = EMBED_DIM
    
    # ID and Target
    ID_COLS = ['ID', 'scenario_id', 'layout_id']
    TARGET = 'target'
    
    @classmethod
    def setup_directories(cls):
        for path in [cls.PROCESSED_PATH, cls.MODEL_PATH, cls.SUBMISSION_PATH, cls.LOG_PATH]:
            os.makedirs(path, exist_ok=True)
