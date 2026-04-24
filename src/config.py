import os
from .schema import FEATURE_SCHEMA

class Config:
    # Execution Mode
    MODE = 'full' 
    RUN_ID = os.getenv('RUN_ID', 'full_restoration_v2') # Bump version
    
    DEBUG_PHASE3 = False
    DEBUG_MINIMAL = False 
    DEBUG_MINIMAL_ROWS = 100
    TRACE_MODE = False    
    TRACE_ROWS = 1000     
    TRACE_LEVEL = 'FULL'  
    SMOKE_ROWS = 500      # NEW: Row cap for --smoke-test
    RUN_DIAGNOSTICS = True # Re-enable for restoration
    LEAKAGE_FREE_MODE = True # [ZERO_TOLERANCE_CV]
    SMOKE_TEST = False       # NEW: Global flag for verification runs
    FORCE_OVERWRITE = False  # NEW: Safe-by-default isolation guard
    MEMORY_THRESHOLD_GB = 4.0 
    DRY_RUN = False 
    
    # Paths
    DATA_PATH = './data/'
    LAYOUT_PATH = './data/layout_info.csv'
    
    # Run-specific Paths
    OUTPUT_BASE = f'./outputs/{RUN_ID}'
    PROCESSED_PATH = f'{OUTPUT_BASE}/processed'
    GLOBAL_STATS_PATH = f'{PROCESSED_PATH}/global_stats.json'
    MODELS_PATH = f'{OUTPUT_BASE}/models'
    PREDICTIONS_PATH = f'{OUTPUT_BASE}/predictions'
    SUBMISSION_PATH = f'{OUTPUT_BASE}/submission.csv'
    LOG_DIR = f'logs/{RUN_ID}'
    SUMMARY_DIR = f'metadata/{RUN_ID}'
    AUDIT_HISTORY_PATH = f'metadata/{RUN_ID}/audit_history.json'
    DONE_DIR = f'.done/{RUN_ID}'

    @classmethod
    def recompute_paths(cls):
        cls.OUTPUT_BASE = f'./outputs/{cls.RUN_ID}'
        cls.PROCESSED_PATH = f'{cls.OUTPUT_BASE}/processed'
        cls.GLOBAL_STATS_PATH = f'{cls.PROCESSED_PATH}/global_stats.json'
        cls.MODELS_PATH = f'{cls.OUTPUT_BASE}/models'
        cls.PREDICTIONS_PATH = f'{cls.OUTPUT_BASE}/predictions'
        cls.SUBMISSION_PATH = f'{cls.OUTPUT_BASE}/submission.csv'
        cls.LOG_DIR = f'logs/{cls.RUN_ID}'
        cls.SUMMARY_DIR = f'metadata/{cls.RUN_ID}'
        cls.AUDIT_HISTORY_PATH = f'metadata/{cls.RUN_ID}/audit_history.json'
        cls.DONE_DIR = f'.done/{cls.RUN_ID}'
    
    # Columns
    TARGET = 'avg_delay_minutes_next_30m'
    ID_COLS = ['ID', 'layout_id', 'scenario_id']
    
    # [EMBEDDING_CONTRACT]
    from .schema import BASE_COLS, EMBED_DIM, MULTI_K
    EMBED_BASE_COLS = BASE_COLS
    MULTI_K = MULTI_K
    EMBED_DIM = EMBED_DIM
    
    STABILITY_THRESHOLD = 0.20 # KS-statistic limit for feature stability
    
    # Model Parameters
    RAW_LGBM_PARAMS = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 31,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    EMBED_LGBM_PARAMS = {
        'n_estimators': 2000,
        'learning_rate': 0.03,
        'max_depth': 7,
        'num_leaves': 63,
        'min_data_in_leaf': 50, # Variance Recovery: Avoid overfitting
        'lambda_l2': 1.0,      # Variance Recovery: Stronger regularization
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    CHUNK_SIZE = 5000
    EMBED_CHUNK_SIZE = 1000 # [OOM_GUARD] Reduced from 2000 to 1000 for WSL stability
    
    # Feature Engineering
    TS_TOP_K = 30           
    INT_TOP_K = 20          
    INT_PRUNE_K = 50        
    WINDOWS = [3, 5]
    VARIANCE_THRESHOLD = 1e-6
    VELOCITY_CLIP_PERCENTILE = (1, 99)
    CAUSAL_SLOPE_WINDOW = 5
    NORMALIZED_TIME_DENOM = 24.0
    RATE_EPS = 1e-6
    
    # Feature Pruning
    IMPORTANCE_THRESHOLD = 0.95
    
    # Validation & Stability
    SPLIT_STRATEGY = 'GroupKFold' 
    ADAPTIVE_FOLDS = {'full': 5, 'debug': 2, 'trace': 2}
    NFOLDS = ADAPTIVE_FOLDS[MODE]
    SEEDS = [42] 
    STABILITY_VAR_THRESHOLD = 0.5
    OVERFITTING_GAP_THRESHOLD = 0.2
    WORST_CASE_SELECTION = True
    EXTREME_TARGET_QUANTILE = 0.9
    EXTREME_SAMPLE_WEIGHT = 1.2 # Increased for tail focus
    PRED_VARIANCE_RATIO_TARGET_MIN = 0.7
    PRED_VARIANCE_RATIO_REJECT = 0.5
    ENSEMBLE_MAX_CORR = 0.95
    ADVERSARIAL_SAMPLE_SIZE = 20000
    
    # [DECOMPOSITION_CONTRACT]
    ENABLE_CONTRASTIVE_AMP = False
    CONTRASTIVE_AMP_FACTOR = 1.0
    FORCED_AMP_MODE = False
    
    ACTIVATION_IMPORTANCE_THRESHOLD = 0.20
    ACTIVATION_VARIANCE_LIFT_MIN = 0.5
    DECORRELATION_THRESHOLD = 0.90
    EMBED_MODE_LOCKED = True
    
    # Stacking
    META_MODEL = 'explosion_v5' 
    STACKING_TOP_K = 30  
    
    # Drift Control
    ENABLE_STAGE1_SHIELD = True    
    ENABLE_STAGE2_RATIO = True     
    ENABLE_STAGE3_SOFTENING = True 
    ENABLE_STAGE4_GATING = True   
    ENABLE_STAGE5_PRUNING = True   
    
    # SSOT Metrics
    METRIC_SCHEMA = {
        "mean_mae": float,      
        "worst_mae": float,     
        "extreme_mae": float,   
        "variance_ratio": float 
    }

    @classmethod
    def get_feature_schema(cls):
        return FEATURE_SCHEMA

    # Artifact Registry
    ARTIFACT_MANIFEST = {
        "1_data_check": [{"pattern": "train_raw.npy", "severity": "OPTIONAL", "desc": "Raw train snapshot"}],
        "2_build_raw": [
            {"pattern": "X_train_raw.npy", "severity": "CRITICAL", "desc": "Raw feature matrix (Train)"},
            {"pattern": "X_test_raw.npy", "severity": "CRITICAL", "desc": "Raw feature matrix (Test)"},
            {"pattern": "y_train.npy", "severity": "CRITICAL", "desc": "Training targets"},
            {"pattern": "scenario_id.npy", "severity": "CRITICAL", "desc": "Scenario IDs"}
        ],
        "3_train_raw": [
            {"pattern": "oof_raw.npy", "severity": "CRITICAL", "desc": "OOF predictions from raw model"},
            {"pattern": "residuals_raw.npy", "severity": "CRITICAL", "desc": "Residuals from raw model"}
        ],
        "4_build_full": [
            {"pattern": "X_train_full.npy", "severity": "CRITICAL", "desc": "Full feature matrix (Train)"},
            {"pattern": "X_test_full.npy", "severity": "CRITICAL", "desc": "Full feature matrix (Test)"}
        ],
        "5_train_final": [
            {"pattern": "oof_stable.npy", "severity": "CRITICAL", "desc": "Final OOF predictions"},
            {"pattern": "test_stable.npy", "severity": "CRITICAL", "desc": "Final test predictions"}
        ],
        "7_inference": [{"pattern": "final_submission.npy", "severity": "CRITICAL", "desc": "Final prediction array"}],
        "8_submission": [{"pattern": "submission.csv", "severity": "CRITICAL", "desc": "Final competition submission file"}]
    }
