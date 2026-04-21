import os

class Config:
    # Execution Mode
    MODE = 'full' # Full Pipeline mode enabled
    RUN_ID = os.getenv('RUN_ID', 'full_restoration_v1')
    
    DEBUG_PHASE3 = False # For Phase 3 logic testing
    DEBUG_MINIMAL = False # Full pipeline sanity check (< 10s)
    DEBUG_MINIMAL_ROWS = 100
    TRACE_MODE = False    # Forensic Audit Trace Mode
    TRACE_ROWS = 1000     # Default rows for trace mode
    TRACE_LEVEL = 'FULL'  # 'OFF', 'LIGHT', 'FULL'
    RUN_DIAGNOSTICS = False # Core Performance Guard: Toggle expensive ablation tests
    MEMORY_THRESHOLD_GB = 1.5 # Warning threshold for memory footprint
    DRY_RUN = False # Structural validation mode skip heavy training
    
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
    
    # Columns
    TARGET = 'avg_delay_minutes_next_30m'
    ID_COLS = ['ID', 'layout_id', 'scenario_id']
    
    # [EMBEDDING_CONTRACT] Stable features for representation learning (<15% NaN)
    EMBED_BASE_COLS = [
        'order_inflow_15m', 'unique_sku_15m', 'avg_items_per_order', 'urgent_order_ratio',
        'heavy_item_ratio', 'cold_chain_ratio', 'sku_concentration', 'robot_active',
        'robot_idle', 'robot_charging', 'robot_utilization', 'avg_trip_distance',
        'task_reassign_15m', 'battery_std', 'low_battery_ratio', 'charge_queue_length',
        'avg_charge_wait', 'congestion_score', 'max_zone_density', 'blocked_path_15m',
        'near_collision_15m', 'fault_count_15m', 'avg_recovery_time', 'replenishment_overlap',
        'pack_utilization', 'manual_override_ratio', 'warehouse_temp_avg', 'humidity_pct',
        'day_of_week', 'air_quality_idx'
    ]
    
    MULTI_K = [10, 20, 40]
    EMBED_DIM = 32 # Supercharged PCA: 8(raw) + 8(log) + 8(rank) + 8(local)
    CHUNK_SIZE = 5000
    
    # Feature Engineering (Full Set)
    TS_TOP_K = 30           # Number of features for basic TS
    INT_TOP_K = 20          # Number of features for interaction pair candidates
    INT_PRUNE_K = 50        # Number of interactions to keep after pruning
    WINDOWS = [3, 5]
    VARIANCE_THRESHOLD = 1e-6
    VELOCITY_CLIP_PERCENTILE = (1, 99)
    CAUSAL_SLOPE_WINDOW = 5
    NORMALIZED_TIME_DENOM = 24.0
    RATE_EPS = 1e-6
    
    # Feature Pruning (95% Importance)
    IMPORTANCE_THRESHOLD = 0.95
    
    # Validation & Stability
    SPLIT_STRATEGY = 'GroupKFold' 
    ADAPTIVE_FOLDS = {'full': 5, 'debug': 2, 'trace': 2}
    NFOLDS = ADAPTIVE_FOLDS[MODE]
    SEEDS = [42] # Rule 1: Fixed Seed
    STABILITY_VAR_THRESHOLD = 0.5
    OVERFITTING_GAP_THRESHOLD = 0.2
    WORST_CASE_SELECTION = True
    EXTREME_TARGET_QUANTILE = 0.9
    EXTREME_SAMPLE_WEIGHT = 1.0
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
    
    # Memory Guard (in MB)
    MEMORY_WARN_THRESHOLD_MB = 6500
    
    # Stacking
    META_MODEL = 'mini_ridge' # 'mini_ridge': OOF-only Ridge stacking (v7.0)
    STACKING_TOP_K = 30  # Top features to include in stacking meta-model
    
    # Conditional Blending (v7.0)
    CONDITIONAL_BLEND_THRESHOLDS = [20, 25, 30, 35]
    CONDITIONAL_BLEND_TEMP = 5.0
    
    # Advanced Feature Pruning (v7.0)
    COLLINEAR_THRESHOLD = 0.98
    DRIFT_KS_THRESHOLD = 0.15
    PROTECTION_TOP_K = 50

    # --- [PHASE 1: METRIC SCHEMA DECLARATION (SSOT)] ---
    # Global Immutable Metric Schema
    # PURPOSE: Define the absolute contract for model evaluation.
    # FAILURE: Structural mismatch in build_metrics or intelligence will cause a hard stop.
    METRIC_SCHEMA = {
        "mean_mae": float,      # Global Mean Absolute Error
        "worst_mae": float,     # MAE of top 10% highest error samples (Prediction error tail)
        "extreme_mae": float,   # MAE of samples where target is in top 5% (Target tail)
        "variance_ratio": float # Ratio of pred_std to target_std
    }

    # --- [PHASE 1: FEATURE DECLARATION LAYER] ---
    # Global Immutable Feature Schema
    @classmethod
    def get_feature_schema(cls):
        if hasattr(cls, '_feature_schema'):
            return cls._feature_schema
            
        base_features = cls.EMBED_BASE_COLS
        
        # 1. Raw Features (Base + TS + Extreme)
        ts_suffixes = [
            '_rolling_mean_3', '_rolling_mean_5', '_rolling_std_3', '_rolling_std_5',
            '_diff_1', '_diff_3', '_rate_1', '_slope_5', '_recent_max_5', '_recent_min_5',
            '_range_5', '_expanding_mean', '_expanding_sum', '_expanding_std'
        ]
        extreme_suffixes = [
            '_rel_to_mean_5', '_rel_to_max_5', '_rel_rank_5', '_accel', '_accel_mean_5',
            '_volatility_expansion_std', '_volatility_expansion_range', '_regime_id',
            '_consecutive_above_q75', '_consecutive_increase_count', '_consecutive_above_mean_count'
        ]
        
        raw_features = list(base_features)
        raw_features += ['timestep_index', 'normalized_time', 'cold_start_flag']
        
        for col in base_features:
            for s in ts_suffixes:
                raw_features.append(f"{col}{s}")
            for s in extreme_suffixes:
                raw_features.append(f"{col}{s}")
        
        # Interactions (Fixed list from add_extreme_detection_features)
        interaction_pairs = [
            ('order_inflow_15m', 'robot_utilization'),
            ('heavy_item_ratio', 'order_inflow_15m'),
            ('heavy_item_ratio', 'robot_utilization'),
        ]
        for f1, f2 in interaction_pairs:
            raw_features.append(f"inter_{f1}_x_{f2}")
        
        raw_features.extend(["early_warning_flag", "early_warning_score"])
        
        # 2. Embed Features (Residual-dependent)
        # [MISSION: EMBEDDING REMOVAL] Redundant and harmful features removed after forensic audit.
        embed_features = []
        
        all_features = raw_features + embed_features
        
        # Ensure uniqueness while preserving order
        all_features = list(dict.fromkeys(all_features))
        
        cls._feature_schema = {
            "raw_features": raw_features,
            "embed_features": embed_features,
            "all_features": all_features,
            "feature_to_index": {feat: i for i, feat in enumerate(all_features)}
        }
        return cls._feature_schema

    # Pseudo Labeling
    PSEUDO_RATIOS = [0.1, 0.2, 0.3] if MODE == 'full' else []
    MAX_PSEUDO_RATIO = 0.3
    PSEUDO_WEIGHTS = [0.3, 0.4, 0.5] if MODE == 'full' else [0.5]
    
    # Model Hyperparameters - LightGBM
    LGBM_PARAMS = {
        'n_estimators': 300, # Fast Mode
        'learning_rate': 0.03,
        'max_depth': 7,
        'num_leaves': 80,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1,
        'early_stopping_rounds': 50 # Rule 4
    }
    
    # Decomposed Model Specific Params
    RAW_LGBM_PARAMS = {
        'num_leaves': 64,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.7,
        'learning_rate': 0.03,
        'n_estimators': 300,
        'random_state': 42,
        'verbose': -1,
        'early_stopping_rounds': 50
    }
    
    EMBED_LGBM_PARAMS = {
        'num_leaves': 128,
        'min_data_in_leaf': 30,
        'feature_fraction': 0.9,
        'learning_rate': 0.03,
        'n_estimators': 300,
        'random_state': 43,
        'verbose': -1,
        'early_stopping_rounds': 50
    }
    
    META_LGBM_PARAMS = {
        'num_leaves': 16,
        'min_data_in_leaf': 20,
        'feature_fraction': 1.0,
        'learning_rate': 0.05,
        'n_estimators': 300,
        'random_state': 44,
        'verbose': -1
    }
    
    # Model Hyperparameters - CatBoost
    CAT_PARAMS = {
        'n_estimators': 300,
        'learning_rate': 0.03,
        'max_depth': 7,
        'random_state': 42,
        'verbose': False,
        'early_stopping_rounds': 50, # Rule 4
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'allow_writing_files': False,
        'thread_count': -1
    }
    
    # Training
    EARLY_STOPPING_ROUNDS = 100
    LOG_EVALUATION_STEPS = 200
    
    # Logging Control
    LOG_LEVEL = 'IMPORTANT'
    MEMORY_LOG_THROTTLE_N = 50
    
    # Artifact Registry (Fair & Robust Validation System) [v8.5]
    ARTIFACT_MANIFEST = {
        "1_data_check": [
            {"pattern": "train_raw.npy", "severity": "OPTIONAL", "desc": "Raw train snapshot"}
        ],
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
        "7_inference": [
            {"pattern": "final_submission.npy", "severity": "CRITICAL", "desc": "Final prediction array"}
        ],
        "8_submission": [
            {"pattern": "submission.csv", "severity": "CRITICAL", "desc": "Final competition submission file"}
        ]
    }
