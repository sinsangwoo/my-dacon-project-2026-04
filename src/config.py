import os
import numpy as np
from datetime import datetime
from .schema import BASE_COLS, EMBED_DIM, MULTI_K

"""
[CONTEXT — DO NOT REMOVE]

This refactor was triggered by a structural failure where:

1. PCA-driven feature selection reduced EMBED_BASE_COLS from 30 → 15
2. Time-series feature generation depended on EMBED_BASE_COLS
3. This caused ~337 features to silently disappear from runtime
4. schema.py still declared ~700 features → causing fatal mismatch
5. trainer attempted to access non-existent columns → KeyError
6. Config refactor removed critical attributes → AttributeError risk

Root Cause:
- PCA was incorrectly treated as a primary constraint
- Feature generation pipeline became dependent on PCA inputs
- Schema and runtime were no longer synchronized
- [FIXED] Artifact isolation guard previously prevented re-runs in the same directory.

Resolution Strategy:
- Feature generation must be BASE_COLS-driven, NOT PCA-driven
- Schema must EXACTLY match runtime-generated features
- PCA must become OPTIONAL (non-blocking)
- No silent feature drops allowed under any condition
- [NEW] Implement dynamic RUN_ID to ensure automated artifact isolation.

This system now enforces:
- Zero tolerance for Schema-Runtime mismatch
- Deterministic feature generation
- Explicit failure instead of silent fallback
- Mandatory artifact isolation per RUN_ID

[END CONTEXT]
"""

class Config:
    # --- Existing Path Contracts ---
    # [SSOT_FIX] Mandatory RUN_ID contract. 
    # Must be injected by pipeline.sh to ensure artifact isolation across all phases.
    RUN_ID = os.getenv('RUN_ID')
    if not RUN_ID:
        # Fallback to a clear error state instead of silent time-based generation
        raise RuntimeError("[CONTRACT_VIOLATION] RUN_ID environment variable is MISSING. "
                           "Execution must be driven by pipeline.sh or explicit env export.")
    DATA_PATH = "./data"
    OUTPUT_BASE = f"./outputs/{RUN_ID}"
    PROCESSED_PATH = f"{OUTPUT_BASE}/processed"
    MODELS_PATH = f"{OUTPUT_BASE}/models"
    SUBMISSION_PATH = f"{OUTPUT_BASE}/submission.csv"
    PREDICTIONS_PATH = f"{OUTPUT_BASE}/predictions"
    LOG_DIR = f"./logs/{RUN_ID}"
    SUMMARY_DIR = f"{LOG_DIR}/summary"
    LAYOUT_PATH = "./data/layout_info.csv"
    GLOBAL_STATS_PATH = f"{PROCESSED_PATH}/global_stats.json"
    
    # [MISSION 6: JENSEN_RECOVERY]
    # Why: Log-transform collapses global mean by ~28% (Ratio 0.72).
    # Value 1.37 recovers 100% of the Jensen's Inequality loss based on forensic audit.
    BIAS_RECOVERY_FACTOR = 1.37
    
    # --- Operational Parameters ---
    SEED = 42
    N_FOLDS = 5 # Default
    NFOLDS = 5   # Trainer uses NFOLDS
    SMOKE_ROWS = 500
    TRACE_ROWS = 1000
    MODE = "full"
    FORCE_OVERWRITE = False
    TRACE_LEVEL = "INFO"
    LOG_LEVEL = "INFO"
    SMOKE_TEST = False
    
    ADAPTIVE_FOLDS = {
        'raw': 3,
        'full': 5,
        'extreme': 3,
        'trace': 2,
        'debug': 2
    }
    
    # [WHY_THIS_DESIGN] Feature Health Thresholds
    # ADVERSARIAL_THRESHOLD: 0.7 - Observed Data Behavior: AUC > 0.7 indicates a feature 
    #   can reliably distinguish between train and test distributions, leading to leakage.
    # STABILITY_THRESHOLD: 0.15 - Limit for drift (PSI or Mean Shift) allowed before rejection.
    # EXTREME_TARGET_QUANTILE: 0.95 - Aligned with business focus on "tail" delays (top 5%).
    ADVERSARIAL_THRESHOLD = 0.7  # Drop features with AUC > 0.7
    ZERO_IMPORTANCE_DROP = True
    STABILITY_THRESHOLD = 0.15   # Max drift allowed
    EXTREME_TARGET_QUANTILE = 0.95
    EXTREME_SAMPLE_WEIGHT = 3.0
    
    # --- Model Hyperparameters ---
    # [MAE_OPTIMIZATION] Aligned with Mission 5 results:
    # 1. regression_l1 objective for direct MAE minimization.
    # 2. learning_rate 0.1 to force stronger alignment in early trees.
    # 3. n_estimators 500 to prevent saturation and direction flipping.
    # 4. alpha 1.0 for Huber/MAE thresholding in log space.
    LGBM_PARAMS = {
        'objective': 'regression_l1',
        'alpha': 1.0,
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 127,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'seed': SEED,
        'n_estimators': 500,
        'verbose': -1,
        'n_jobs': -1,
    }
    
    RAW_LGBM_PARAMS = LGBM_PARAMS.copy()
    EMBED_LGBM_PARAMS = LGBM_PARAMS.copy()
    
    # [MISSION: 2-STAGE STRUCTURAL ARCHITECTURE]
    # Why: Single-model fails to balance precision in bulk and scale in tails.
    # Stage 1: Binary Classifier (Q90 Tail Detector)
    # Stage 2: Separate Regressors for Tail vs Non-Tail
    USE_2STAGE_MODEL = True
    BLENDING_POWER = 2.0  # Sharpening: p^2 to give more weight to tail model
    BIAS_SCALAR = 1.32    # Global scale recovery (derived from Mean Ratio 0.68)
    
    TAIL_CLASSIFIER_PARAMS = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 63,
        'seed': SEED,
        'n_estimators': 200,
        'verbose': -1,
        'n_jobs': -1,
    }
    TAIL_REGRESSOR_PARAMS = {
        'objective': 'regression', # L2 for better tail scale sensitivity
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,          # [REDUCED] 127 was causing extreme overfitting
        'min_child_samples': 50,   # [ADDED] Force statistical power per leaf
        'seed': SEED,
        'n_estimators': 200,       # [REDUCED] Prevent noise memorization
        'verbose': -1,
        'n_jobs': -1,
    }
    NON_TAIL_REGRESSOR_PARAMS = {
        'objective': 'regression_l1', # L1 for global MAE optimization
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,          # [REDUCED] 127 was too high for a 90% subset
        'min_child_samples': 30,   # [ADDED] Ensure stable leaves
        'seed': SEED,
        'n_estimators': 500,
        'verbose': -1,
        'n_jobs': -1,
    }
    


    
    # [MISSION 1 & 3: DATA-DRIVEN DRIFT ARCHITECTURE]
    # [WHY] Hardcoded thresholds (0.15, 0.70) are arbitrary and cause bias.
    # [FIX] Shift to Signal Curvature Analysis (SCA) and Iterative Pruning.
    ADVERSARIAL_THRESHOLD = 0.70  # Initial classifier limit
    STABILITY_THRESHOLD = None    # Derived dynamically via SCA in DomainShiftAudit
    ADV_PRUNING_THRESHOLD = None  # Derived dynamically via SCA in DomainShiftAudit
    ADV_TARGET_AUC = 0.75         # Target for Collective Drift Pruner
    USE_ADVERSARIAL_WEIGHTING = True
    ADV_WEIGHT_POWER = 1.0 
    
    # [MISSION 2: TAIL OPTIMIZATION]
    # [RECALIBRATION] Disabled aggressive weighting which caused mid-range collapse.
    USE_TARGET_AWARE_WEIGHTING = False 
    TARGET_AWARE_ALPHA = 0.5
    TARGET_AWARE_MAX_WEIGHT = 5.0
    
    # [GENERAL_CONTRACT]
    
    # [STRUCTURAL_REALIGNMENT]
    # EMBED_BASE_COLS now ONLY controls the input to the PCA reconstructor.
    # It NO LONGER controls the feature generation pipeline.
    EMBED_BASE_COLS = [
        'order_inflow_15m', 'unique_sku_15m', 'avg_items_per_order', 'robot_active', 
        'robot_utilization', 'avg_trip_distance', 'charge_queue_length', 'avg_charge_wait', 
        'congestion_score', 'near_collision_15m', 'manual_override_ratio', 
        'warehouse_temp_avg', 'humidity_pct', 'day_of_week', 'air_quality_idx'
    ]
    
    # PCA_INPUT_COLS is the explicit set used for PCA fit/transform
    PCA_INPUT_COLS = EMBED_BASE_COLS
    
    MULTI_K = MULTI_K
    EMBED_DIM = EMBED_DIM
    
    # ID and Target
    ID_COLS = ['ID', 'scenario_id', 'layout_id']
    TARGET = 'avg_delay_minutes_next_30m'
    
    # [STRUCTURAL_REALIGNMENT] Memory Management
    EMBED_CHUNK_SIZE = 2000
    
    @classmethod
    def rebuild_paths(cls, run_id=None):
        """Update all paths when RUN_ID changes at runtime."""
        if run_id:
            cls.RUN_ID = run_id
        cls.OUTPUT_BASE = f"./outputs/{cls.RUN_ID}"
        cls.PROCESSED_PATH = f"{cls.OUTPUT_BASE}/processed"
        cls.MODELS_PATH = f"{cls.OUTPUT_BASE}/models"
        cls.SUBMISSION_PATH = f"{cls.OUTPUT_BASE}/submission.csv"
        cls.PREDICTIONS_PATH = f"{cls.OUTPUT_BASE}/predictions"
        cls.LOG_DIR = f"./logs/{cls.RUN_ID}"
        cls.SUMMARY_DIR = f"{cls.LOG_DIR}/summary"
        cls.GLOBAL_STATS_PATH = f"{cls.PROCESSED_PATH}/global_stats.json"

    @classmethod
    def setup_directories(cls):
        dirs = [cls.PROCESSED_PATH, cls.MODELS_PATH, cls.PREDICTIONS_PATH, 
                cls.LOG_DIR, cls.SUMMARY_DIR, os.path.dirname(cls.SUBMISSION_PATH)]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
