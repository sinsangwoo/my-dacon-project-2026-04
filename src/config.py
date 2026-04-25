import os
import numpy as np
from datetime import datetime

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
    RUN_ID = os.getenv('RUN_ID', datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    DATA_PATH = "./data/"
    OUTPUT_BASE = f"./outputs/{RUN_ID}/"
    PROCESSED_PATH = f"{OUTPUT_BASE}processed/"
    MODELS_PATH = f"{OUTPUT_BASE}models/"
    SUBMISSION_PATH = f"{OUTPUT_BASE}submission/submission.csv"
    PREDICTIONS_PATH = f"{OUTPUT_BASE}predictions/"
    LOG_DIR = f"{OUTPUT_BASE}logs/"
    SUMMARY_DIR = f"{OUTPUT_BASE}summary/"
    LAYOUT_PATH = "./data/layout_info.csv"
    GLOBAL_STATS_PATH = f"{PROCESSED_PATH}global_stats.json"
    
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
    
    RAW_LGBM_PARAMS = LGBM_PARAMS.copy()
    EMBED_LGBM_PARAMS = LGBM_PARAMS.copy()
    
    # [EMBEDDING_CONTRACT]
    from .schema import BASE_COLS, EMBED_DIM, MULTI_K
    
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
        cls.OUTPUT_BASE = f"./outputs/{cls.RUN_ID}/"
        cls.PROCESSED_PATH = f"{cls.OUTPUT_BASE}processed/"
        cls.MODELS_PATH = f"{cls.OUTPUT_BASE}models/"
        cls.SUBMISSION_PATH = f"{cls.OUTPUT_BASE}submission/submission.csv"
        cls.PREDICTIONS_PATH = f"{cls.OUTPUT_BASE}predictions/"
        cls.LOG_DIR = f"{cls.OUTPUT_BASE}logs/"
        cls.SUMMARY_DIR = f"{cls.OUTPUT_BASE}summary/"
        cls.GLOBAL_STATS_PATH = f"{cls.PROCESSED_PATH}global_stats.json"

    @classmethod
    def setup_directories(cls):
        dirs = [cls.PROCESSED_PATH, cls.MODELS_PATH, cls.PREDICTIONS_PATH, 
                cls.LOG_DIR, cls.SUMMARY_DIR, os.path.dirname(cls.SUBMISSION_PATH)]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
