import logging

logger = logging.getLogger(__name__)

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

Resolution Strategy:
- Feature generation must be BASE_COLS-driven, NOT PCA-driven
- Schema must EXACTLY match runtime-generated features
- PCA must become OPTIONAL (non-blocking)
- No silent feature drops allowed under any condition

This system now enforces:
- Zero tolerance for Schema-Runtime mismatch
- Deterministic feature generation
- Explicit failure instead of silent fallback

[END CONTEXT]
"""

# ─────────────────────────────────────────────────────────────────────────────
# [PHASE 1: SINGLE SOURCE OF TRUTH (SSOT)]
# ─────────────────────────────────────────────────────────────────────────────

# 1. Base Features (Raw CSV columns)
BASE_COLS = [
    'order_inflow_15m', 'unique_sku_15m', 'avg_items_per_order', 'urgent_order_ratio',
    'heavy_item_ratio', 'cold_chain_ratio', 'sku_concentration', 'robot_active',
    'robot_idle', 'robot_charging', 'robot_utilization', 'avg_trip_distance', 'task_reassign_15m',
    'battery_std', 'low_battery_ratio', 'charge_queue_length',
    'avg_charge_wait', 'congestion_score', 'max_zone_density', 'blocked_path_15m',
    'near_collision_15m', 'fault_count_15m', 'avg_recovery_time', 'replenishment_overlap',
    'pack_utilization', 'manual_override_ratio', 'warehouse_temp_avg', 'humidity_pct',
    'day_of_week', 'air_quality_idx'
]

# 2. Sequential/Extreme Suffixes
TS_SUFFIXES = [
    # [WHY_THIS_DESIGN] Canonical Trend & Stability Signals
    # Problem: 300+ features were redundant and over-engineered.
    # Why these 3: Captures State (rolling_mean), Stability (rolling_std), and Trend (diff).
    # Why others removed: expanding_mean/std and slope were >90% redundant with these.
    '_rolling_mean_5', '_rolling_std_5', '_diff_1',
    '_is_boundary'
]

EXTREME_SUFFIXES = [
    # [WHY_THIS_DESIGN] Canonical Context Signal
    # Problem: rel_rank, accel, and regime_id were redundant or noisy.
    # Why this 1: rel_to_mean_5 provides necessary situational context (is this value high for this scenario?).
    '_rel_to_mean_5'
]

# 3. Latent / Embedding Features (Supercharged PCA)
EMBED_DIM = 32
MULTI_K = [10, 20, 40]
# [WHY_THIS_DESIGN] volatility_proxy removed as it was identical to embed_std.
LATENT_PATTERNS = ['embed_mean', 'embed_std', 'weighted_mean', 'trend_proxy']

def get_feature_schema():
    """Generates the deterministic feature manifest."""
    
    # [RAW]
    raw_features = list(BASE_COLS)
    raw_features += ['timestep_index', 'normalized_time', 'cold_start_flag']
    
    for col in BASE_COLS:
        for s in TS_SUFFIXES:
            raw_features.append(f"{col}{s}")
        for s in EXTREME_SUFFIXES:
            raw_features.append(f"{col}{s}")
            
    # Interactions
    interaction_pairs = [
        ('order_inflow_15m', 'robot_utilization'),
        ('heavy_item_ratio', 'order_inflow_15m'),
        ('heavy_item_ratio', 'robot_utilization'),
    ]
    for f1, f2 in interaction_pairs:
        raw_features.append(f"inter_{f1}_x_{f2}")
        
    raw_features.extend(["early_warning_flag", "early_warning_score", "is_extreme", "is_extreme_multi"])
    
    # [EMBED]
    embed_features = []
    # Graph-based latent aggregations
    for k in MULTI_K:
        for p in LATENT_PATTERNS:
            for d in range(EMBED_DIM):
                embed_features.append(f"{p}_{k}_d{d}")
                
    # Global Proxies
    embed_features.extend(['regime_proxy', 'local_density', 'similarity_entropy'])
    
    # [TOTAL]
    all_features = raw_features + embed_features
    
    # Unique check
    seen = set()
    unique_features = []
    for f in all_features:
        if f not in seen:
            unique_features.append(f)
            seen.add(f)
            
    schema = {
        "raw_features": raw_features,
        "embed_features": embed_features,
        "all_features": unique_features,
        "feature_to_index": {feat: i for i, feat in enumerate(unique_features)}
    }
    
    logger.info(f"[SCHEMA_INIT] Raw: {len(raw_features)} | Embed: {len(embed_features)} | Total: {len(unique_features)}")
    return schema

FEATURE_SCHEMA = get_feature_schema()
