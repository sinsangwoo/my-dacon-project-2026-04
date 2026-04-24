import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# [PHASE 1: SINGLE SOURCE OF TRUTH (SSOT)]
# ─────────────────────────────────────────────────────────────────────────────

# 1. Base Features (Raw CSV columns)
BASE_COLS = [
    'order_inflow_15m', 'unique_sku_15m', 'avg_items_per_order', 'urgent_order_ratio',
    'heavy_item_ratio', 'cold_chain_ratio', 'sku_concentration', 'robot_active',
    'robot_idle', 'robot_charging', 'robot_utilization', 'avg_trip_distance',
    'task_reassign_15m', 'battery_std', 'low_battery_ratio', 'charge_queue_length',
    'avg_charge_wait', 'congestion_score', 'max_zone_density', 'blocked_path_15m',
    'near_collision_15m', 'fault_count_15m', 'avg_recovery_time', 'replenishment_overlap',
    'pack_utilization', 'manual_override_ratio', 'warehouse_temp_avg', 'humidity_pct',
    'day_of_week', 'air_quality_idx'
]

# 2. Sequential/Extreme Suffixes
TS_SUFFIXES = [
    '_rolling_mean_3', '_rolling_mean_5', '_rolling_std_3', '_rolling_std_5',
    '_diff_1', '_diff_3', '_rate_1', '_slope_5', '_recent_max_5', '_recent_min_5',
    '_range_5', '_expanding_mean', '_expanding_sum', '_expanding_std'
]

EXTREME_SUFFIXES = [
    '_rel_to_mean_5', '_rel_to_max_5', '_rel_rank_5', '_accel',
    '_volatility_expansion_std', '_volatility_expansion_range', '_regime_id',
    '_consecutive_above_q75'
]

# 3. Latent / Embedding Features (Supercharged PCA)
EMBED_DIM = 32
MULTI_K = [10, 20, 40]
LATENT_PATTERNS = ['embed_mean', 'embed_std', 'weighted_mean', 'trend_proxy', 'volatility_proxy']

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
        
    raw_features.extend(["early_warning_flag", "early_warning_score"])
    
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
