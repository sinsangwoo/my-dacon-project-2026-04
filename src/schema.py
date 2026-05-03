import logging
import itertools

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
    # [WHY_THIS_DESIGN] Multi-Scale State + Trend Signals (TASK 7/8/9 — Underfitting Recovery)
    # Problem: Single window (5) and single trend signal (diff_1) collapsed feature space,
    #   losing orthogonal temporal information and causing underfitting.
    # Why multi-scale: 3-step window captures fast dynamics (45min at 15min intervals),
    #   5-step captures medium dynamics (75min). Different time horizons provide
    #   orthogonal information about acceleration vs steady-state behavior.
    # Why slope_5: Linear regression slope over 5 steps captures trend DIRECTION,
    #   complementing diff_1 which only captures instantaneous change.
    # Why rate_1: Normalized change (diff/magnitude) captures RELATIVE change,
    #   which is invariant to feature scale — e.g., +10 orders when baseline is 100
    #   vs +10 when baseline is 1000 carry different semantic weight.
    # Why accel REJECTED: 2nd derivative with 15min granularity and 10-17% NaN
    #   produces 46%+ NaN and amplifies sensor noise. Empirically >90% redundant with diff_1.
    # [FAILURE_MODE_PREVENTED] Underfitting from feature space collapse.
    '_rolling_mean_3', '_rolling_std_3',   # Short-window (45min) state & stability
    '_rolling_mean_5', '_rolling_std_5',   # Medium-window (75min) state & stability
    '_diff_1',                              # Absolute first-order change
    '_slope_5',                             # Linear trend direction over 5 steps
    '_rate_1'                               # Normalized change (relative to magnitude)
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
    raw_features += ['timestep_index', 'normalized_time', 'cold_start_flag', 'is_scenario_boundary']
    
    for col in BASE_COLS:
        for s in TS_SUFFIXES:
            raw_features.append(f"{col}{s}")
        for s in EXTREME_SUFFIXES:
            raw_features.append(f"{col}{s}")
            
    # [STRUCTURAL_FIX] Dynamic Interaction Engine (v16.5)
    # [ROOT_CAUSE] Previous hardcoded logic (3 features) caused Interaction=0 survivors.
    # [SOLUTION] Iterate through top sensor pairs to explore the sensor interaction space.
    interaction_targets = ['order_inflow_15m', 'robot_utilization', 'congestion_score', 'battery_std', 'heavy_item_ratio']
    # [SSOT_FIX] Local import removed
    for f1, f2 in itertools.combinations(interaction_targets, 2):
        raw_features.append(f"inter_{f1}_x_{f2}")
        raw_features.append(f"ratio_{f1}_to_{f2}")
        raw_features.append(f"diff_{f1}_{f2}")
        raw_features.append(f"logprod_{f1}_{f2}")
        raw_features.append(f"bucket_{f1}_x_{f2}")
        
    raw_features.extend(["early_warning_flag", "early_warning_score", "is_extreme", "is_extreme_multi"])
    
    # [MISSION 3] Physical Features
    raw_features.extend(["surge_velocity", "load_utilization_ratio", "charging_stress", "robot_density", "sku_density", 
                        "congestion_surge", "battery_starvation"])
    
    # [MISSION: SCENARIO CONTEXT - CAUSAL]
    raw_features.extend(["scenario_max_historic", "scenario_volatility_causal"])
    
    # [LAYOUT_AGGREGATION] Task 2.3
    layout_target_cols = ['order_inflow_15m', 'robot_utilization', 'congestion_score', 'avg_trip_distance', 'pack_utilization']
    for col in layout_target_cols:
        raw_features.append(f"{col}_layout_mean")
        raw_features.append(f"{col}_layout_std")
    
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
