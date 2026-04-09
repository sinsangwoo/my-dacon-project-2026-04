import pandas as pd
import numpy as np
import logging
import gc
from .config import Config
from .utils import downcast_df, log_memory_usage

logger = logging.getLogger(__name__)

def load_data():
    """Load train, test, and layout datasets."""
    train = pd.read_csv(f'{Config.DATA_PATH}train.csv')
    test = pd.read_csv(f'{Config.DATA_PATH}test.csv')
    layout = pd.read_csv(Config.LAYOUT_PATH)
    
    # 1. Layout Merge
    train = train.merge(layout, on='layout_id', how='left')
    test = test.merge(layout, on='layout_id', how='left')
    
    # 2. Downcast to save memory early
    train = downcast_df(train)
    test = downcast_df(test)
    
    return train, test

def add_time_series_features(df, top_ts_cols):
    """Add basic TS features: Lag, Diff, Rolling with Optimized Concat."""
    df = df.copy()
    df = df.sort_values(by=['scenario_id', 'ID']).reset_index(drop=True)
    
    new_features = {}
    for col in top_ts_cols:
        # Lag
        new_features[f'{col}_lag1'] = df.groupby('scenario_id')[col].shift(1)
        new_features[f'{col}_lag2'] = df.groupby('scenario_id')[col].shift(2)
        # Diff
        new_features[f'{col}_diff'] = df[col] - new_features[f'{col}_lag1']
        # Rolling
        for w in Config.WINDOWS:
            new_features[f'{col}_rolling_mean{w}'] = df.groupby('scenario_id')[col].transform(lambda x: x.rolling(window=w, min_periods=1).mean())
        new_features[f'{col}_rolling_std3'] = df.groupby('scenario_id')[col].transform(lambda x: x.rolling(window=3, min_periods=1).std())

    # Single Concat to avoid fragmentation
    df = pd.concat([df, pd.DataFrame(new_features)], axis=1)
    return df

def add_advanced_predictive_features(df, top_ts_cols, global_stds=None):
    """Add Velocity, Trajectory Slope, Future Proxy, and Cumulative Trend."""
    df = df.copy()
    new_features = {}
    
    # Identify index within scenario (0-24)
    new_features['timestep_index'] = df.groupby('scenario_id').cumcount()
    
    for col in top_ts_cols:
        # 1. Velocity (Normalized & Clipped)
        diff = df[col] - df.groupby('scenario_id')[col].shift(1)
        if global_stds is not None and col in global_stds:
            g_std = global_stds[col] + 1e-9
            velocity_norm = diff / g_std
            p1, p99 = np.percentile(velocity_norm.dropna(), Config.VELOCITY_CLIP_PERCENTILE)
            new_features[f'{col}_velocity_norm'] = velocity_norm.clip(p1, p99)
        else:
            new_features[f'{col}_velocity'] = diff
            
        # 2. Future Proxy (Use existing rolling columns if available, else re-calc)
        rolling_col = f'{col}_rolling_mean3'
        if rolling_col in df.columns:
            new_features[f'{col}_future_proxy'] = df.groupby('scenario_id')[rolling_col].shift(1)
        else:
            temp_rolling = df.groupby('scenario_id')[col].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
            new_features[f'{col}_future_proxy'] = df.groupby('scenario_id')[temp_rolling].shift(1)
        
        # 3. Cumulative Trend
        diff_col = f'{col}_diff'
        if diff_col in df.columns:
            new_features[f'{col}_cum_trend'] = df.groupby('scenario_id')[diff_col].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        else:
            temp_diff = df[col] - df.groupby('scenario_id')[col].shift(1)
            new_features[f'{col}_cum_trend'] = temp_diff.rolling(window=3, min_periods=1).mean()
        
        # 4. Trajectory Slope
        def calc_slope_5(x):
            if len(x) < 5: return 0
            weights = np.array([-2, -1, 0, 1, 2])
            return np.sum(weights * x.values) / 10.0
            
        new_features[f'{col}_trajectory_slope'] = df.groupby('scenario_id')[col].transform(lambda x: x.rolling(window=5, min_periods=5).apply(calc_slope_5, raw=False)).fillna(0)
        
    df = pd.concat([df, pd.DataFrame(new_features)], axis=1)
    return df

def add_scenario_summary_features(df, top_ts_cols):
    """Add overall scenario intelligence with Optimized Concat."""
    df = df.copy()
    new_features = {}
    
    if 'timestep_index' not in df.columns:
        new_features['timestep_index'] = df.groupby('scenario_id').cumcount()
        
    # We must add timestep_index to df immediately if it's new, as other logic might need it
    if 'timestep_index' in new_features:
        df['timestep_index'] = new_features.pop('timestep_index')

    for col in top_ts_cols:
        groups = df.groupby('scenario_id')[col]
        new_features[f'{col}_sc_mean'] = groups.transform('mean').astype('float32')
        new_features[f'{col}_sc_var'] = groups.transform('var').astype('float32')
        
        def calc_segments(x):
            early = x.iloc[0:9]
            mid = x.iloc[9:17]
            late = x.iloc[17:25]
            return {
                'early_mean': early.mean(), 'early_var': early.var(),
                'mid_mean': mid.mean(), 'mid_var': mid.var(),
                'late_mean': late.mean(), 'late_var': late.var(),
                'early_slope': (early.iloc[-1] - early.iloc[0]) / 8 if len(early) > 1 else 0,
                'mid_slope': (mid.iloc[-1] - mid.iloc[0]) / 7 if len(mid) > 1 else 0,
                'late_slope': (late.iloc[-1] - late.iloc[0]) / 7 if len(late) > 1 else 0,
            }

        seg_stats = df.groupby('scenario_id')[col].apply(calc_segments).unstack()
        for s_col in seg_stats.columns:
            new_features[f'{col}_{s_col}'] = df['scenario_id'].map(seg_stats[s_col]).astype('float32')

        new_features[f'{col}_late_early_diff'] = (new_features[f'{col}_late_mean'] - new_features[f'{col}_early_mean']).astype('float32')
        new_features[f'{col}_mid_early_diff'] = (new_features[f'{col}_mid_mean'] - new_features[f'{col}_early_mean']).astype('float32')
        new_features[f'{col}_peak_step'] = groups.transform(lambda x: x.reset_index(drop=True).idxmax()).astype('int16')
        
        del seg_stats; gc.collect()

    # Inplace-like concat
    for k, v in new_features.items():
        df[k] = v
    del new_features; gc.collect()
    return df

def add_sequence_trajectory_features(df, top_ts_cols):
    """Add advanced sequence-aware trajectory features.
    NO df.copy() - operates directly on input df columns.
    """
    logger.info(f"[SEQUENCE_FE] Starting column-wise trajectory generation...")
    
    for col in top_ts_cols:
        ev, mv, lv = f'{col}_early_var', f'{col}_mid_var', f'{col}_late_var'
        es, ms, ls = f'{col}_early_slope', f'{col}_mid_slope', f'{col}_late_slope'
        em, mm, lm = f'{col}_early_mean', f'{col}_mid_mean', f'{col}_late_mean'
        sv = f'{col}_sc_var'
        
        if ev in df.columns:
            df[f'{col}_seq_early_std'] = np.sqrt(df[ev].fillna(0).clip(lower=0)).astype('float32')
            df[f'{col}_seq_mid_std'] = np.sqrt(df[mv].fillna(0).clip(lower=0)).astype('float32')
            df[f'{col}_seq_late_std'] = np.sqrt(df[lv].fillna(0).clip(lower=0)).astype('float32')
            df[f'{col}_seq_volatility_ratio'] = (df[lv].fillna(0) / (df[ev].fillna(0) + 1e-9)).astype('float32')
        
        if es in df.columns and ls in df.columns:
            df[f'{col}_seq_slope_accel'] = (df[ls].fillna(0) - df[es].fillna(0)).astype('float32')
            df[f'{col}_seq_slope_accel_mid'] = (df[ms].fillna(0) - df[es].fillna(0)).astype('float32')
        
        if em in df.columns and lm in df.columns:
            global_std = np.sqrt(df[sv].fillna(0).clip(lower=0)) + 1e-9 if sv in df.columns else np.float32(1.0)
            df[f'{col}_seq_trend_strength'] = ((df[lm].fillna(0) - df[em].fillna(0)) / global_std).astype('float32')
            
            if mm in df.columns:
                late_delta = df[lm].fillna(0) - df[mm].fillna(0)
                early_delta = df[mm].fillna(0) - df[em].fillna(0)
                df[f'{col}_seq_momentum_diff'] = (late_delta - early_delta).astype('float32')
                df[f'{col}_seq_momentum_ratio'] = (late_delta / (early_delta.abs() + 1e-9)).astype('float32')
        
        # Periodic cleanup during heavy loop
        if Config.MODE == 'full': gc.collect()
    
    return df

def add_binary_thresholding(df, top_ts_cols):
    """Flag features exceeding 80th percentile with Optimized Concat."""
    df = df.copy()
    new_features = {}
    for col in top_ts_cols:
        threshold = df[col].quantile(0.8)
        new_features[f'{col}_is_high'] = (df[col] > threshold).astype(np.int32)
    df = pd.concat([df, pd.DataFrame(new_features)], axis=1)
    return df

def add_nan_flags(df, primary_cols):
    """Add binary flags for NaN positions in critical features."""
    df = df.copy()
    new_flags = {}
    for col in primary_cols:
        if col in df.columns:
            new_flags[f'{col}_nan_flag'] = df[col].isna().astype(np.int8)
    return pd.concat([df, pd.DataFrame(new_flags)], axis=1)

def handle_engineered_nans(df, feature_cols):
    """Centralized high-fidelity imputation policy v4.3."""
    df = df.copy()
    
    # 1. Category: Mean-based (ffill within scenario)
    mean_cols = [c for c in feature_cols if 'rolling_mean' in c or 'sc_mean' in c or 'proxy' in c]
    for col in mean_cols:
        if col in df.columns:
            # ffill within scenario_id to preserve local state
            df[col] = df.groupby('scenario_id')[col].ffill()
    
    # 2. Category: Delta/Stability/Slope (zero-fill)
    # This covers: lag, diff, std, var, slope, velocity
    delta_cols = [c for c in feature_cols if any(x in c for x in ['lag', 'diff', 'std', 'var', 'slope', 'velocity', 'trend'])]
    fill_values = {c: 0 for c in delta_cols if c in df.columns}
    
    # Also add remaining mean_cols to zero-fill if ffill didn't catch everything
    for col in mean_cols:
        if col in df.columns:
            df[col].fillna(0, inplace=True)
            
    # Final catch-all for any remaining NaNs
    for col in feature_cols:
        if df[col].isna().any():
            df[col].fillna(0, inplace=True)
            
    return df

def select_top_ts_features(train):
    """Select top features for TS transformations."""
    # Ensure numeric columns only and handle NaNs for calculation
    num_cols = train.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if c not in Config.ID_COLS + [Config.TARGET]]
    
    # Temporary fill for variance/corr calculation
    temp_train = train[num_cols + [Config.TARGET]].fillna(0)
    
    variances = temp_train[num_cols].var()
    valid_cols = variances[variances > Config.VARIANCE_THRESHOLD].index.tolist()
    corrs = temp_train[valid_cols + [Config.TARGET]].corr()[Config.TARGET].abs().sort_values(ascending=False)
    return corrs.head(Config.TS_TOP_K + 1).index.tolist()[1:]

def get_features(train, test):
    """Filter columns to get final numeric feature set."""
    feature_cols = [c for c in train.columns if c not in Config.ID_COLS + [Config.TARGET]]
    final_cols = train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    return final_cols

def align_features(df, reference_cols, logger=None):
    """Memory-Safe Feature Alignment (v6.0 Inplace-First).
    
    Builds the target DataFrame column-wise, dropping used columns from source immediately
    to restrict the peak memory barrier.
    """
    if logger:
        logger.info(f"── Memory-Safe Alignment (v6.0) ──")
        logger.info(f"  Target columns:  {len(reference_cols)}")

    # 1. Create Empty Skeleton with same index
    final_df = pd.DataFrame(index=df.index)
    
    # 2. Process Reference Columns one-by-one
    target_set = set(reference_cols)
    source_cols = list(df.columns)
    
    for col in reference_cols:
        if col in df.columns:
            # Transfer and convert to float32
            final_df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('float32')
            # Drop from source to free memory
            df.drop(columns=[col], inplace=True)
        else:
            # Fill missing with zeros
            final_df[col] = np.zeros(len(df), dtype='float32')
            
        # Periodic cleanup
        if len(final_df.columns) % 100 == 0:
            gc.collect()

    if logger:
        logger.info(f"  ✓ Column-wise transfer complete. Cleaning up source...")

    # 3. STRICT POST-CONDITIONS (as before)
    assert final_df.shape[1] == len(reference_cols)
    assert list(final_df.columns) == reference_cols
    assert not final_df.isna().any().any()
    
    if logger:
        logger.info(f"  ✓ Alignment complete: shape={final_df.shape}")
    
    return final_df
