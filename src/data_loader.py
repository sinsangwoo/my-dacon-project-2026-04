import pandas as pd
import numpy as np
import logging
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
        new_features[f'{col}_sc_mean'] = groups.transform('mean')
        new_features[f'{col}_sc_var'] = groups.transform('var')
        
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
            new_features[f'{col}_{s_col}'] = df['scenario_id'].map(seg_stats[s_col])

        new_features[f'{col}_late_early_diff'] = new_features[f'{col}_late_mean'] - new_features[f'{col}_early_mean']
        new_features[f'{col}_mid_early_diff'] = new_features[f'{col}_mid_mean'] - new_features[f'{col}_early_mean']
        new_features[f'{col}_peak_step'] = groups.transform(lambda x: x.reset_index(drop=True).idxmax())

    df = pd.concat([df, pd.DataFrame(new_features)], axis=1)
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
    
    # Also add remaining mean_cols to zero-fill if ffill didn't catch everything (start of scenario)
    for col in mean_cols:
        if col in df.columns:
            fill_values[col] = 0
            
    df = df.fillna(value=fill_values)
    
    # Final catch-all for any remaining NaNs (e.g. from merge or unexpected sources)
    if df[feature_cols].isna().any().any():
        df[feature_cols] = df[feature_cols].fillna(0)
        
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
    """Force a DataFrame to match a reference feature list exactly (v5.1 SSOT).
    
    This is the SINGLE SOURCE OF TRUTH for schema enforcement.
    Phase 2 calls this ONCE before saving. Phase 3+ NEVER calls this.
    """
    df = df.copy()
    
    # 1. Standardize column names
    reference_cols = [str(c) for c in reference_cols]
    
    current_cols = set(df.columns)
    target_cols = set(reference_cols)
    
    missing = sorted(target_cols - current_cols)
    extra = sorted(current_cols - target_cols)
    
    if logger:
        logger.info(f"── Feature Alignment Report (v5.1 SSOT) ──")
        logger.info(f"  Input columns:   {len(current_cols)}")
        logger.info(f"  Target columns:  {len(reference_cols)}")
        logger.info(f"  Missing columns: {len(missing)} → filled with 0")
        logger.info(f"  Extra columns:   {len(extra)} → dropped")
        if missing: logger.info(f"  Missing examples: {missing[:5]}")
        if extra: logger.info(f"  Extra examples:   {extra[:5]}")
        
    # 2. Add missing as 0
    for col in missing:
        df[col] = 0.0
        
    # 3. Handle object types BEFORE selection
    for col in reference_cols:
        if pd.api.types.is_object_dtype(df[col]):
            if logger: logger.warning(f"  Object dtype detected in '{col}' → forcing numeric coercion")
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
    # 4. Select EXACTLY in reference order → drop extras by exclusion
    final_df = df[reference_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
    
    # 5. STRICT POST-CONDITIONS
    assert final_df.shape[1] == len(reference_cols), \
        f"Alignment shape mismatch! Got {final_df.shape[1]}, expected {len(reference_cols)}"
    assert list(final_df.columns) == reference_cols, \
        "Alignment column order mismatch!"
    assert not (final_df.dtypes == object).any(), \
        f"Object dtype leaked through alignment! {final_df.dtypes[final_df.dtypes == object].index.tolist()}"
    assert final_df.dtypes.apply(lambda d: d in [np.float32, np.float64]).all(), \
        f"Non-float dtype detected! {final_df.dtypes.unique()}"
    assert not final_df.isna().any().any(), \
        f"NaN survived alignment! Count: {final_df.isna().sum().sum()}"
    
    if logger:
        logger.info(f"  ✓ Alignment complete: shape={final_df.shape}, dtype={final_df.dtypes.unique()}")
    
    return final_df
