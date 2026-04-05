import pandas as pd
import numpy as np
import logging
from .config import Config

logger = logging.getLogger(__name__)

def load_data():
    """Load train, test, and layout datasets."""
    train = pd.read_csv(f'{Config.DATA_PATH}train.csv')
    test = pd.read_csv(f'{Config.DATA_PATH}test.csv')
    layout = pd.read_csv(Config.LAYOUT_PATH)
    
    # 1. Layout Merge
    train = train.merge(layout, on='layout_id', how='left')
    test = test.merge(layout, on='layout_id', how='left')
    
    return train, test

def add_time_series_features(df, top_ts_cols):
    """Add basic TS features: Lag, Diff, Rolling."""
    df = df.copy()
    df = df.sort_values(by=['scenario_id', 'ID']).reset_index(drop=True)
    
    for col in top_ts_cols:
        # Lag
        df[f'{col}_lag1'] = df.groupby('scenario_id')[col].shift(1)
        df[f'{col}_lag2'] = df.groupby('scenario_id')[col].shift(2)
        # Diff
        df[f'{col}_diff'] = df[col] - df[f'{col}_lag1']
        # Rolling
        for w in Config.WINDOWS:
            df[f'{col}_rolling_mean{w}'] = df.groupby('scenario_id')[col].transform(lambda x: x.rolling(window=w, min_periods=1).mean())
        df[f'{col}_rolling_std3'] = df.groupby('scenario_id')[col].transform(lambda x: x.rolling(window=3, min_periods=1).std())

    return df

def add_advanced_predictive_features(df, top_ts_cols, global_stds=None):
    """Add Velocity, Trajectory Slope, Future Proxy, and Cumulative Trend."""
    df = df.copy()
    
    # Identify index within scenario (0-24)
    df['timestep_index'] = df.groupby('scenario_id').cumcount()
    
    for col in top_ts_cols:
        # 1. Velocity (Normalized & Clipped)
        diff = df[col] - df.groupby('scenario_id')[col].shift(1)
        if global_stds is not None and col in global_stds:
            g_std = global_stds[col] + 1e-9
            velocity_norm = diff / g_std
            # Clipping at 1st and 99th percentile
            p1, p99 = np.percentile(velocity_norm.dropna(), Config.VELOCITY_CLIP_PERCENTILE)
            df[f'{col}_velocity_norm'] = velocity_norm.clip(p1, p99)
        else:
            df[f'{col}_velocity'] = diff # Simple velocity fallback if no global std
            
        # 2. Future Proxy (Rolling mean shift 1)
        df[f'{col}_future_proxy'] = df.groupby('scenario_id')[f'{col}_rolling_mean3'].shift(1)
        
        # 3. Cumulative Trend (3-step mean rate of change)
        df[f'{col}_cum_trend'] = df.groupby('scenario_id')[f'{col}_diff'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        
        # 4. Trajectory Slope (5-step linear regression slope)
        # Manual formula for fixed window 5: (-2*y_t-4 - 1*y_t-3 + 0*y_t-2 + 1*y_t-1 + 2*y_t) / 10
        def calc_slope_5(x):
            if len(x) < 5: return 0
            weights = np.array([-2, -1, 0, 1, 2])
            return np.sum(weights * x.values) / 10.0
            
        df[f'{col}_trajectory_slope'] = df.groupby('scenario_id')[col].transform(lambda x: x.rolling(window=5, min_periods=5).apply(calc_slope_5, raw=False))
        df[f'{col}_trajectory_slope'] = df[f'{col}_trajectory_slope'].fillna(0)
        
    return df

def add_scenario_summary_features(df, top_ts_cols):
    """Add overall scenario intelligence: Early-Mid-Late segments and delta changes."""
    df = df.copy()
    
    # Ensure timestep_index exists
    if 'timestep_index' not in df.columns:
        df['timestep_index'] = df.groupby('scenario_id').cumcount()
        
    for col in top_ts_cols:
        # Broadcasted Summary Stats
        groups = df.groupby('scenario_id')[col]
        df[f'{col}_sc_mean'] = groups.transform('mean')
        df[f'{col}_sc_var'] = groups.transform('var')
        
        # Segment-based features (0-8: early, 9-16: mid, 17-24: late)
        def calc_segments(x):
            idx = x.index.get_level_values(0) if isinstance(x.index, pd.MultiIndex) else range(len(x))
            # We use the implicit order since we sorted by scenario and ID
            # But more robustly, we use the timestep_index if available in the group
            # However, transform passes the Series directly.
            # Let's assume the series x is already in order 0-24
            early = x.iloc[0:9]
            mid = x.iloc[9:17]
            late = x.iloc[17:25]
            
            res = {
                'early_mean': early.mean(),
                'early_var': early.var(),
                'mid_mean': mid.mean(),
                'mid_var': mid.var(),
                'late_mean': late.mean(),
                'late_var': late.var(),
                'early_slope': (early.iloc[-1] - early.iloc[0]) / 8 if len(early) > 1 else 0,
                'mid_slope': (mid.iloc[-1] - mid.iloc[0]) / 7 if len(mid) > 1 else 0,
                'late_slope': (late.iloc[-1] - late.iloc[0]) / 7 if len(late) > 1 else 0,
            }
            return res

        # Efficiently apply segment stats
        seg_stats = df.groupby('scenario_id')[col].apply(calc_segments).unstack()
        for s_col in seg_stats.columns:
            df[f'{col}_{s_col}'] = df['scenario_id'].map(seg_stats[s_col])

        # Delta features
        df[f'{col}_late_early_diff'] = df[f'{col}_late_mean'] - df[f'{col}_early_mean']
        df[f'{col}_mid_early_diff'] = df[f'{col}_mid_mean'] - df[f'{col}_early_mean']
        
        # Peak Position
        df[f'{col}_peak_step'] = groups.transform(lambda x: x.reset_index(drop=True).idxmax())

    return df

def add_binary_thresholding(df, top_ts_cols):
    """Flag features exceeding 80th percentile."""
    df = df.copy()
    for col in top_ts_cols:
        threshold = df[col].quantile(0.8)
        df[f'{col}_is_high'] = (df[col] > threshold).astype(int)
    return df

def select_top_ts_features(train):
    """Select top features for TS transformations."""
    num_cols = [c for c in train.columns if c not in Config.ID_COLS + [Config.TARGET] and np.issubdtype(train[c].dtype, np.number)]
    variances = train[num_cols].var()
    valid_cols = variances[variances > Config.VARIANCE_THRESHOLD].index.tolist()
    corrs = train[valid_cols + [Config.TARGET]].corr()[Config.TARGET].abs().sort_values(ascending=False)
    return corrs.head(Config.TS_TOP_K + 1).index.tolist()[1:]

def get_features(train, test):
    """Filter columns to get final numeric feature set."""
    feature_cols = [c for c in train.columns if c not in Config.ID_COLS + [Config.TARGET]]
    final_cols = [c for c in feature_cols if not np.issubdtype(train[c].dtype, object)]
    return final_cols
