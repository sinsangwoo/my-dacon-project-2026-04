import random
import os
import logging
import gc
import numpy as np
import pandas as pd

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def get_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def log_memory_usage(message="Memory Stats"):
    """No-op version (psutil removed for stability)."""
    pass

def downcast_df(df, verbose=True):
    """Float64 -> Float32 downcasting to save 50% memory."""
    start_mem = df.memory_usage().sum() / 1024**2
    
    # 1. Float downcast
    f_cols = df.select_dtypes(include=['float64']).columns
    df[f_cols] = df[f_cols].astype('float32')
    
    # 2. Int downcast
    i_cols = df.select_dtypes(include=['int64']).columns
    df[i_cols] = df[i_cols].astype('int32')
    
    if verbose:
        end_mem = df.memory_usage().sum() / 1024**2
        print(f"Memory reduced: {start_mem:.2f}MB -> {end_mem:.2f}MB")
    return df

def save_pkl(df, path):
    """Save with immediate reload verification (Integrity Guard)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_pickle(path)
    
    # Reload test
    try:
        check = pd.read_pickle(path)
        if len(check) != len(df) or len(check.columns) != len(df.columns):
            raise ValueError("Corrupted save detected (Shape mismatch).")
        del check
        gc.collect()
    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        raise IOError(f"Pickle save failed verification for {path}: {str(e)}")

def load_pkl(path):
    """Load pickle with GC cleanup."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle not found: {path}")
    df = pd.read_pickle(path)
    gc.collect()
    return df
