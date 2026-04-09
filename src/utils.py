import random
import os
import logging
import gc
import time
import traceback
import subprocess
import numpy as np
import pandas as pd
from .config import Config

# Forensic Global Tracking
CUMULATIVE_MEMORY_MB = 0.0

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def get_system_ram():
    """Get total system RAM in MB (v5.3)."""
    total_ram = 16384.0 # Default 16GB
    try:
        if os.name == 'nt':
            cmd = 'wmic computersystem get TotalPhysicalMemory'
            out = subprocess.check_output(cmd, shell=True).decode().split()
            if len(out) >= 2: total_ram = float(out[1]) / (1024 * 1024)
        else:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        total_ram = float(line.split()[1]) / 1024.0
                        break
    except: pass
    return total_ram

def get_logger():
    # Force INFO level and consistent format for forensic tracing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def get_process_memory():
    """Get current process RSS memory in MB (Forensic Trace v5.3)."""
    pid = os.getpid()
    try:
        if os.name == 'nt': # Windows
            # Use wmic for faster/cleaner output than tasklist
            cmd = f'wmic process where "ProcessID={pid}" get WorkingSetSize'
            output = subprocess.check_output(cmd, shell=True).decode().split()
            if len(output) >= 2:
                return float(output[1]) / (1024 * 1024)
        else: # Linux/WSL
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return float(line.split()[1]) / 1024.0
    except:
        pass
    return 0.0

def log_memory_usage(data, label, logger):
    """Memory Tracking with Process-level Telemetry (v5.3)."""
    obj_mb = 0.0
    if isinstance(data, pd.DataFrame):
        obj_mb = data.memory_usage(deep=True).sum() / 1024**2
    elif isinstance(data, np.ndarray):
        obj_mb = data.nbytes / 1024**2
    
    global CUMULATIVE_MEMORY_MB
    CUMULATIVE_MEMORY_MB += obj_mb
    
    proc_mb = get_process_memory()
    
    logger.info(f"[MEMORY_TRACE] {label} | Obj: {obj_mb:.2f} MB | Process: {proc_mb:.2f} MB | Cumulative: {CUMULATIVE_MEMORY_MB:.2f} MB")
    
    if obj_mb > Config.MEMORY_THRESHOLD_GB * 1024:
        logger.warning(f"[MEMORY_ALERT] Object '{label}' ({obj_mb:.2f} MB) exceeds threshold!")
    
    if proc_mb > Config.MEMORY_WARN_THRESHOLD_MB:
        logger.warning(f"[MEMORY_CRITICAL] Process Memory ({proc_mb:.2f} MB) dangerously high!")

def log_forensic_snapshot(data, label, logger, prev_cols=None):
    """Capture full DATA_SNAPSHOT and FEATURE_SNAPSHOT for audit."""
    logger.info(f"\n--- [DATA_SNAPSHOT] {label} ---")
    
    # 1. Basic Stats
    shape = data.shape
    dtype_dist = "N/A"
    nans = 0
    
    if isinstance(data, pd.DataFrame):
        dtype_dist = data.dtypes.value_counts().to_dict()
        nans = data.isna().sum().sum()
        cols = list(data.columns)
    else:
        dtype_dist = {str(data.dtype): 1}
        nans = np.isnan(data).sum() if np.issubdtype(data.dtype, np.number) else 0
        cols = None
        
    logger.info(f"Shape: {shape} | NaNs: {nans}")
    logger.info(f"Dtypes: {dtype_dist}")
    
    # 2. Feature Tracking
    if cols:
        logger.info(f"[FEATURE_SNAPSHOT] Count: {len(cols)}")
        logger.info(f"Columns (First 5): {cols[:5]}")
        logger.info(f"Columns (Last 5): {cols[-5:]}")
        
        if prev_cols is not None:
            delta = len(cols) - prev_cols
            logger.info(f"[FEATURE_DELTA] Previous: {prev_cols} | Current: {len(cols)} | Delta: {delta}")
            
    # 3. Memory Tracking
    log_memory_usage(data, label, logger)
    logger.info(f"--- [/DATA_SNAPSHOT] ---\n")
    return len(cols) if cols else 0

def downcast_df(df, verbose=True):
    """Float64 -> Float32 downcasting to save 50% memory."""
    start_mem = df.memory_usage().sum() / 1024**2
    
    f_cols = df.select_dtypes(include=['float64']).columns
    df[f_cols] = df[f_cols].astype('float32')
    
    i_cols = df.select_dtypes(include=['int64']).columns
    df[i_cols] = df[i_cols].astype('int32')
    
    if verbose:
        end_mem = df.memory_usage().sum() / 1024**2
        print(f"Memory reduced: {start_mem:.2f}MB -> {end_mem:.2f}MB")
    return df

def save_pkl(data, path):
    """Save with standard I/O trace."""
    logger = get_logger()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info(f"[FILE_IO_TRACE] SAVE | Path: {path} | Type: {type(data)}")
    data.to_pickle(path)
    logger.info(f"✓ SAVE Success: {path}")

def load_pkl(path):
    """Load with standard I/O trace."""
    logger = get_logger()
    logger.info(f"[FILE_IO_TRACE] LOAD | Path: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle not found: {path}")
    data = pd.read_pickle(path)
    logger.info(f"✓ LOAD Success: {path} | Shape: {data.shape}")
    gc.collect()
    return data

def save_npy(data, path):
    """Numpy save with traceability."""
    logger = get_logger()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info(f"[FILE_IO_TRACE] SAVE_NPY | Path: {path} | Shape: {data.shape}")
    np.save(path, data)
    logger.info(f"✓ SAVE_NPY Success: {path}")

def load_npy(path, allow_pickle=False):
    """Numpy load with traceability."""
    logger = get_logger()
    logger.info(f"[FILE_IO_TRACE] LOAD_NPY | Path: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPY not found: {path}")
    data = np.load(path, allow_pickle=allow_pickle)
    logger.info(f"✓ LOAD_NPY Success: {path} | Shape: {data.shape}")
    return data

class PhaseTracer:
    """Standardized Forensic Phase Wrapper (v5.2)."""
    def __init__(self, phase_name, logger):
        self.phase = phase_name
        self.logger = logger
        self.start_time = None
        self.last_checkpoint = "PHASE_START"

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"\n{'='*50}\n[PHASE_START] {self.phase}\nTimestamp: {time.ctime()}\n{'='*50}")
        return self

    def checkpoint(self, name):
        self.last_checkpoint = name
        self.logger.info(f"[CHECKPOINT] {name} arrived at {time.ctime()}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type:
            self.logger.error(f"[PHASE_FAILED] {self.phase}")
            self.logger.error(f"Last Successful Checkpoint: {self.last_checkpoint}")
            self.logger.error(f"Reason: {str(exc_val)}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            return False # Re-raise
        else:
            self.logger.info(f"[PHASE_END] {self.phase} | Duration: {duration:.2f}s | Status: SUCCESS\n{'='*50}\n")
            return True
