import random
import os
import logging
import gc
import time
import traceback
import subprocess
import hashlib
import json
import shutil
import warnings
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error
from .config import Config

# Forensic Global Tracking
CUMULATIVE_MEMORY_MB = 0.0

import glob

def check_model_contract_compliance():
    """HARD FAIL POLICY: Scan entire codebase for direct model.fit/predict calls (v13.0).
    Implements Scoped Fit Sandbox to allow preprocessing fit calls in src/ modules.
    """
    # [MISSION: SAFE MODEL FIT SANDBOX] Configuration
    SAFE_FIT_MODULES = ["src/data_loader.py", "src/embedding", "src/preprocessing", "src/utils.py"]
    SAFE_PREPROCESSING_PATTERNS = [
        "scaler.fit(", "pca.fit(", "kmeans.fit(", "imputer.fit(", 
        "reconstructor.fit(", "self.global_imputer.fit(", "self.pca_8.fit(", 
        "self.pca_16.fit(", "self.kmeans.fit(", "self.scaler.fit(",
        "self.ae.train(", "fit_transform(", "self.kmeans.predict(", "kmeans.predict(",
        "self.pca_raw.fit(", "self.pca_log.fit(", "self.pca_rank.fit(", 
        "self.global_pca_local.fit(", "lpca.fit("
    ]
    
    # Patterns to catch direct model calls
    forbidden = [".fit(", ".predict(", ".predict_proba("]
    
    files = glob.glob("**/*.py", recursive=True)
    violations = []
    
    for file_path in files:
        f_path_norm = file_path.replace("\\", "/")
        
        # [GATEWAY_EXEMPTION] The protection implementation itself must be exempt
        if "src/utils.py" in f_path_norm:
            continue
            
        # [SCRATCH AUTO-BLOCK] Automatically ignore scratch and experimental scripts 
        # to ensure they don't block main pipeline execution.
        if "scratch/" in f_path_norm or "experimental/" in f_path_norm:
            continue
            
        # [SAFE ZONE] Allow specific modules to use preprocessing fits
        is_safe_module = any(m in f_path_norm for m in SAFE_FIT_MODULES)
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                        
                    # Check for direct calls
                    for p in forbidden:
                        if p in stripped:
                            # 1. Is it using the SAFE_ gateway? (Always allowed)
                            if f"SAFE_{p[1:]}" in stripped:
                                continue
                            
                            # 2. Is it a whitelisted Preprocessing Fit in a Safe Module?
                            if is_safe_module and not is_scratch:
                                is_preproc = any(pp in stripped for pp in SAFE_PREPROCESSING_PATTERNS)
                                if is_preproc:
                                    continue
                            
                            # [VIOLATION] Either in Restricted Zone without gateway, or in Forbidden Zone
                            violations.append(f"{file_path}:{i} | {stripped}")
        except Exception:
            continue
            
    if violations:
        msg = "[UNSAFE_MODEL_CALL_DETECTED] Direct model calls are strictly forbidden.\n"
        msg += "  (Allowed: Preprocessing fits in src/ only via whitelisted patterns)\n"
        msg += "\n".join([f"  Violation at {v}" for v in violations])
        logging.getLogger(__name__).error(msg)
        raise RuntimeError("[UNSAFE_MODEL_CALL_DETECTED]")

def SAFE_FIT(model, X, y, **kwargs):
    """SAFE GATEWAY for model.fit (v12.0)
    Enforces numpy-only, float32-only, 2D-only inputs.
    Redefined Contract: Allows only LGBM auto-generated 'Column_i' names.
    """
    logger = logging.getLogger(__name__)
    
    # [1. INPUT VALIDATION]
    if isinstance(X, pd.DataFrame):
        raise RuntimeError("[GLOBAL_CONTRACT_VIOLATION] DataFrame passed to model.fit")
    
    assert isinstance(X, np.ndarray), f"[GLOBAL_CONTRACT_VIOLATION] Expected np.ndarray, got {type(X)}"
    assert X.dtype == np.float32, f"[GLOBAL_CONTRACT_VIOLATION] Expected float32, got {X.dtype}"
    assert X.ndim == 2, f"[GLOBAL_CONTRACT_VIOLATION] Expected 2D array, got {X.ndim}D"
    
    # [2. EXECUTE FIT]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.fit(X, y, **kwargs)
        # Filter out environmental/platform warnings that don't indicate data/model contract violations
        w = [warn for warn in w if "Could not find the number of physical cores" not in str(warn.message)]
        
        if len(w) > 0:
            for warn in w:
                logger.error(f"[GLOBAL_CONTRACT_WARNING] {warn.category.__name__}: {warn.message}")
            raise RuntimeError(f"[GLOBAL_CONTRACT_VIOLATION] Model fit produced {len(w)} warnings. Zero warning policy enforced.")
    
    # [3. REDEFINED FEATURE NAME CONTRACT VALIDATION]
    n_features = X.shape[1]
    
    def validate_names(names, source_tag):
        if names is None: return
        if len(names) != n_features:
            raise RuntimeError(f"[ILLEGAL_FEATURE_NAME_SOURCE_DETECTED] {source_tag} length mismatch: {len(names)} != {n_features}")
        
        for i, name in enumerate(names):
            expected = f"Column_{i}"
            if name != expected:
                logger.error(f"[CONTRACT_FAIL] Illegal name detected: '{name}' at index {i}. Expected: '{expected}'")
                raise RuntimeError(f"[ILLEGAL_FEATURE_NAME_SOURCE_DETECTED] {source_tag} contains non-LGBM name: '{name}'")

    # Check sklearn-level feature_names_in_
    if hasattr(model, "feature_names_in_"):
        logger.info(f"[CONTRACT_CHECK] Validating feature_names_in_ for {n_features} features...")
        validate_names(model.feature_names_in_, "feature_names_in_")
        logger.info("[CONTRACT_PASS] feature_names_in_ matches LGBM auto-generation pattern.")

    # Check LightGBM-level booster names
    if "LGBM" in str(type(model)) and hasattr(model, "booster_"):
        logger.info("[CONTRACT_CHECK] Validating booster.feature_name()...")
        validate_names(model.booster_.feature_name(), "booster.feature_name()")
        logger.info("[CONTRACT_PASS] booster.feature_name() matches LGBM auto-generation pattern.")

    # Check n_features_in_ consistency
    if hasattr(model, "n_features_in_"):
        assert model.n_features_in_ == n_features, f"[CONTRACT_FAIL] n_features_in_ mismatch: {model.n_features_in_} != {n_features}"

def SAFE_PREDICT(model, X):
    """SAFE GATEWAY for model.predict (v12.0)
    Enforces numpy-only, float32-only inputs.
    """
    if isinstance(X, pd.DataFrame):
        raise RuntimeError("[GLOBAL_CONTRACT_VIOLATION] DataFrame passed to model.predict")
    
    # Enforce contract
    assert isinstance(X, np.ndarray), f"[GLOBAL_CONTRACT_VIOLATION] Expected np.ndarray, got {type(X)}"
    assert X.dtype == np.float32, f"[GLOBAL_CONTRACT_VIOLATION] Expected float32, got {X.dtype}"
    
    return model.predict(X)

def SAFE_PREDICT_PROBA(model, X):
    """SAFE GATEWAY for model.predict_proba (v12.0)
    Enforces numpy-only, float32-only inputs.
    """
    if isinstance(X, pd.DataFrame):
        raise RuntimeError("[GLOBAL_CONTRACT_VIOLATION] DataFrame passed to model.predict_proba")
    
    # Enforce contract
    assert isinstance(X, np.ndarray), f"[GLOBAL_CONTRACT_VIOLATION] Expected np.ndarray, got {type(X)}"
    assert X.dtype == np.float32, f"[GLOBAL_CONTRACT_VIOLATION] Expected float32, got {X.dtype}"
    
    return model.predict_proba(X)

def generate_submission_fingerprint(df, file_path):
    """Generate structural fingerprint for submission (v11.0)."""
    try:
        with open(file_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        fingerprint = {
            "column_list": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "row_count": len(df),
            "null_count": int(df.isnull().sum().sum()),
            "first_5_rows": df.head(5).to_dict(orient='records'),
            "last_5_rows": df.tail(5).to_dict(orient='records'),
            "file_hash": file_hash,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        return fingerprint
    except Exception as e:
        logging.getLogger(__name__).error(f"[FINGERPRINT_ERR] Failed: {str(e)}")
        return {"error": str(e)}

def validate_submission(df, sample_df, logger):
    """Strict Contract Validator for Dacon Submissions (v11.0)."""
    errors = []
    
    # [1] Column Match
    if list(df.columns) != list(sample_df.columns):
        errors.append(f"Column mismatch! Expected {list(sample_df.columns)}, got {list(df.columns)}")
    
    # [2] Row Count
    if len(df) != len(sample_df):
        errors.append(f"Row count mismatch! Expected {len(sample_df)}, got {len(df)}")
    
    # [3] ID Match (Order & Values)
    if not (df['ID'].values == sample_df['ID'].values).all():
        errors.append("ID column order or values do not match sample_submission!")
        # Check if at least the set is the same
        if set(df['ID']) != set(sample_df['ID']):
            errors.append("ID set mismatch! Missing or extra IDs found.")
    
    # [4] Dtypes (Target must be numeric, ID must be object/str)
    if not np.issubdtype(df[Config.TARGET].dtype, np.number):
        errors.append(f"Target column {Config.TARGET} is not numeric: {df[Config.TARGET].dtype}")
    if df['ID'].dtype != object and df['ID'].dtype != 'O':
         # Pandas often represents strings as object, but sometimes 'string' dtype in newer versions
         if not pd.api.types.is_string_dtype(df['ID']):
            errors.append(f"ID column is not object/string: {df['ID'].dtype}")

    # [5] NaN / Inf Check
    nans = df.isnull().sum().sum()
    if nans > 0:
        errors.append(f"Found {nans} NaN values in submission!")
    
    infs = np.isinf(df.select_dtypes(include=np.number)).sum().sum()
    if infs > 0:
        errors.append(f"Found {infs} Inf values in submission!")

    # [6] Float Precision Check (Optional warning)
    # If values are suspiciously integers or have too many digits, log it
    
    if errors:
        for err in errors:
            logger.error(f"[VALIDATION_FAILED] {err}")
        raise RuntimeError(f"Submission validation failed with {len(errors)} errors. Check logs.")
    
    logger.info("[VALIDATION_PASSED] Submission contract verified 100%.")

def save_submission_trace(trace_data, path):
    """Save forensic trace of submission generation."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace_data, f, indent=2)

def preserve_fail_case(file_path, run_id):
    """Copy failed submission to metadata/failed_submissions/ (v11.0)."""
    fail_dir = os.path.join("metadata", "failed_submissions", run_id)
    os.makedirs(fail_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dest_path = os.path.join(fail_dir, f"submission_fail_{timestamp}.csv")
    shutil.copy2(file_path, dest_path)
    return dest_path

def build_metrics(y_true, y_pred):
    """
    PURPOSE: [PHASE 2: DETERMINISTIC METRIC BUILDER]
    Computes a fixed set of performance metrics for model evaluation.
    
    DEFINITION:
    - mean_mae: Global Mean Absolute Error.
    - worst_mae: MAE of the top 10% highest error samples (measures prediction tail).
    - extreme_mae: MAE where target (y_true) is in the top 5% (measures tail risk).
    - variance_ratio: Ratio of prediction standard deviation to target standard deviation.
    
    INPUT:
    - y_true: numpy.ndarray (Ground truth targets)
    - y_pred: numpy.ndarray (Model predictions)
    
    OUTPUT:
    - metrics: dict matching Config.METRIC_SCHEMA exactly.
    
    FAILURE:
    - Raises ValueError if input shapes mismatch or are empty.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError(f"[METRIC_BUILD_FAILED] Shape mismatch or empty: y_true={len(y_true)}, y_pred={len(y_pred)}")
        
    # 1. mean_mae
    mean_mae = float(mean_absolute_error(y_true, y_pred))
    
    # 2. worst_mae (Prediction Error Tail - Top 10%)
    errors = np.abs(y_true - y_pred)
    worst_threshold = np.quantile(errors, 0.9)
    worst_mae = float(np.mean(errors[errors >= worst_threshold]))
    
    # 3. extreme_mae (Target Tail - Top 5%)
    extreme_threshold = np.quantile(y_true, 0.95)
    mask_extreme = y_true >= extreme_threshold
    if mask_extreme.any():
        extreme_mae = float(mean_absolute_error(y_true[mask_extreme], y_pred[mask_extreme]))
    else:
        extreme_mae = mean_mae # Fallback if no extreme samples (unlikely)
        
    # 4. variance_ratio
    std_pred = np.std(y_pred)
    std_true = np.std(y_true)
    variance_ratio = float(std_pred / (std_true + 1e-9))
    
    metrics = {
        "mean_mae": mean_mae,
        "worst_mae": worst_mae,
        "extreme_mae": extreme_mae,
        "variance_ratio": variance_ratio
    }
    
    # [PHASE 3: ASSERTION FIREWALL]
    for key in Config.METRIC_SCHEMA:
        if key not in metrics:
            raise RuntimeError(f"[METRIC_MISSING_FATAL] Metric '{key}' missing from builder output!")
            
    # [PHASE 6: DEBUG TRACE]
    logger = logging.getLogger("METRIC_BUILDER")
    logger.info(f"[METRIC_BUILD_SUCCESS] mean={mean_mae:.4f} | worst={worst_mae:.4f} | extreme={extreme_mae:.4f} | var={variance_ratio:.4f}")
    
    return metrics

def build_submission(preds, ids):
    """SINGLE AUTHORITATIVE FUNCTION for submission creation.
    
    Contract:
    - MUST apply np.round(preds, 3)
    - MUST enforce float64 dtype
    - MUST use float_format="%.3f"
    - MUST reload and verify equality
    """
    preds = np.asarray(preds, dtype=np.float64)
    preds = np.round(preds, 3)
    preds = np.clip(preds, 0, None)
    
    submission = pd.DataFrame({
        'ID': ids,
        Config.TARGET: preds
    })
    
    submission_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    
    submission.to_csv(
        submission_path,
        index=False,
        encoding="utf-8",
        float_format="%.3f",
        lineterminator="\n"
    )
    
    reload_check = pd.read_csv(submission_path)
    assert list(reload_check.columns) == ['ID', Config.TARGET], \
        f"[BUILD_SUBMISSION] Column mismatch! Expected ['ID', '{Config.TARGET}'], got {list(reload_check.columns)}"
    assert reload_check[Config.TARGET].dtype == np.float64, \
        f"[BUILD_SUBMISSION] Reload dtype mismatch! Expected float64, got {reload_check[Config.TARGET].dtype}"
    assert len(reload_check) == len(submission), \
        f"[BUILD_SUBMISSION] Row count mismatch! Expected {len(submission)}, got {len(reload_check)}"
    
    with open(submission_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    return submission, file_hash

def inspect_columns(df, tag, logger=None):
    """Upgraded Forensic Inspection (v6.0 - Causality Reconstruction)."""
    if Config.TRACE_LEVEL == 'OFF': return
    
    if not isinstance(df, pd.DataFrame):
        if logger: logger.warning(f"[COL_CHECK] {tag} | SKIPPED (Not a DataFrame: {type(df)})")
        return

    cols = df.columns
    types = set(type(c) for c in cols)
    bad = [c for c in cols if not isinstance(c, str)]
    
    msg = f"[COL_CHECK] {tag} | shape={df.shape} | types={types}"
    if logger: logger.info(msg)
    else: print(msg)
    
    if bad:
        err_msg = f"!!! [COL_CORRUPTION] {tag} Found {len(bad)} non-string columns !!!"
        if logger:
            logger.error(err_msg)
            logger.error(f"non_string_columns_sample={bad[:10]}")
            logger.error(f"column_types_detail={[(c, type(c)) for c in bad[:10]]}")
            
            # Lineage Lookup
            if 'lineage' in df.attrs:
                logger.error("[LINEAGE_TRACE] Checking lineage for corrupted columns...")
                for c in bad[:5]:
                    lin = df.attrs['lineage'].get(c, "UNKNOWN (Structural Corruption)")
                    logger.error(f"  Column {c} | Lineage: {lin}")
        else:
            print(err_msg)
            print(f"Sample: {bad[:10]}")

def track_lineage(df, new_cols, source, function, operation):
    """Register feature lineage in DataFrame metadata."""
    if 'lineage' not in df.attrs:
        df.attrs['lineage'] = {}
    
    for col in new_cols:
        df.attrs['lineage'][col] = {
            "source": source,
            "function": function,
            "operation": operation
        }

def ensure_dataframe(X, feature_names=None, tag="UNKNOWN"):
    """Enforces DataFrame schema with string column names.
    
    INVARIANT: The returned object is ALWAYS a pd.DataFrame
    with ALL column names as Python str.
    """
    if isinstance(X, np.ndarray):
        assert feature_names is not None, f"[FATAL] feature_names required for numpy→DataFrame at {tag}"
        assert len(feature_names) == X.shape[1], (
            f"[FATAL] shape mismatch at {tag}: "
            f"feature_names={len(feature_names)}, array.shape[1]={X.shape[1]}"
        )
        X = pd.DataFrame(X, columns=feature_names)
    elif isinstance(X, pd.DataFrame):
        # MANDATORY CHECK: All columns must be str. NO silent coercion allowed.
        bad_cols = [c for c in X.columns if not isinstance(c, str)]
        if bad_cols:
            raise TypeError(
                f"[FATAL] Non-string columns detected at {tag}! "
                f"Sample: {bad_cols[:5]}. types: {[(c, type(c)) for c in bad_cols[:5]]}"
            )
    elif isinstance(X, dict):
        # Convert dict to DataFrame (usually used in Data Loader FE)
        X = pd.DataFrame(X)
        # Validate result
        bad_cols = [c for c in X.columns if not isinstance(c, str)]
        if bad_cols:
            raise TypeError(f"[FATAL] Dict-to-DataFrame created non-string columns at {tag}")
    else:
        raise TypeError(f"[FATAL] Unsupported type at {tag}: {type(X)}")
    
    # POST-CONDITION: all columns are strictly str
    assert all(isinstance(c, str) for c in X.columns), (
        f"[FATAL] ensure_dataframe post-condition violated at {tag}"
    )
    
    logger = logging.getLogger(__name__)
    if Config.TRACE_LEVEL != 'OFF':
        logger.info(f"[SCHEMA_OK] {tag} | shape={X.shape} | cols_sample={list(X.columns[:5])}")
    
    return X

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

class GlobalStatStore:
    """Single Source of Truth (SSOT) for feature statistics (v13.0).
    Stores mean, std, quantiles, and max to enforce semantic invariance.
    """
    @staticmethod
    def compute_and_save(df, feature_cols, path):
        stats = {}
        logger = logging.getLogger("GlobalStatStore")
        logger.info(f"[GlobalStatStore] Computing stats for {len(feature_cols)} features...")
        
        for col in feature_cols:
            if col not in df.columns: continue
            series = df[col].dropna()
            if len(series) == 0: continue
            
            stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "max": float(series.max()),
                "q25": float(series.quantile(0.25)),
                "q50": float(series.quantile(0.50)),
                "q75": float(series.quantile(0.75)),
                "q90": float(series.quantile(0.90)),
            }
            
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"✓ Global stats saved to {path}")

    @staticmethod
    def load(path):
        logger = logging.getLogger("GlobalStatStore")
        if not os.path.exists(path):
            logger.warning(f"[GlobalStatStore] Stats file not found: {path}")
            return None
        with open(path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        logger.info(f"✓ Global stats loaded from {path} (Features: {len(stats)})")
        return stats

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

def get_logger(name=None, level=None):
    """Standardized Forensic Logger (v10.0).
    Args:
        name: Logger name (usually __name__ or phase name)
        level: Logging level (str or int). If None, defaults to INFO or Config.LOG_LEVEL.
    """
    # 1. Resolve Name
    logger_name = name if name else "root"
    
    # 2. Configure Global BasicConfig (Only once)
    # We use a custom level mapping for flexibility
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
        'IMPORTANT': logging.INFO # Custom mapping
    }
    
    log_level = level if level else getattr(Config, 'LOG_LEVEL', 'INFO')
    if isinstance(log_level, str):
        numeric_level = level_map.get(log_level.upper(), logging.INFO)
    else:
        numeric_level = log_level if log_level else logging.INFO

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True # Ensure our config takes precedence
    )
    
    return logging.getLogger(logger_name)

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

GLOBAL_PEAK_MEMORY = 0.0

def memory_guard(label, logger, threshold=0.8):
    """Memory Guard System (v10.0 - Selective Recovery).
    Checks system RAM usage. If > threshold (default 80%), triggers aggressive GC.
    Non-critical: failure in guard shouldn't crash pipeline.
    """
    try:
        total_ram = get_system_ram()
        proc_mem = get_process_memory()
        
        # Update peak
        global GLOBAL_PEAK_MEMORY
        if proc_mem > GLOBAL_PEAK_MEMORY:
            GLOBAL_PEAK_MEMORY = proc_mem
            
        SAFE_LIMIT_MB = total_ram * threshold
        
        if proc_mem > SAFE_LIMIT_MB:
            logger.warning(f"!!! [MEMORY_GUARD] High RAM Detection ({proc_mem:.1f} MB > {SAFE_LIMIT_MB:.1f} MB) !!!")
            logger.info("[MEMORY_GUARD] Triggering aggressive Garbage Collection...")
            gc.collect()
            new_mem = get_process_memory()
            logger.info(f"[MEMORY_GUARD] GC Complete. Memory: {proc_mem:.1f} -> {new_mem:.1f} MB")
            return False
        
        logger.info(f"[MEMORY_GUARD] {label} | System Healthy | Process: {proc_mem:.1f} MB (Peak: {GLOBAL_PEAK_MEMORY:.1f} MB)")
        return True
    except Exception as e:
        if logger:
            logger.warning(f"[MEMORY_GUARD_ERR] Non-critical guard failure: {str(e)}")
        return True # Continue execution

def log_memory_usage(tag: str, logger=None):
    assert isinstance(tag, str)
    try:
        import psutil, os as _os
        process = psutil.Process(_os.getpid())
        mem_mb = process.memory_info().rss / 1024**2
    except Exception:
        mem_mb = get_process_memory()
    msg = f"[MEMORY] {tag} | {mem_mb:.2f} MB"
    if logger:
        logger.info(msg)
    else:
        print(msg)

def log_forensic_snapshot(data, label, logger, prev_cols=None):
    """Forensic Data Snapshot (v10.0 - Selective Recovery)."""
    try:
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
                
        log_memory_usage(f"SNAPSHOT {label}", logger)
        logger.info(f"--- [/DATA_SNAPSHOT] ---\n")
        return len(cols) if cols else 0
    except Exception as e:
        if logger:
            logger.warning(f"[SNAPSHOT_ERR] Non-critical snapshot failure for {label}: {str(e)}")
        return 0

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
    """Save with standard I/O trace (CRITICAL)."""
    logger = get_logger("FILE_IO")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info(f"[FILE_IO_TRACE] SAVE | Path: {path} | Type: {type(data)}")
    if hasattr(data, "to_pickle"):
        data.to_pickle(path)
    else:
        with open(path, "wb") as f:
            pickle.dump(data, f)
    logger.info(f"✓ SAVE Success: {path}")

def load_pkl(path):
    """Load with standard I/O trace (CRITICAL)."""
    logger = get_logger("FILE_IO")
    logger.info(f"[FILE_IO_TRACE] LOAD | Path: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle not found: {path}")
    try:
        # Try pandas first for legacy compatibility
        data = pd.read_pickle(path)
    except Exception:
        # Fallback to standard pickle for custom objects
        with open(path, "rb") as f:
            data = pickle.load(f)
    
    shape_info = f" | Shape: {data.shape}" if hasattr(data, "shape") else ""
    logger.info(f"✓ LOAD Success: {path}{shape_info}")
    gc.collect()
    return data

def save_npy(data, path):
    """Numpy save with traceability (CRITICAL)."""
    logger = get_logger("FILE_IO")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info(f"[FILE_IO_TRACE] SAVE_NPY | Path: {path} | Shape: {data.shape}")
    np.save(path, data)
    logger.info(f"✓ SAVE_NPY Success: {path}")

def load_npy(path, allow_pickle=False):
    """Numpy load with traceability (CRITICAL)."""
    logger = get_logger("FILE_IO")
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
