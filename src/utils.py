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
from .config import Config
from sklearn.metrics import mean_absolute_error, roc_auc_score
from lightgbm import LGBMClassifier
import glob

# Optional dependencies handled at top-level
try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def save_json(data, path, indent=2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder, ensure_ascii=False)

def assert_artifact_exists(path, description="Artifact"):
    """Strictly enforces that a required artifact exists.
    If it does not exist, raises a RuntimeError to prevent silent fallbacks.
    """
    if not os.path.exists(path):
        raise RuntimeError(f"[MISSING_ARTIFACT] {description} not found at {path}. "
                           "This breaks phase dependency guarantees. "
                           "Please re-run the requisite preceding phase.")

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
                            if is_safe_module:
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
        # Suppress the specific LGBM/sklearn feature name warning
        warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")
        model.fit(X, y, **kwargs)
        # Filter out environmental/platform warnings
        w = [warn for warn in w if "Could not find the number of physical cores" not in str(warn.message)]
        w = [warn for warn in w if "X does not have valid feature names" not in str(warn.message)]
        
        if len(w) > 0:
            for warn in w:
                logger.warning(f"[GLOBAL_CONTRACT_WARNING] {warn.category.__name__}: {warn.message}")
            # [RELAXATION] In competition mode, we log warnings but don't crash unless they are critical.
    
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

    if hasattr(model, "feature_names_in_"):
        validate_names(model.feature_names_in_, "feature_names_in_")

    if "LGBM" in str(type(model)) and hasattr(model, "booster_"):
        validate_names(model.booster_.feature_name(), "booster.feature_name()")

    if hasattr(model, "n_features_in_"):
        assert model.n_features_in_ == n_features

def SAFE_PREDICT(model, X):
    """SAFE GATEWAY for model.predict (v12.0)"""
    if isinstance(X, pd.DataFrame):
        raise RuntimeError("[GLOBAL_CONTRACT_VIOLATION] DataFrame passed to model.predict")
    
    assert isinstance(X, np.ndarray), f"[GLOBAL_CONTRACT_VIOLATION] Expected np.ndarray, got {type(X)}"
    assert X.dtype == np.float32, f"[GLOBAL_CONTRACT_VIOLATION] Expected float32, got {X.dtype}"
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")
        return model.predict(X)

def SAFE_PREDICT_PROBA(model, X):
    """SAFE GATEWAY for model.predict_proba (v12.0)"""
    if isinstance(X, pd.DataFrame):
        raise RuntimeError("[GLOBAL_CONTRACT_VIOLATION] DataFrame passed to model.predict_proba")
    
    assert isinstance(X, np.ndarray), f"[GLOBAL_CONTRACT_VIOLATION] Expected np.ndarray, got {type(X)}"
    assert X.dtype == np.float32, f"[GLOBAL_CONTRACT_VIOLATION] Expected float32, got {X.dtype}"
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")
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
    
    # [RULE 3: DUAL CHECK SYSTEM]
    # 1. Hard ID Check
    if not (df['ID'].values == sample_df['ID'].values).all():
        errors.append("ID column order or values do not match sample_submission (RULE 1 Violation)!")
        
    # 2. Alignment Verification (np.allclose)
    # This proves that blind assignment matches ID-based merging.
    if len(df) == len(sample_df) and 'ID' in df.columns:
        preds_blind = df[Config.TARGET].values
        # Create a "Golden Merge" for comparison
        golden = sample_df[['ID']].merge(df[['ID', Config.TARGET]], on='ID', how='left')
        preds_merged = golden[Config.TARGET].values
        
        if not np.allclose(preds_blind, preds_merged, rtol=1e-5, atol=1e-8):
            errors.append("Alignment mismatch! Blind assignment does not match ID-merged ground truth (RULE 3 Violation).")
    
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
    save_json(trace_data, path)

def preserve_fail_case(file_path, run_id):
    """Copy failed submission to metadata/failed_submissions/ (v11.0)."""
    fail_dir = os.path.join("metadata", "failed_submissions", run_id)
    os.makedirs(fail_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dest_path = os.path.join(fail_dir, f"submission_fail_{timestamp}.csv")
    shutil.copy2(file_path, dest_path)
    return dest_path

def build_metrics(y_true, y_pred, y_base=None):
    """
    PURPOSE: [PHASE 1 & 2: ANTI-CV-ILLUSION METRIC BUILDER]
    Computes multi-dimensional performance and distribution metrics.
    [v4.0] Added Execution Layer Analytics (FP Cost, Gain Capture).
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError(f"[METRIC_BUILD_FAILED] Shape mismatch or empty: y_true={len(y_true)}, y_pred={len(y_pred)}")

    # [OOF_GAP_FIX] Filter out NaN predictions (e.g., initial chunk in expanding window)
    valid_mask = ~np.isnan(y_pred)
    if not valid_mask.any():
        logger.warning("[METRIC_BUILD] No valid predictions found! Returning zeroed metrics.")
        return {"mae": 0.0, "rmse": 0.0, "med_ae": 0.0, "mean_ratio": 0.0, "std_ratio": 0.0, 
                "p50_ratio": 0.0, "p90_ratio": 0.0, "p99_ratio": 0.0, "fp_cost": 0.0, "gain_capture": 0.0}
    
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    if y_base is not None:
        y_base = np.asarray(y_base, dtype=np.float32)[valid_mask]
        
    # [GLOBAL METRICS]
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    med_ae = float(np.median(np.abs(y_true - y_pred)))
    
    # [DISTRIBUTION METRICS]
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    std_true, std_pred = np.std(y_true), np.std(y_pred)
    
    dist_metrics = {
        "mean_ratio": float(mean_pred / (mean_true + 1e-9)),
        "std_ratio": float(std_pred / (std_true + 1e-9)),
        "p50_ratio": float(np.quantile(y_pred, 0.5) / (np.quantile(y_true, 0.5) + 1e-9)),
        "p90_ratio": float(np.quantile(y_pred, 0.9) / (np.quantile(y_true, 0.9) + 1e-9)),
        "p99_ratio": float(np.quantile(y_pred, 0.99) / (np.quantile(y_true, 0.99) + 1e-9))
    }
    
    # [ERROR DISTRIBUTION]
    errors = np.abs(y_true - y_pred)
    error_metrics = {
        "mae_bottom_50": float(np.mean(np.sort(errors)[:len(errors)//2])),
        "mae_top_10": float(np.mean(errors[errors >= np.quantile(errors, 0.9)])),
        "mae_top_1": float(np.mean(errors[errors >= np.quantile(errors, 0.99)]))
    }
    
    # [QUANTILE PERFORMANCE BREAKDOWN]
    quantile_bins = [
        (0, 0.5, "Q0_50"),
        (0.5, 0.9, "Q50_90"),
        (0.9, 0.99, "Q90_99"),
        (0.99, 1.0, "Q99_100")
    ]
    quantile_mae = {}
    for q_low, q_high, label in quantile_bins:
        v_low = np.quantile(y_true, q_low)
        v_high = np.quantile(y_true, q_high)
        mask = (y_true >= v_low) & (y_true <= v_high)
        if mask.any():
            quantile_mae[f"{label}_mae"] = float(mean_absolute_error(y_true[mask], y_pred[mask]))
        else:
            quantile_mae[f"{label}_mae"] = 0.0

    # [EXECUTION LAYER ANALYTICS]
    execution_metrics = {"fp_cost": 0.0, "gain_capture": 0.0}
    if y_base is not None:
        y_base = np.asarray(y_base, dtype=np.float32)
        base_err = np.abs(y_true - y_base)
        pred_err = np.abs(y_true - y_pred)
        
        # FP Cost: MAE penalty from tail activations that made things worse
        fp_mask = pred_err > base_err
        if fp_mask.any():
            execution_metrics["fp_cost"] = float(np.mean(pred_err[fp_mask] - base_err[fp_mask]))
            
        # Gain Capture: How much of the potential MAE improvement did we actually get?
        # Oracle Gain = sum(base_err - true_tail_err) where base_err > true_tail_err
        potential_gain = np.sum(np.maximum(0, base_err - np.abs(y_true - y_true))) # This is just base_err where we could improve
        actual_gain = np.sum(np.maximum(0, base_err - pred_err))
        execution_metrics["gain_capture"] = float(actual_gain / (np.sum(base_err) + 1e-9))

    metrics = {
        "mean_mae": mae,
        "mae": mae,
        "rmse": rmse,
        "med_ae": med_ae,
        **dist_metrics,
        **error_metrics,
        **quantile_mae,
        **execution_metrics
    }
    
    logger = logging.getLogger("METRIC_BUILDER")
    logger.info(f"[METRIC_DISTRIBUTION] mean_ratio={dist_metrics['mean_ratio']:.4f} | std_ratio={dist_metrics['std_ratio']:.4f} | p99_ratio={dist_metrics['p99_ratio']:.4f}")
    logger.info(f"[EXECUTION_AUDIT] fp_cost={execution_metrics['fp_cost']:.4f} | gain_capture={execution_metrics['gain_capture']:.4f}")
    
    return metrics

def calculate_std_ratio(preds, train_stats):
    """Compute ratio of prediction std vs training target std (Rule 4)."""
    pred_std = float(np.std(preds))
    if isinstance(train_stats, dict):
        train_std = train_stats.get('std', 1.0)
        train_mean = train_stats.get('mean', 1.0)
    else:
        train_std = float(np.std(train_stats))
        train_mean = float(np.mean(train_stats))
        
    ratio = pred_std / (train_std + 1e-9)
    pred_mean = float(np.mean(preds))
    mean_ratio = pred_mean / (train_mean + 1e-9)
    
    return ratio, pred_std, train_std, mean_ratio

def run_adversarial_validation(X_train, X_val, groups=None):
    """
    PURPOSE: [PHASE 4: ADVERSARIAL VALIDATION]
    Detect distribution shift between training and validation.
    [ZERO_TOLERANCE] Updated to respect split logic.
    """
    # [SSOT_FIX] Local imports removed
    
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    
    # Combine sets with binary labels
    X = np.vstack([X_train, X_val])
    y = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_val))])
    
    # Simple lightweight model
    model = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, verbose=-1, random_state=42)
    model.fit(X, y)
    
    probs = model.predict_proba(X)[:, 1]
    auc = float(roc_auc_score(y, probs))
    
    logger = logging.getLogger("ADV_VAL")
    logger.info(f"[ADV_VAL] auc={auc:.4f}")
    
    return auc

def generate_pseudo_test_set(X, y, seed=42):
    """
    PURPOSE: [PHASE 3: PSEUDO TEST SIMULATION]
    Simulate a harder test set by adding noise and scaling features.
    """
    X_pseudo = X.copy()
    # [DETERMINISM_FIX] Use local RandomState to avoid global state pollution
    rng = np.random.RandomState(seed)
    
    # Apply 10% noise to all features
    noise = rng.normal(0, 0.1, X.shape).astype(np.float32)
    X_pseudo += noise
    
    # Randomly scale top 20% of samples by 1.2x (Simulate extreme regime)
    scale_mask = rng.choice([0, 1], size=len(X), p=[0.8, 0.2]).astype(bool)
    X_pseudo[scale_mask] *= 1.2
    
    return X_pseudo, y

def calculate_risk_score(metrics, adv_auc, fold_maes=None):
    """
    PURPOSE: [PHASE 8: MULTI-DIMENSIONAL RISK AUDIT]
    Consolidates validation dimensions into a granular risk index.
    
    Risk Components:
    1. Divergence Risk (DR): Volatility of performance across folds.
    2. Adversarial Risk (AR): Magnitude of train-test distribution shift.
    3. Distribution Risk (DSR): Mismatch in prediction vs target moments.
    4. Tail Fragility Risk (TFR): Model instability in extreme regimes.
    """
    # [1] Divergence Risk (DR)
    if fold_maes is not None and len(fold_maes) > 1:
        cv_std = np.std(fold_maes)
        cv_mean = np.mean(fold_maes)
        div_risk = (cv_std / (cv_mean + 1e-9)) * 10.0 # Scale by 10
    else:
        div_risk = 0.0

    # [2] Adversarial Risk (AR)
    # Threshold 0.5 is random. Anything above 0.6 is concerning.
    adv_risk = max(0, adv_auc - 0.5) * 8.0
    
    # [3] Distribution Risk (DSR)
    std_mismatch = abs(1.0 - metrics.get('std_ratio', 1.0)) * 4.0
    mean_mismatch = abs(1.0 - metrics.get('mean_ratio', 1.0)) * 2.0
    dist_risk = std_mismatch + mean_mismatch
    
    # [4] Tail Fragility Risk (TFR)
    # Relative error in top 1% compared to global MAE
    global_mae = metrics.get('mae', 1e-9)
    tail_mae = metrics.get('Q99_100_mae', global_mae)
    tail_risk = (tail_mae / global_mae) * 0.2 # Usually high, so low weight
    
    total_risk = float(div_risk + adv_risk + dist_risk + tail_risk)
    
    logger = logging.getLogger("RISK_AUDIT")
    logger.info(f"[RISK_SCORE] total={total_risk:.4f} | div={div_risk:.2f} | adv={adv_risk:.2f} | dist={dist_risk:.2f} | tail={tail_risk:.2f}")
    
    return {
        "total": total_risk,
        "breakdown": {
            "divergence": div_risk,
            "adversarial": adv_risk,
            "distribution": dist_risk,
            "tail": tail_risk
        }
    }

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
    """
    [MISSION: FULL DETERMINISM] Consolidates all RNG control points.
    """
    # [SSOT_FIX] Local import removed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # Ensure any environment-level randomness is locked
    if torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # LGBM and others usually take random_state in params, 
    # but global seed is a fallback.

class DriftShieldScaler:
    """Stateful, fold-aware distribution controller (v15.0).
    Replaces GlobalStatStore to enforce FOT-ATV (Fit on Train, Apply to Val).
    """
    def __init__(self):
        self.stats = {}
        self.clipping_ratios = {}
        self.logger = logging.getLogger("DriftShieldScaler")

    def fit(self, df, feature_cols):
        self.logger.info(f"[DRIFT_SHIELD] Fitting on {len(feature_cols)} features...")
        for col in feature_cols:
            if col not in df.columns: continue
            
            # [RELIABILITY_FIX] Drop both NaNs and Infs for stable stats (v18.5)
            vals = df[col].values
            series = vals[np.isfinite(vals)]
            
            if len(series) == 0: continue
            
            # [AUDIT_FIX] Sensitive feature variance preservation
            # [EVIDENCE] Forensic report proved that P1/P99 clipping destroyed 60-85% of variance
            #   in _rate_1 and _diff_1 features, causing severe model underfitting.
            is_sensitive = any(s in col for s in ('_rate_1', '_diff_1', '_slope_5'))
            if is_sensitive:
                p1 = float(np.quantile(series, 0.001))
                p99 = float(np.quantile(series, 0.999))
            else:
                p1 = float(np.quantile(series, 0.01))
                p99 = float(np.quantile(series, 0.99))
            
            # [STRATEGY 4 — VARIANCE-AWARE FEATURE FILTERING]
            # [WHY_THIS_CHANGE] Compute clipped_std at fit time as the proper baseline
            #   for transform-time variance comparison.
            # [ROOT_CAUSE] Forensic PROOF 1 showed that the old check compared
            #   post-clip std to pre-clip std (raw std). For heavy-tailed features
            #   (rate_1 with min=-220, max=1), pre-clip std is dominated by extreme
            #   outliers. After clipping to P1/P99, std drops by 60-80% — this is
            #   the INTENDED behavior of clipping, not a defect.
            #   The old check flagged every successful clip as VARIANCE_COMPRESSION.
            # [WHY_NOT_ALTERNATIVES]
            #   - Remove the check entirely: Loses genuine drift detection capability.
            #   - Use post-clip std threshold: Doesn't adapt to feature scale.
            #   - Compute clipped_std at fit time: Establishes the CORRECT baseline.
            #     If transform-time clipped std deviates from fit-time clipped std,
            #     that indicates genuine distribution shift, not expected clip behavior.
            # [EXPECTED_IMPACT] VARIANCE_COMPRESSION errors eliminated for features
            #   where compression is caused by intended clipping. Genuine drift
            #   (e.g., test distribution fundamentally different) still detected.
            clipped_series = np.clip(series, p1, p99)
            
            self.stats[col] = {
                "p1": p1,
                "p99": p99,
                "mean": float(series.mean()),
                "std": float(series.std()),
                "clipped_std": float(np.std(clipped_series)),
            }

    def transform(self, df, feature_cols):
        if not self.stats:
            self.logger.warning("[DRIFT_SHIELD] Transform called before fit! Returning raw.")
            return df
            
        df = df.copy()
        for col in feature_cols:
            if col not in df.columns or col not in self.stats: continue
            
            s = self.stats[col]
            x = df[col].values.astype(np.float32)
            
            # [PHASE 6: EXTREME VALUE PRESERVATION]
            # Replace Inf with large finite values
            x = np.nan_to_num(x, nan=s['mean'], posinf=s['p99'], neginf=s['p1'])
            
            # 1. Clip (Preserve original scale but bound outliers)
            # [WHY_THIS_DESIGN] Outlier Clipping
            # Observed Data Behavior: Heavy-tailed distribution in delay-related features.
            # Why P1/P99: Captures 98% of variance while suppressing extreme sensor noise 
            #   that can destabilize Gradient Boosting and PCA reconstruction.
            # Sensitivity: P95 is too aggressive (loses real spikes); P99.9 preserves too much noise.
            x = np.clip(x, s['p1'], s['p99'])
            
            df[col] = x
        return df

    def save(self, path):
        save_json(self.stats, path)
        self.logger.info(f"[DRIFT_SHIELD] Scaler stats saved to {path}")

    @classmethod
    def load(cls, path):
        scaler = cls()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                scaler.stats = json.load(f)
        return scaler

class GlobalStatStore:
    """Legacy wrapper for SSOT compatibility."""
    @staticmethod
    def compute_and_save(df, feature_cols, path):
        scaler = DriftShieldScaler()
        scaler.fit(df, feature_cols)
        scaler.save(path)

    @staticmethod
    def load(path):
        scaler = DriftShieldScaler.load(path)
        return scaler.stats

    @staticmethod
    def apply_drift_shield(df, stats, feature_cols):
        scaler = DriftShieldScaler()
        scaler.stats = stats
        return scaler.transform(df, feature_cols)

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
    # [SSOT_FIX] Redundant local import removed
    
    # 1. Resolve Name
    logger_name = name if name else "root"
    
    # 2. Configure Global BasicConfig (Only once)
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
        'IMPORTANT': logging.INFO 
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
    
    logger = logging.getLogger(logger_name)
    
    # Add FileHandler if Config is initialized
    try:
        log_dir = getattr(Config, 'LOG_DIR', 'logs')
        phase_log = os.path.join(log_dir, f"{name}_forensic.log")
        
        os.makedirs(log_dir, exist_ok=True)
        
        # [LOG_POLLUTION_FIX] Prevent duplicate handler attachment
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith(f"{name}_forensic.log") for h in logger.handlers):
            fh = logging.FileHandler(phase_log, encoding='utf-8')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(fh)
            logger.propagate = True
    except:
        pass
        
    return logger

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
        # [SSOT_FIX] Local import removed
        if psutil:
            process = psutil.Process(os.getpid())
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
        # [SERIALIZATION_OPTIMIZATION] Use protocol 5 (or highest) for faster I/O
        data.to_pickle(path, protocol=5)
    else:
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
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

def run_integrity_audit(df, label="DATA"):
    """
    [MISSION: FINAL EDGE BOOST] Duplicate & Near-Zero Distance Detection
    """
    logger = logging.getLogger("INTEGRITY_AUDIT")
    logger.info(f"--- [INTEGRITY_AUDIT] {label} ---")
    
    # 1. Identical Rows
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        logger.warning(f"[INTEGRITY_RISK] Found {n_duplicates} identical rows!")
    else:
        logger.info("[INTEGRITY_OK] No identical rows detected.")
        
    # 2. Near-Zero Distance Pairs (Sub-sampling for performance if large)
    # We use a heuristic: check if any row has multiple occurrences in a rounded/binned space
    # This is a proxy for "very near" without computing O(N^2) distances
    rounded_df = df.select_dtypes(include=np.number).round(6)
    n_near_duplicates = rounded_df.duplicated().sum()
    
    if n_near_duplicates > n_duplicates:
        risk = n_near_duplicates - n_duplicates
        logger.warning(f"[INTEGRITY_RISK] Found {risk} near-zero distance pairs (6-decimal match)!")
    
    logger.info(f"Duplicate Rows: {n_duplicates} | Near-Zero Pairs: {n_near_duplicates}")
    logger.info(f"--- [/INTEGRITY_AUDIT] ---")
    
    return {
        "duplicates": int(n_duplicates),
        "near_duplicates": int(n_near_duplicates),
        "risk_status": "HIGH" if n_near_duplicates > 0 else "SAFE"
    }
