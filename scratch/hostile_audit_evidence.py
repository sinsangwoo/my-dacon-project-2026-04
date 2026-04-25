
import os
import sys
import json
import numpy as np
import pandas as pd
import logging
from scipy.stats import ks_2samp

# Add src to path
sys.path.append(os.getcwd())

from src.config import Config
from src.data_loader import load_data, build_base_features
from src.schema import FEATURE_SCHEMA, BASE_COLS
from src.distribution import DomainShiftAudit, FeatureStabilityFilter

# Configure logging to capture output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hostile_audit")

def run_audit():
    results = {}
    
    # 1. Load Data (Smoke size for speed)
    logger.info("Loading data for audit...")
    train_raw, test_raw = load_data()
    train_raw = train_raw.head(5000)
    test_raw = test_raw.head(5000)
    
    # 2. Build Base Features and capture thresholds
    logger.info("Building base features...")
    train_base, manifest, registry = build_base_features(train_raw)
    
    # Task 1: Threshold Extraction
    results['thresholds'] = registry._derived_thresholds
    
    # Task 2 & 3: Pruning Analysis
    results['pruning'] = {
        'initial_features': len(FEATURE_SCHEMA['raw_features']),
        'after_base_features': len(train_base.columns),
        'nan_dropped': len(manifest.cols_to_drop_nan),
        'corr_dropped': len(manifest.cols_to_drop_corr),
        'var_dropped': len(manifest.cols_to_drop_var),
    }
    
    # Correlation Distribution Analysis
    runtime_raw_current = [c for c in train_base.columns if c not in Config.ID_COLS and c != Config.TARGET]
    sample_df = train_base[runtime_raw_current].sample(n=min(2000, len(train_base)), random_state=42).astype("float32")
    sample_filled = sample_df.fillna(sample_df.median())
    corr_matrix = sample_filled.corr().abs()
    corr_values = corr_matrix.to_numpy(copy=True)
    upper_tri = corr_values[np.triu_indices_from(corr_values, k=1)]
    upper_tri = upper_tri[~np.isnan(upper_tri)]
    
    results['corr_dist'] = {
        'mean': float(np.mean(upper_tri)),
        'median': float(np.median(upper_tri)),
        'p95': float(np.percentile(upper_tri, 95)),
        'p99': float(np.percentile(upper_tri, 99)),
    }
    
    # Task 4: Type-Aware fillna Validation
    # We'll check the source code for this, but let's see if we can find any mismatch
    from src.data_loader import COUNT_RATE_COLS, RATIO_PERCENT_COLS, SENSOR_ENV_COLS
    results['type_categories'] = {
        'count_rate': list(COUNT_RATE_COLS),
        'ratio_percent': list(RATIO_PERCENT_COLS),
        'sensor_env': list(SENSOR_ENV_COLS),
    }
    
    # Task 9: Feature Stability (KS)
    audit = DomainShiftAudit()
    numeric_cols = train_base.select_dtypes(include=[np.number]).columns
    features_to_audit = [c for c in numeric_cols if c in test_raw.columns]
    drift_df = audit.calculate_drift(train_base, test_raw.head(5000), features_to_audit)
    results['ks_dist'] = {
        'mean': float(drift_df['ks_stat'].mean()),
        'max': float(drift_df['ks_stat'].max()),
        'unstable_count': int((drift_df['ks_stat'] > 0.15).sum())
    }
    
    # Save results
    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/hostile_audit_evidence.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Audit evidence collected and saved to artifacts/hostile_audit_evidence.json")

if __name__ == "__main__":
    run_audit()
