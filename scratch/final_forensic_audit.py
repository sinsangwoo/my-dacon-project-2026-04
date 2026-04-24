import numpy as np
import pandas as pd
import logging
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.utils import DriftShieldScaler, run_integrity_audit
from src.data_loader import SuperchargedPCAReconstructor, FEATURE_SCHEMA, add_time_series_features, add_extreme_detection_features
from src.config import Config
from src.schema import BASE_COLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FORENSIC_AUDIT")

def audit_driftshield():
    logger.info("\n=== PART 1: DriftShield FULL FORENSIC ANALYSIS ===")
    scaler = DriftShieldScaler()
    
    n_samples = 1000
    X = np.random.normal(10, 2, (n_samples, 2)).astype(np.float32)
    # Add mild outliers
    X[0, 0] = 30.0 
    X[1, 1] = -10.0
    
    cols = ['order_inflow_15m', 'robot_utilization']
    df = pd.DataFrame(X, columns=cols)
    
    scaler.fit(df, cols)
    
    std_before_0 = np.std(X[:, 0])
    
    df_drifted = scaler.transform(df, cols)
    
    logger.info(f"Clipping Ratios: {scaler.clipping_ratios}")
    
    std_after_0 = np.std(df_drifted[cols[0]])
    ratio = std_after_0 / std_before_0
    
    logger.info(f"feat0 ratio: {ratio:.4f}")
    
    if ratio < 0.6:
        logger.error("!!! FLAG: VARIANCE COMPRESSION DETECTED !!!")
        return "CRITICAL FAILURE"
    
    if cols[0] not in scaler.clipping_ratios:
        logger.error("!!! CLIPPING MONITOR FAILURE: Ratio not tracked !!!")
        return "CRITICAL FAILURE"
        
    return "SAFE"

def audit_integrity():
    logger.info("\n=== PART 2: INTEGRITY & DUPLICATE DETECTION ===")
    n_samples = 100
    X = np.random.rand(n_samples, 5)
    df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(5)])
    
    # Add exact duplicate
    df.loc[10] = df.loc[0]
    # Add near-duplicate (6 decimal)
    df.loc[11] = df.loc[1] + 1e-8
    
    report = run_integrity_audit(df, label="AUDIT_TEST")
    
    if report['duplicates'] != 1:
        logger.error(f"!!! DUPLICATE DETECTION FAILURE: Expected 1, got {report['duplicates']} !!!")
        return "CRITICAL FAILURE"
        
    if report['near_duplicates'] < 2: # 1 exact + 1 near
        logger.error(f"!!! NEAR-DUPLICATE DETECTION FAILURE: Expected >=2, got {report['near_duplicates']} !!!")
        return "CRITICAL FAILURE"
        
    return "SAFE"

def audit_extreme_intelligence():
    logger.info("\n=== PART 3: EXTREME INTELLIGENCE (MULTI) ===")
    n_samples = 1000
    cols = BASE_COLS
    X = np.random.normal(0, 1, (n_samples, len(cols))).astype(np.float32)
    df = pd.DataFrame(X, columns=cols)
    df['scenario_id'] = 0
    df['ID'] = [f"ID_{i}" for i in range(n_samples)]
    
    # Force extreme in one col
    df.loc[:10, 'order_inflow_15m'] = 100.0
    # Force extreme in another col (different rows)
    df.loc[20:30, 'robot_utilization'] = 100.0
    
    df_ts = add_time_series_features(df)
    df_ext = add_extreme_detection_features(df_ts)
    
    if 'is_extreme_multi' not in df_ext.columns:
        logger.error("!!! FEATURE MISSING: is_extreme_multi !!!")
        return "CRITICAL FAILURE"
        
    coverage = df_ext['is_extreme_multi'].mean()
    logger.info(f"is_extreme_multi coverage: {coverage:.2%}")
    
    # Should cover at least the manual extreme rows
    if df_ext.loc[:10, 'is_extreme_multi'].sum() < 10:
        logger.error("!!! is_extreme_multi failed to capture order_inflow extremes !!!")
        return "CRITICAL FAILURE"
        
    return "SAFE"

if __name__ == "__main__":
    verdict = "SAFE"
    
    if audit_driftshield() == "CRITICAL FAILURE": verdict = "CRITICAL FAILURE"
    if audit_integrity() == "CRITICAL FAILURE": verdict = "CRITICAL FAILURE"
    if audit_extreme_intelligence() == "CRITICAL FAILURE": verdict = "CRITICAL FAILURE"
    
    logger.info(f"\nFINAL EDGE BOOST VERDICT: {verdict}")
    if verdict == "CRITICAL FAILURE":
        sys.exit(1)
    sys.exit(0)
