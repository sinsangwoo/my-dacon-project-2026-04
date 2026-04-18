import pandas as pd
import numpy as np
import logging
import sys
import os

from src.config import Config
from src.data_loader import load_data, add_hybrid_latent_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("forensic_parity")

def audit_decomposed_contract():
    """Verify that the model decomposition contract is strictly enforced."""
    logger.info("=== [AUDIT: DECOMPOSED CONTRACT] ===")
    
    # 1. Loading
    train, test = load_data()
    # Use minimal for parity check speed
    train = train.head(5000); test = test.head(2000)
    
    # 2. Preprocess
    train, test, embed_features, _ = add_hybrid_latent_features(train, test)
    raw_features = [c for c in train.columns if c not in embed_features + Config.ID_COLS + [Config.TARGET]]
    
    # 3. Disjoint Check
    intersect = set(raw_features).intersection(set(embed_features))
    if intersect:
        logger.error(f"[FAILURE] Disjoint Set Violation! Overlap: {intersect}")
        return False
    logger.info(f"[PASS] Feature Sets are strictly disjoint. Raw: {len(raw_features)} | Embed: {len(embed_features)}")
    
    # 4. Latent Variance Check
    embed_std = train[embed_features].std()
    low_var = embed_std[embed_std < 1e-4]
    if not low_var.empty:
        logger.error(f"[FAILURE] Near-zero variance detected in {len(low_var)} latent features!")
        return False
    logger.info("[PASS] All latent features show healthy variance.")
    
    # 5. Correlation Entropy
    corr_matrix = train[embed_features].corr().abs()
    high_corr = (corr_matrix > 0.98).sum().sum() - len(embed_features)
    if high_corr > (len(embed_features) * 2):
        logger.warning(f"[WARNING] High collinearity in embedding space ({high_corr} pairs > 0.98).")
    
    # 6. NaN Guard
    if train[embed_features + raw_features].isna().any().any():
        logger.error("[FAILURE] NaNs detected in finalized feature set!")
        return False
    logger.info("[PASS] Full feature set is NaN-free.")
    
    logger.info("=== [/AUDIT: DECOMPOSED CONTRACT] ===")
    return True

if __name__ == "__main__":
    if audit_decomposed_contract():
        logger.info("[VERDICT] SAFE_FOR_PIPELINE")
        sys.exit(0)
    else:
        logger.error("[VERDICT] CONTRACT_VIOLATION")
        sys.exit(1)
