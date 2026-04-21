import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.model_selection import train_test_split
import sys

# Add current directory to path so we can import src
sys.path.append(os.getcwd())

from src.config import Config
from src.utils import load_npy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_sample():
    logger.info("[SAMPLING] Preparing 40% Stratified Audit Sample...")
    
    # Path to y_train (Phase 4 artifact in default_run)
    y_path = f'./outputs/default_run/processed/y_train.npy'
    if not os.path.exists(y_path):
        raise RuntimeError("Phase 4 y_train.npy missing! Run Phase 1-4 first.")
        
    y = load_npy(y_path)
    indices = np.arange(len(y))
    
    # Quantize y for stratification (ensure we get extreme delays)
    # Using 10 bins
    y_bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
    
    # Stratified 40% sample
    _, sample_idx = train_test_split(
        indices,
        test_size=0.4,
        random_state=42, # RULE 1: Fix Random Seed
        stratify=y_bins
    )
    
    # Sort for deterministic index mapping in Trainer
    sample_idx = np.sort(sample_idx)
    
    sample_path = 'audit_sample.pkl'
    with open(sample_path, 'wb') as f:
        pickle.dump(sample_idx, f)
        
    logger.info(f"[SAMPLING] Generated sample with {len(sample_idx)} rows (40% of {len(y)}).")
    logger.info(f"[SAMPLING] Saved to {sample_path}")

if __name__ == "__main__":
    prepare_sample()
