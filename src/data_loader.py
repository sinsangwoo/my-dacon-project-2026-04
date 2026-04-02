import pandas as pd
from .config import Config

def load_data():
    """Load train and test datasets."""
    train = pd.read_csv(f'{Config.DATA_PATH}train.csv')
    test = pd.read_csv(f'{Config.DATA_PATH}test.csv')
    return train, test

def get_features(train, test):
    """Filter columns to get features."""
    feature_cols = [c for c in train.columns if c not in Config.ID_COLS + [Config.TARGET]]
    return feature_cols
