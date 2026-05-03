import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import pickle
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from src.data_loader import apply_latent_features
from src.config import Config

RUN_ID = "run_20260429_115949"
PROCESSED_PATH = f"./outputs/{RUN_ID}/processed"
MODELS_PATH = f"./outputs/{RUN_ID}/models/reconstructors"

def get_auc(X_tr, X_te, name):
    X = np.vstack([X_tr, X_te])
    y = np.hstack([np.zeros(len(X_tr)), np.ones(len(X_te))])
    clf = LGBMClassifier(n_estimators=100, max_depth=3, random_state=42, verbose=-1)
    clf.fit(X, y)
    auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    print(f"[{name}] ADV AUC: {auc:.4f}")
    return auc

def run_latent_forensic():
    # Load fold 0 reconstructor
    with open(f"{MODELS_PATH}/recon_fold_0.pkl", "rb") as f: recon = pickle.load(f)
    with open(f"{MODELS_PATH}/features_fold_0.pkl", "rb") as f: fold_features = pickle.load(f)
    
    train_base = pd.read_pickle(f"{PROCESSED_PATH}/train_base.pkl")
    test_base = pd.read_pickle(f"{PROCESSED_PATH}/test_base.pkl")
    
    # We need to simulate the trainer's fold split to get exactly what it used
    # But for a general check, let's just use the whole sample
    s_tr = train_base.sample(min(5000, len(train_base)), random_state=42)
    s_te = test_base.sample(min(5000, len(test_base)), random_state=42)
    
    # Apply latent features
    # Note: apply_latent_features expects already scaled data if it uses the pool.
    # But for a quick AUC check on embeddings only, we can just look at the embeddings.
    
    # Let's get embeddings directly from the recon
    base_cols = Config.EMBED_BASE_COLS
    
    # Impute and get embeddings
    X_tr_embed_base = recon.imputer.transform(s_tr[base_cols].fillna(-999).values)
    X_te_embed_base = recon.imputer.transform(s_te[base_cols].fillna(-999).values)
    
    e_tr = recon.get_embeddings(X_tr_embed_base, already_scaled=True)
    e_te = recon.get_embeddings(X_te_embed_base, already_scaled=True)
    
    get_auc(e_tr, e_te, "H4: PCA EMBEDDINGS ONLY")
    
    # Test regime proxy
    r_tr = recon.kmeans.predict(e_tr.astype(np.float64)).reshape(-1, 1)
    r_te = recon.kmeans.predict(e_te.astype(np.float64)).reshape(-1, 1)
    get_auc(r_tr, r_te, "H4: CLUSTERS ONLY")

if __name__ == "__main__":
    run_latent_forensic()
