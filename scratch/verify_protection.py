import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from src.data_loader import get_protected_candidates

# Load drift report from current run
RUN_ID = "run_20260429_133848"
DRIFT_REPORT = f"logs/{RUN_ID}/summary/distribution/drift_audit_raw.csv"
TRAIN_BASE = f"outputs/{RUN_ID}/processed/train_base.pkl"

def verify():
    if not os.path.exists(DRIFT_REPORT):
        print("Drift report not found yet.")
        return
        
    drift_df = pd.read_csv(DRIFT_REPORT)
    train = pd.read_pickle(TRAIN_BASE)
    
    # Run protection logic
    protected = get_protected_candidates(train.columns, drift_df=drift_df)
    
    print(f"Total features: {len(train.columns)}")
    print(f"Protected count: {len(protected)}")
    
    drifty_bases = drift_df[drift_df['ks_stat'] > 0.10]['feature'].tolist()
    print(f"Drifty bases (KS > 0.10): {drifty_bases}")
    
    for db in drifty_bases:
        # Check if any derivatives are in the protected list
        derivatives = [p for p in protected if p.startswith(db) and p != db]
        print(f"Protected derivatives for '{db}': {derivatives}")
        
    # Check a stable base
    stable_bases = drift_df[drift_df['ks_stat'] < 0.02]['feature'].tolist()
    if stable_bases:
        sb = stable_bases[0]
        derivatives = [p for p in protected if p.startswith(sb) and p != sb]
        print(f"Protected derivatives for stable '{sb}': {derivatives[:5]}... (Total {len(derivatives)})")

if __name__ == "__main__":
    verify()
