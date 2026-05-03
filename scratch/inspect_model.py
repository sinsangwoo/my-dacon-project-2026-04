import os
import pickle
import numpy as np

def inspect_model_features():
    RUN_ID = "run_20260430_231842"
    model_path = f"./outputs/{RUN_ID}/models/lgbm/model_fold_0.pkl"
    
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    
    clf = bundle["clf"]
    print(f"Classifier expects {len(clf.feature_name_)} features.")
    print(f"Sample features: {clf.feature_name_[:20]}")
    
    # Save the feature list for the forensic script
    with open("scratch/fold_0_features.pkl", "wb") as f:
        pickle.dump(clf.feature_name_, f)

if __name__ == "__main__":
    inspect_model_features()
