import os
import sys
import pickle
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import Config
from src.data_loader import SuperchargedPCAReconstructor, load_data, build_base_features

def serialization_audit():
    print("Starting Serialization Audit...")
    
    # 1. Setup mock data
    X = np.random.rand(100, 10).astype(np.float32)
    reconstructor = SuperchargedPCAReconstructor(input_dim=10)
    reconstructor.fit(X)
    
    # Build cache
    pool_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(10)])
    # Mock Config for test
    Config.EMBED_BASE_COLS = pool_df.columns.tolist()
    reconstructor.build_fold_cache(pool_df)
    
    # Capture original state
    orig_pool_embed = reconstructor.pool_embed.copy()
    orig_pool_norm = reconstructor.pool_norm.copy()
    orig_dtype = orig_pool_embed.dtype
    
    # 2. Serialize
    save_path = "scratch/audit_recon.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(reconstructor, f)
    
    # 3. Load
    with open(save_path, "rb") as f:
        loaded_reconstructor = pickle.load(f)
    
    # 4. Compare
    print(f"Original dtype: {orig_dtype}")
    print(f"Loaded dtype: {loaded_reconstructor.pool_embed.dtype}")
    
    max_diff_embed = np.abs(orig_pool_embed - loaded_reconstructor.pool_embed).max()
    mean_diff_embed = np.abs(orig_pool_embed - loaded_reconstructor.pool_embed).mean()
    
    print(f"Max Absolute Difference (pool_embed): {max_diff_embed}")
    print(f"Mean Absolute Difference (pool_embed): {mean_diff_embed}")
    
    # 5. Functional Equivalence
    test_X = np.random.rand(10, 10).astype(np.float32)
    test_df = pd.DataFrame(test_X, columns=Config.EMBED_BASE_COLS)
    
    stats_orig = reconstructor.calculate_graph_stats(test_df)
    stats_loaded = loaded_reconstructor.calculate_graph_stats(test_df)
    
    functional_equiv = True
    for key in stats_orig:
        diff = np.abs(stats_orig[key] - stats_loaded[key]).max()
        if diff > 1e-7:
            print(f"Functional Mismatch in {key}: {diff}")
            functional_equiv = False
            
    print(f"Functional Equivalence: {functional_equiv}")
    
    # Cleanup
    if os.path.exists(save_path):
        os.remove(save_path)

if __name__ == "__main__":
    serialization_audit()
