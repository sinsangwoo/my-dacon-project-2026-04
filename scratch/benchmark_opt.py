import numpy as np
import time

def original_logic(target_norm, pool_norm, pool_embed, multi_k):
    results = {}
    target_embed = target_norm # Simplified for benchmark
    for k in multi_k:
        dist_matrix = 1 - np.dot(target_norm, pool_norm.T)
        nn_indices = np.argsort(dist_matrix, axis=1)[:, :k]
        
        results[f'embed_mean_{k}'] = np.array([pool_embed[idx].mean(axis=0) for idx in nn_indices])
        results[f'embed_std_{k}'] = np.array([pool_embed[idx].std(axis=0) for idx in nn_indices])
        
        sim_scores = 1.0 - dist_matrix[np.arange(len(dist_matrix))[:, None], nn_indices]
        weights = np.exp(sim_scores) / np.sum(np.exp(sim_scores), axis=1, keepdims=True)
        results[f'weighted_mean_{k}'] = np.sum(weights[:, :, None] * pool_embed[nn_indices], axis=1)
        
        if k == 20:
            results[f'trend_proxy_{k}'] = results[f'embed_mean_{k}'] - target_embed
            results[f'volatility_proxy_{k}'] = results[f'embed_std_{k}']

    avg_dist_20 = np.sort(dist_matrix, axis=1)[:, :20].mean(axis=1)
    results['local_density'] = 1.0 / (avg_dist_20 + 1e-6)
    results['similarity_entropy'] = -np.sum(weights * np.log(weights + 1e-8), axis=1)
    return results

def optimized_logic(target_norm, pool_norm, pool_embed, multi_k):
    results = {}
    target_embed = target_norm # Simplified for benchmark
    
    # 1. Calculate dist_matrix once
    dist_matrix = 1 - np.dot(target_norm, pool_norm.T)
    
    # 2. Find max K for partitioning
    max_k = max(multi_k)
    
    # 3. Partition to get top max_k indices
    # We use argpartition which is O(N)
    nn_indices_all = np.argpartition(dist_matrix, max_k, axis=1)[:, :max_k]
    
    # 4. For each row, sort only the top max_k indices to get correct order for sim_scores
    # Extract distances for the top max_k
    top_dists = dist_matrix[np.arange(len(dist_matrix))[:, None], nn_indices_all]
    # Sort these small distances
    sort_idx = np.argsort(top_dists, axis=1)
    # Reorder nn_indices_all based on sorted top_dists
    nn_indices_all = nn_indices_all[np.arange(len(nn_indices_all))[:, None], sort_idx]
    
    for k in multi_k:
        # Take first k
        nn_indices = nn_indices_all[:, :k]
        
        # Vectorized aggregation
        # pool_embed[nn_indices] shape: (N_target, k, embed_dim)
        neighbor_embeds = pool_embed[nn_indices]
        
        results[f'embed_mean_{k}'] = neighbor_embeds.mean(axis=1)
        results[f'embed_std_{k}'] = neighbor_embeds.std(axis=1)
        
        # Get distances for top k (already sorted)
        k_dists = dist_matrix[np.arange(len(dist_matrix))[:, None], nn_indices]
        sim_scores = 1.0 - k_dists
        
        weights = np.exp(sim_scores) / np.sum(np.exp(sim_scores), axis=1, keepdims=True)
        results[f'weighted_mean_{k}'] = np.sum(weights[:, :, None] * neighbor_embeds, axis=1)
        
        if k == 20:
            results[f'trend_proxy_{k}'] = results[f'embed_mean_{k}'] - target_embed
            results[f'volatility_proxy_{k}'] = results[f'embed_std_{k}']

    # For local_density, use already computed distances
    avg_dist_20 = dist_matrix[np.arange(len(dist_matrix))[:, None], nn_indices_all[:, :20]].mean(axis=1)
    results['local_density'] = 1.0 / (avg_dist_20 + 1e-6)
    
    # similarity_entropy uses weights from the LAST k in the loop (which is 40 in schema)
    # In original code, 'weights' is updated in each k loop. 
    # Let's check which k it uses for 'similarity_entropy'.
    # In original: it's outside the loop? No, it's inside the for k in multi_k loop but overwritten.
    # Wait, in the original code, 'similarity_entropy' is calculated OUTSIDE the loop using the 'weights' from the LAST k.
    results['similarity_entropy'] = -np.sum(weights * np.log(weights + 1e-8), axis=1)
    
    return results

# Setup benchmark
N_target = 100
N_pool = 10000
Embed_dim = 32
multi_k = [10, 20, 40]

target_norm = np.random.rand(N_target, Embed_dim).astype(np.float32)
pool_norm = np.random.rand(N_pool, Embed_dim).astype(np.float32)
pool_embed = np.random.rand(N_pool, Embed_dim).astype(np.float32)

# Warmup
_ = original_logic(target_norm, pool_norm, pool_embed, multi_k)
_ = optimized_logic(target_norm, pool_norm, pool_embed, multi_k)

# Run benchmark
start = time.time()
res_orig = original_logic(target_norm, pool_norm, pool_embed, multi_k)
end_orig = time.time()
print(f"Original time: {end_orig - start:.4f}s")

start = time.time()
res_opt = optimized_logic(target_norm, pool_norm, pool_embed, multi_k)
end_opt = time.time()
print(f"Optimized time: {end_opt - start:.4f}s")

# Check correctness
for key in res_orig:
    diff = np.abs(res_orig[key] - res_opt[key]).max()
    print(f"Key: {key}, Max Diff: {diff:.6e}")
    assert diff < 1e-4
