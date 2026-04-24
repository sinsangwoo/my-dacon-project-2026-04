import sys
import os
import time
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import Config
from src.data_loader import load_data, build_base_features, SuperchargedPCAReconstructor, apply_latent_features
from src.schema import FEATURE_SCHEMA

def test_single_pass():
    print("Loading data...")
    train, test = load_data()
    train = train.iloc[:1000].copy() # small subset for testing
    
    # add base features
    train = build_base_features(train)
    
    # create train/val split
    tr_df = train.iloc[:800].copy()
    val_df = train.iloc[800:].copy()
    
    print("Fitting reconstructor...")
    reconstructor = SuperchargedPCAReconstructor(input_dim=len(FEATURE_SCHEMA['raw_features']))
    reconstructor.fit(tr_df[Config.EMBED_BASE_COLS].values)
    
    print("Running original apply_latent_features on tr_df...")
    t0 = time.time()
    tr_df_full_orig = apply_latent_features(tr_df, reconstructor, pool_df=tr_df)
    t1 = time.time()
    print(f"Original time (tr_df): {t1 - t0:.4f}s")
    
    # Now simulate the cached version
    class CachedReconstructor(SuperchargedPCAReconstructor):
        def __init__(self, orig):
            self.__dict__ = orig.__dict__.copy()
            self.pool_embed = None
            self.pool_norm = None
            
        def build_fold_cache(self, df_train_pool):
            base_cols = Config.EMBED_BASE_COLS
            self.pool_embed = self.get_embeddings(df_train_pool[base_cols].values)
            self.pool_norm = self.pool_embed / (np.linalg.norm(self.pool_embed, axis=1, keepdims=True) + 1e-8)
            
        def calculate_graph_stats(self, df_target, df_train_pool=None):
            base_cols = Config.EMBED_BASE_COLS
            target_embed = self.get_embeddings(df_target[base_cols].values)
            
            if self.pool_embed is not None and self.pool_norm is not None:
                pool_embed = self.pool_embed
                pool_norm = self.pool_norm
            else:
                pool_embed = self.get_embeddings(df_train_pool[base_cols].values)
                pool_norm = pool_embed / (np.linalg.norm(pool_embed, axis=1, keepdims=True) + 1e-8)
                
            results = {}
            for k in Config.MULTI_K:
                dist_matrix = 1 - np.dot(target_embed / (np.linalg.norm(target_embed, axis=1, keepdims=True) + 1e-8), pool_norm.T)
                nn_indices = np.argsort(dist_matrix, axis=1)[:, :k]
                
                results[f'embed_mean_{k}'] = np.array([pool_embed[idx].mean(axis=0) for idx in nn_indices])
                results[f'embed_std_{k}'] = np.array([pool_embed[idx].std(axis=0) for idx in nn_indices])
                
                sim_scores = 1.0 - dist_matrix[np.arange(len(dist_matrix))[:, None], nn_indices]
                weights = np.exp(sim_scores) / np.sum(np.exp(sim_scores), axis=1, keepdims=True)
                results[f'weighted_mean_{k}'] = np.sum(weights[:, :, None] * pool_embed[nn_indices], axis=1)
                
                if k == 20:
                    results[f'trend_proxy_{k}'] = results[f'embed_mean_{k}'] - target_embed
                    results[f'volatility_proxy_{k}'] = results[f'embed_std_{k}']

            results['regime_proxy'] = self.kmeans.predict(target_embed.astype(np.float64))
            avg_dist_20 = np.sort(dist_matrix, axis=1)[:, :20].mean(axis=1)
            results['local_density'] = 1.0 / (avg_dist_20 + 1e-6)
            results['similarity_entropy'] = -np.sum(weights * np.log(weights + 1e-8), axis=1)
            
            return results

    def apply_latent_features_cached(df, reconstructor, pool_df=None, scaler=None):
        df = df.copy()
        
        all_schema_features = FEATURE_SCHEMA['all_features']
        existing_cols = set(df.columns)
        new_cols_to_add = [c for c in all_schema_features if c not in existing_cols]
        if new_cols_to_add:
            df = pd.concat([df, pd.DataFrame(0.0, index=df.index, columns=new_cols_to_add, dtype='float32')], axis=1)
            
        if scaler is not None:
            df = scaler.transform(df, FEATURE_SCHEMA['raw_features'])
            
        reference_pool = pool_df if pool_df is not None else df
        
        cache_managed_locally = False
        if reconstructor.pool_embed is None:
            reconstructor.build_fold_cache(reference_pool)
            cache_managed_locally = True
            
        for lid in df['layout_id'].unique():
            mask = df['layout_id'] == lid
            latent_stats = reconstructor.calculate_graph_stats(df[mask], reference_pool)
            
            for feat_name, values in latent_stats.items():
                if isinstance(values, np.ndarray) and values.ndim > 1:
                    for d in range(values.shape[1]):
                        col_name = f"{feat_name}_d{d}"
                        if col_name in df.columns:
                            df.loc[mask, col_name] = values[:, d].astype('float32')
                elif feat_name in df.columns:
                    df.loc[mask, feat_name] = values.astype('float32')
                    
        if cache_managed_locally:
            reconstructor.pool_embed = None
            reconstructor.pool_norm = None
            
        return df

    print("Running CACHED apply_latent_features on tr_df...")
    cached_recon = CachedReconstructor(reconstructor)
    t0 = time.time()
    tr_df_full_new = apply_latent_features_cached(tr_df, cached_recon, pool_df=tr_df)
    t1 = time.time()
    print(f"Cached time (tr_df): {t1 - t0:.4f}s")
    
    # Compare outputs
    cols = tr_df_full_orig.select_dtypes(include=[np.number]).columns
    diffs = []
    for c in cols:
        diff = np.abs(tr_df_full_orig[c] - tr_df_full_new[c]).max()
        if diff > 1e-5:
            diffs.append((c, diff))
    
    if diffs:
        print("DIFFERENCES FOUND!")
        for c, d in diffs[:10]:
            print(f"{c}: max diff = {d}")
    else:
        print("ALL OUTPUTS IDENTICAL!")

if __name__ == '__main__':
    test_single_pass()
