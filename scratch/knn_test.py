import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import time

def build_knn_features(df_train, df_test, base_cols, k=20):
    start = time.time()
    
    # Scale base features
    scaler = StandardScaler()
    scaler.fit(df_train[base_cols])
    
    train_scaled = pd.DataFrame(scaler.transform(df_train[base_cols]), index=df_train.index, columns=base_cols)
    test_scaled = pd.DataFrame(scaler.transform(df_test[base_cols]), index=df_test.index, columns=base_cols)
    
    train_out = []
    test_out = []
    
    layouts = set(df_train['layout_id']) & set(df_test['layout_id'])
    
    for lid in layouts:
        # Sort indices to keep alignment mapping (safe since we only rely on the indices later)
        tr_idx = df_train[df_train['layout_id'] == lid].index
        te_idx = df_test[df_test['layout_id'] == lid].index
        
        tr_X = train_scaled.loc[tr_idx].values
        te_X = test_scaled.loc[te_idx].values
        
        if len(tr_X) < k:
            continue
            
        nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean', n_jobs=-1)
        nn.fit(tr_X)
        
        # Train (excluding self)
        dists_tr, inds_tr = nn.kneighbors(tr_X)
        dists_tr = dists_tr[:, 1:] # exclude self
        inds_tr = inds_tr[:, 1:]
        
        # Test
        dists_te, inds_te = nn.kneighbors(te_X, n_neighbors=k)
        
        for name, idxs_local, q_idx, dists in [('train', inds_tr, tr_idx, dists_tr), ('test', inds_te, te_idx, dists_te)]:
            res = {}
            res['ID'] = df_train.loc[q_idx, 'ID'].values if name == 'train' else df_test.loc[q_idx, 'ID'].values
            
            res['knn_dist_mean'] = dists.mean(axis=1)
            res['knn_dist_std'] = dists.std(axis=1)
            res['similarity_score'] = 1.0 / (res['knn_dist_mean'] + 1e-6)
            
            for col in base_cols:
                # Raw train values mapped by the absolute train index
                raw_tr_vals = df_train.loc[tr_idx, col].values
                # Get the neighbor values
                neigh_vals = raw_tr_vals[idxs_local]
                
                res[f'{col}_pseudo_mean_{k}'] = neigh_vals.mean(axis=1)
                res[f'{col}_pseudo_std_{k}'] = neigh_vals.std(axis=1)
                res[f'{col}_pseudo_min_{k}'] = neigh_vals.min(axis=1)
                res[f'{col}_pseudo_max_{k}'] = neigh_vals.max(axis=1)
            
            df_res = pd.DataFrame(res)
            if name == 'train':
                train_out.append(df_res)
            else:
                test_out.append(df_res)
                
    return pd.concat(train_out), pd.concat(test_out)

if __name__ == '__main__':
    print("Loading data...")
    train = pd.read_csv('data/train.csv', nrows=20000)
    test = pd.read_csv('data/test.csv', nrows=5000)
    
    base = ['order_inflow_15m', 'robot_idle']
    print(f"Building KNN for {len(train)} train, {len(test)} test")
    
    tr_f, te_f = build_knn_features(train, test, base, k=20)
    print("Train pseudo cols:", tr_f.columns[:5])
    print("Train shape:", tr_f.shape)
    print(tr_f.head())
    print("Train pseudo std:", tr_f['order_inflow_15m_pseudo_std_20'].mean())
    print("Test pseudo std:", te_f['order_inflow_15m_pseudo_std_20'].mean())
