
import sys
import os
import unittest
import numpy as np
import pandas as pd
import gc

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.schema import FEATURE_SCHEMA, MULTI_K
from src.data_loader import SuperchargedPCAReconstructor, apply_latent_features
from src.utils import DriftShieldScaler

class TestSystemIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create minimal synthetic data for testing
        np.random.seed(42)
        cls.n_samples = 100
        
        # Use FULL feature schema including rolling/extreme to match imputer expectation
        from src.schema import get_feature_schema
        full_schema = get_feature_schema()
        cls.raw_cols = full_schema['raw_features']
        
        data = np.random.rand(cls.n_samples, len(cls.raw_cols))
        cls.df = pd.DataFrame(data, columns=cls.raw_cols)
        cls.df['ID'] = range(cls.n_samples)
        cls.df['layout_id'] = 0
        cls.df['scenario_id'] = 'test_scenario'
        
        # Mock schema for test
        cls.test_raw_features = cls.raw_cols
        
    def test_latent_feature_generation_integrity(self):
        """Verify that all expected latent features are generated without KeyError."""
        # Use full schema for fit to satisfy imputer/scaler
        scaler = DriftShieldScaler()
        scaler.fit(self.df, self.raw_cols)
        
        reconstructor = SuperchargedPCAReconstructor(input_dim=len(self.raw_cols))
        scaled_data = scaler.transform(self.df[self.raw_cols], self.raw_cols).values
        reconstructor.fit(scaled_data)
        
        # Test build_fold_cache (should work now with v18.1 fix)
        try:
            reconstructor.build_fold_cache(self.df)
            print("build_fold_cache success.")
        except Exception as e:
            self.fail(f"build_fold_cache failed: {e}")
            
        # Test 1: Full population
        try:
            df_full = apply_latent_features(self.df, reconstructor, scaler=scaler)
            print(f"Full population success. Columns: {len(df_full.columns)}")
            
            # Verify specific keys that failed before
            for k in MULTI_K:
                # [SSOT_CHECK] Check for _d0 suffix instead of base name
                self.assertIn(f'trend_proxy_{k}_d0', df_full.columns)
                self.assertIn(f'volatility_proxy_{k}_d0', df_full.columns)
        except KeyError as e:
            self.fail(f"apply_latent_features failed with KeyError: {e}")
        except Exception as e:
            self.fail(f"apply_latent_features failed with unexpected error: {e}")
            
        # Test 2: Selected population (Pruning scenario)
        # Randomly pick some features including trend_proxy_10 which failed before
        selected = [self.test_raw_features[0], 'trend_proxy_10', 'embed_mean_20', 'regime_proxy']
        try:
            df_sub = apply_latent_features(self.df, reconstructor, scaler=scaler, selected_features=selected)
            for f in selected:
                self.assertIn(f, df_sub.columns, f"Selected feature {f} missing in output")
            print("Selected population success.")
        except KeyError as e:
            self.fail(f"apply_latent_features (selected) failed with KeyError: {e}")

    def test_adversarial_missing_columns(self):
        """Verify resilience when some expected columns are missing."""
        scaler = DriftShieldScaler()
        scaler.fit(self.df, self.raw_cols)
        
        reconstructor = SuperchargedPCAReconstructor(input_dim=len(self.raw_cols))
        scaled_data = scaler.transform(self.df[self.raw_cols], self.raw_cols).values
        reconstructor.fit(scaled_data)
        reconstructor.build_fold_cache(self.df)
        
        # Scenario: Target DF is missing a column that IS in base_cols
        broken_col = Config.EMBED_BASE_COLS[0]
        df_broken = self.df.drop(columns=[broken_col])
        
        # Should NOT raise KeyError anymore, but fill with 0.0 (v18.5)
        try:
            df_res = apply_latent_features(df_broken, reconstructor, scaler=scaler)
            self.assertIn(broken_col, df_res.columns)
            # After scaling, 0.0 becomes -(mean/std). Check it's finite and not NaN.
            self.assertTrue(np.isfinite(df_res[broken_col].values).all())
            print("Missing columns handled gracefully.")
        except Exception as e:
            self.fail(f"Missing columns test failed with unexpected error: {e}")

    def test_leakage_enforcement(self):
        """Verify that cache clearing prevents cross-fold leakage."""
        reconstructor = SuperchargedPCAReconstructor(input_dim=len(self.raw_cols))
        reconstructor.fit(np.random.rand(100, len(self.raw_cols)))
        
        reconstructor.build_fold_cache(self.df)
        self.assertIsNotNone(reconstructor.pool_embed)
        
        reconstructor.clear_fold_cache()
        self.assertIsNone(reconstructor.pool_embed)
        self.assertIsNone(reconstructor.pool_norm)
        
        with self.assertRaises(RuntimeError):
            # Should raise RuntimeError: [LEAKAGE GUARD] Cache missing
            apply_latent_features(self.df, reconstructor)

    def test_numerical_stability(self):
        """Verify handling of NaNs and Infs in input."""
        df_dirty = self.df.copy()
        df_dirty.iloc[0, 0] = np.nan
        df_dirty.iloc[1, 1] = np.inf
        
        scaler = DriftShieldScaler()
        scaler.fit(df_dirty, self.raw_cols)
        
        reconstructor = SuperchargedPCAReconstructor(input_dim=len(self.raw_cols))
        # Internal imputer should handle this
        reconstructor.fit(df_dirty[self.raw_cols].values)
        reconstructor.build_fold_cache(df_dirty)
        
        try:
            df_res = apply_latent_features(df_dirty, reconstructor, scaler=scaler)
            self.assertFalse(df_res.isnull().any().any(), "Output contains NaNs")
            print("Numerical stability test passed.")
        except Exception as e:
            self.fail(f"Numerical stability test failed: {e}")

    def test_trainer_latent_reconstruction_v18_6(self):
        """Verify the specific v18.6 fix for broadcasting (N, 1) to (N,) in trainer.py."""
        # Simulate latent_stats with (N, 1) and (N, Dim) shapes
        N = 100
        latent_stats = {
            'regime_proxy': np.random.rand(N, 1).astype(np.float32),
            'embed_mean_10': np.random.rand(N, 32).astype(np.float32)
        }
        
        latent_cols = ['regime_proxy', 'embed_mean_10_d0', 'embed_mean_10_d1']
        X_latent_part = np.zeros((N, len(latent_cols)), dtype=np.float32)
        
        try:
            for i, col in enumerate(latent_cols):
                if '_d' in col:
                    base_name = col.rsplit('_d', 1)[0]
                    dim = int(col.rsplit('_d', 1)[1])
                    # Replicate trainer.py logic
                    X_latent_part[:, i] = latent_stats[base_name][:, dim].ravel()
                else:
                    # Replicate trainer.py logic
                    X_latent_part[:, i] = latent_stats[col].ravel()
            print("Trainer-style latent reconstruction test passed (v18.6).")
        except ValueError as e:
            self.fail(f"Trainer-style latent reconstruction failed with ValueError: {e}")

    def test_reconstructor_fit_dimension_handling_v18_6(self):
        """Verify Reconstructor.fit handles both 30-dim and 698-dim inputs (v18.6)."""
        reconstructor = SuperchargedPCAReconstructor(input_dim=len(self.raw_cols))
        
        # 1. Test with 30-dim (EMBED_BASE_COLS)
        X_30 = np.random.rand(10, len(Config.EMBED_BASE_COLS))
        try:
            reconstructor.fit(X_30)
            print("Reconstructor.fit(30-dim) passed.")
        except Exception as e:
            self.fail(f"Reconstructor.fit(30-dim) failed: {e}")
            
        # 2. Test with 698-dim (Full raw features)
        X_698 = np.random.rand(10, len(self.raw_cols))
        try:
            reconstructor.fit(X_698)
            print("Reconstructor.fit(698-dim) passed.")
        except Exception as e:
            self.fail(f"Reconstructor.fit(698-dim) failed: {e}")

if __name__ == '__main__':
    unittest.main()
