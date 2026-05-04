import numpy as np
import logging
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from .config import Config
from . import utils

logger = logging.getLogger(__name__)

class ExplosionInference:
    """Restored Leaderboard Explosion Architecture v5."""

    @staticmethod
    def _build_meta_features(oof_raw, oof_embed, regime_proxy):
        """SSOT-aligned meta features."""
        # 3-dim meta features for extreme robustness
        meta = np.column_stack([oof_raw, oof_embed, regime_proxy])
        return meta.astype(np.float32)

    @classmethod
    def train_and_infer(cls, oof_raw, oof_embed, regime_tr,
                        test_raw, test_embed, regime_te,
                        y_train, train_stats):
        
        # [PHASE 4: META MODEL REACTIVATION]
        logger.info("[EXPLOSION] Activating meta-model stack...")
        
        # 1. Correlation Check
        corr = np.corrcoef(oof_raw, oof_embed)[0, 1]
        logger.info(f"[META] Raw vs Embed Correlation: {corr:.4f}")
        
        # 2. Meta Features
        X_meta_tr = cls._build_meta_features(oof_raw, oof_embed, regime_tr)
        X_meta_te = cls._build_meta_features(test_raw, test_embed, regime_te)
        
        # 3. Ridge Stacking (Continuous Weighting, NO Alpha Collapse)
        model = Ridge(alpha=1.0)
        utils.SAFE_FIT(model, X_meta_tr, y_train)
        
        final_preds = utils.SAFE_PREDICT(model, X_meta_te)
        
        # 4. Clipping
        final_preds = np.clip(final_preds, 0.0, train_stats['p99'] * 1.5)
        
        logger.info(f"[EXPLOSION] Meta-model inference complete. Mean: {final_preds.mean():.4f}")
        return final_preds
