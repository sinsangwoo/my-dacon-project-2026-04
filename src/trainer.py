import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from .config import Config

class Trainer:
    """Class to handle the 5-Fold Cross Validation and training of the LightGBM model."""
    def __init__(self, train, test, feature_cols):
        self.train = train
        self.test = test
        self.feature_cols = feature_cols
        self.kf = KFold(n_splits=Config.NFOLDS, shuffle=True, random_state=Config.SEED)
        self.oof_preds = np.zeros(len(train))
        self.test_preds = np.zeros(len(test))

    def train_kfolds(self):
        """Train the model across 5 folds and return test predictions."""
        for fold, (tr_idx, val_idx) in enumerate(self.kf.split(self.train)):
            print(f"── Fold {fold + 1} ──")
            X_tr = self.train.loc[tr_idx, self.feature_cols]
            y_tr = self.train.loc[tr_idx, Config.TARGET]
            X_val = self.train.loc[val_idx, self.feature_cols]
            y_val = self.train.loc[val_idx, Config.TARGET]

            model = LGBMRegressor(**Config.LGBM_PARAMS)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(Config.EARLY_STOPPING_ROUNDS),
                    lgb.log_evaluation(Config.LOG_EVALUATION_STEPS)
                ],
            )

            self.oof_preds[val_idx] = model.predict(X_val)
            self.test_preds += model.predict(self.test[self.feature_cols]) / Config.NFOLDS

        oof_mae = mean_absolute_error(self.train[Config.TARGET], self.oof_preds)
        print(f"OOF MAE: {oof_mae:.4f}")
        return self.test_preds
