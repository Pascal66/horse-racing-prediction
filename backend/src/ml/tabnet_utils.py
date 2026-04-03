# src/ml/tabnet_utils.py
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()  # avant tous les imports sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
import logging
logger = logging.getLogger('sklearnex')
logger.setLevel(logging.WARNING)

import os
os.environ["SKLEARNEX_VERBOSE"] = "WARNING"

class TabNetEnsembleWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, models=None, feature_names=None):
        self.models = models
        self.feature_names = feature_names
        self.classes_ = [0, 1]

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        if self.models is None:
            raise ValueError("Models are not initialized.")
        X_vals = X.values if hasattr(X, 'values') else X
        all_preds = [m.predict_proba(X_vals)[:, 1] for m in self.models]
        avg_preds = np.mean(all_preds, axis=0)
        return np.vstack([1 - avg_preds, avg_preds]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def __sklearn_is_fitted__(self):
        return self.models is not None and len(self.models) > 0