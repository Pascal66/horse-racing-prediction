
import numpy as np
import pandas as pd
from sklearnex import patch_sklearn
patch_sklearn(verbose=False)  # avant tous les imports sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
import logging
logger = logging.getLogger('sklearnex')
logger.setLevel(logging.WARNING)

import os
os.environ["SKLEARNEX_VERBOSE"] = "WARNING"

class HyperStackModel(BaseEstimator, ClassifierMixin):
    def __init__(self, fold_models, stacker, feature_names=None):
        self.fold_models = fold_models
        self.stacker = stacker
        self.feature_names = feature_names
        self.classes_ = [0, 1]
        self._estimator_type = "classifier"

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        if self.feature_names is not None and not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        base_preds = []
        for name, fold_models in self.fold_models:
            preds = np.mean([m.predict_proba(X)[:, 1] for m in fold_models], axis=0)
            base_preds.append(preds)
        return self.stacker.predict_proba(np.vstack(base_preds).T)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)