from pathlib import Path
from src.ml.safe_loader import safe_load

from sklearnex import patch_sklearn
patch_sklearn(verbose=False)  # avant tous les imports sklearn
import logging
logger = logging.getLogger('sklearnex')
logger.setLevel(logging.WARNING)

import os
os.environ["SKLEARNEX_VERBOSE"] = "WARNING"
os.environ["SCIPY_ARRAY_API"] = "1"

from sklearn.base import BaseEstimator, TransformerMixin

class LTRBridge(BaseEstimator, TransformerMixin):
    def __init__(self, ltr_path: str = None):
        self.ltr_path = ltr_path
        self.ltr_pipeline_ = None

    def fit(self, X, y=None):
        if self.ltr_path and Path(self.ltr_path).exists():
            self.ltr_pipeline_ = safe_load(self.ltr_path)
        return self

    def transform(self, X):
        df = X.copy()
        if self.ltr_pipeline_ is not None:
            try:
                df['proba_ltr'] = self.ltr_pipeline_.predict_proba(X)[:, 1]
            except Exception:
                df['proba_ltr'] = 0.5
        else:
            df['proba_ltr'] = 0.5
        return df