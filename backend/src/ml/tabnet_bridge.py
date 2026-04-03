# src/ml/tabnet_bridge.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearnex import patch_sklearn
patch_sklearn(verbose=False)  # avant tous les imports sklearn
import logging
logger = logging.getLogger('sklearnex')
logger.setLevel(logging.WARNING)

import os
os.environ["SKLEARNEX_VERBOSE"] = "WARNING"

from sklearn.base import BaseEstimator, TransformerMixin
from src.ml.safe_loader import safe_load

class TabNetBridge(BaseEstimator, TransformerMixin):
    """
    Transformer sklearn qui ajoute la proba TabNet comme feature.
    S'intègre dans le Pipeline HS — calcule proba_tabnet à l'entraînement
    ET à l'inférence automatiquement.
    """
    def __init__(self, tabnet_path: str = None):
        self.tabnet_path = tabnet_path
        self.tabnet_pipeline_ = None

    def fit(self, X, y=None):
        if self.tabnet_path and Path(self.tabnet_path).exists():
            self.tabnet_pipeline_ = safe_load(self.tabnet_path)
        return self

    def transform(self, X):
        df = X.copy()
        if self.tabnet_pipeline_ is not None:
            try:
                df['proba_tabnet'] = self.tabnet_pipeline_.predict_proba(X)[:, 1]
            except Exception:
                df['proba_tabnet'] = 0.5  # fallback neutre
        else:
            df['proba_tabnet'] = 0.5
        return df