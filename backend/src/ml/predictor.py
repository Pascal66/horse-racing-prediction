import joblib
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union
from pathlib import Path

from sklearnex import patch_sklearn
patch_sklearn(verbose=False)  # avant tous les imports sklearn

from src.ml.safe_loader import safe_load

pd.set_option('display.max_columns', None)

import logging
logger = logging.getLogger('sklearnex')
logger.setLevel(logging.WARNING)

import os
os.environ["SKLEARNEX_VERBOSE"] = "WARNING"

class RacePredictor:
    """
    Multi-model inference module.
    Loads specialty models per discipline and a global fallback.
    """

    def __init__(self, model_dir: str = "data") -> None:
        """
        Initializes the predictor by loading all available pipelines in the directory.
        """
        self.logger = logging.getLogger("ML.Predictor")
        self.model_dir = Path(model_dir).resolve()
        self.models: Dict[str, Any] = {}

        self._load_models()

    def _load_models(self) -> None:
        """Scans the model directory and loads all model_*.pkl files."""
        if not self.model_dir.exists():
            self.logger.error(f"Model directory not found: {self.model_dir}")
            return

        self.logger.info(f"Scanning directory: {self.model_dir}")
        model_files = list(self.model_dir.glob("model_*.pkl"))
        
        for model_file in model_files:
            try:
                discipline = model_file.stem.replace("model_", "").lower()
                self.logger.info(f"Loading model '{discipline}' from {model_file.name}...")
                self.models[discipline] = safe_load(model_file)  # remplace joblib.load(model_file)
            except Exception as error:
                self.logger.error(f"Failed to load model {model_file}: {error}")

        if not self.models:
            self.logger.warning(f"No models successfully loaded from {self.model_dir}")
        else:
            self.logger.info(f"Successfully loaded {len(self.models)} models: {list(self.models.keys())}")

    @property
    def pipeline(self):
        return self.models.get("global")

    def predict_race(self, participants: Union[List[Dict[str, Any]], pd.DataFrame]) -> tuple[List[float], str]:
        """
        Performs inference. Full Pipelines handle feature engineering internally.
        Returns a tuple (probabilities, model_version).
        """
        if not self.models:
            return ([] if isinstance(participants, list) else np.array([])), "none"

        try:
            df = participants if isinstance(participants, pd.DataFrame) else pd.DataFrame(participants)
            if df.empty: return [], "none"

            discipline = "global"
            if 'discipline' in df.columns:
                disc_val = str(df['discipline'].iloc[0]).lower().strip()
                if disc_val in self.models:
                    discipline = disc_val
            
            model = self.models.get(discipline, self.models.get("global"))
            if not model: return [0.0] * len(df), "none"

            # Predict using the Pipeline (which internally calls HyperStackModel)

            probabilities = model.predict_proba(df)[:, 1]
            return probabilities.tolist(), discipline

        except Exception as error:
            self.logger.error(f"Error during prediction: {error}")
            return [0.0] * len(df), "error"
