import joblib
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union
from pathlib import Path

pd.set_option('display.max_columns', None)

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
        # Ensure path is absolute for consistency
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
        
        if not model_files:
            self.logger.warning(f"No .pkl files starting with 'model_' found in {self.model_dir}")
            return

        for model_file in model_files:
            try:
                # Extract discipline name from model_discipline.pkl
                discipline = model_file.stem.replace("model_", "").lower()
                self.logger.info(f"Loading model '{discipline}' from {model_file.name}...")
                self.models[discipline] = joblib.load(model_file)
            except Exception as error:
                self.logger.error(f"Failed to load model {model_file}: {error}")

        if not self.models:
            self.logger.warning(f"No models successfully loaded from {self.model_dir}")
        else:
            self.logger.info(f"Successfully loaded {len(self.models)} models: {list(self.models.keys())}")

    @property
    def pipeline(self):
        """Compatibility property: returns the global model if it exists."""
        return self.models.get("global")

    def predict_race(self, participants: Union[List[Dict[str, Any]], pd.DataFrame]) -> List[float]:
        """
        Predicts win probabilities for raw participants.
        Supports both list of dicts or a single DataFrame.
        """
        if not self.models:
            self.logger.warning("No models available for prediction.")
            return [] if isinstance(participants, list) else np.array([])

        try:
            df = participants if isinstance(participants, pd.DataFrame) else pd.DataFrame(participants)
            if df.empty: return []

            # Determine the model to use based on discipline
            # If all participants belong to the same race/discipline (standard case for single race)
            discipline = "global"
            if 'discipline' in df.columns:
                disc_val = str(df['discipline'].iloc[0]).lower().strip()
                if disc_val in self.models:
                    discipline = disc_val
            
            model = self.models.get(discipline, self.models.get("global"))
            
            if not model:
                self.logger.error("No model found (specialty or global fallback).")
                return [0.0] * len(df)

            self.logger.debug(f"Predicting {len(df)} rows using '{discipline}' model.")
            
            # Use the pipeline (Engineer -> Preprocessor -> Calibrated Model)
            probabilities = model.predict_proba(df)[:, 1]
            return probabilities.tolist()

        except Exception as error:
            self.logger.error(f"Error during prediction: {error}")
            return [0.0] * len(df)
