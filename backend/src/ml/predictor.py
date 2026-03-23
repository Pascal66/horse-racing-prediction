import joblib
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any
pd.set_option('display.max_columns', None)

from pathlib import Path

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
        self.model_dir = Path(model_dir)
        self.models: Dict[str, Any] = {}
        
        self._load_models()

    def _load_models(self) -> None:
        """Scans the model directory and loads all model_*.pkl files."""
        if not self.model_dir.exists():
            self.logger.error(f"Model directory not found: {self.model_dir}")
            return

        for model_file in self.model_dir.glob("model_*.pkl"):
            try:
                # Extract discipline name from model_discipline.pkl
                discipline = model_file.stem.replace("model_", "")
                self.logger.info(f"Loading model '{discipline}' from {model_file}...")
                self.models[discipline] = joblib.load(model_file)
            except Exception as error:
                self.logger.error(f"Failed to load model {model_file}: {error}")

        if not self.models:
            self.logger.warning(f"No models loaded from {self.model_dir}")
        else:
            self.logger.info(f"Successfully loaded {len(self.models)} models: {list(self.models.keys())}")

    @property
    def pipeline(self):
        """Compatibility property: returns the global model if it exists."""
        return self.models.get("global")

    def predict_race(self, participants: List[Dict[str, Any]]) -> List[float]:
        """
        Predicts win probabilities for a list of raw participants.
        Uses a discipline-specific model if available, otherwise falls back to 'global'.
        """
        if not self.models:
            self.logger.warning("No models available for prediction.")
            return [0.0] * len(participants)

        if not participants:
            return []

        try:
            df = pd.DataFrame(participants)
            
            # Identify discipline for this race
            discipline = str(df['discipline'].iloc[0]).lower() if 'discipline' in df.columns else "unknown"

            # Selection Logic
            model = self.models.get(discipline)
            if model:
                self.logger.debug(f"Using specialty model for: {discipline}")
            else:
                model = self.models.get("global")
                if model:
                    self.logger.debug(f"No model for '{discipline}', using GLOBAL fallback.")
                else:
                    self.logger.warning("No suitable model found (no specialty, no global).")
                    return [0.0] * len(participants)

            # 2. Inference
            probabilities = model.predict_proba(df)[:, 1]
            return probabilities.tolist()

        except Exception as error:
            self.logger.error(f"Error during prediction: {error}")
            return [0.0] * len(participants)