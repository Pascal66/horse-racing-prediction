import joblib
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional
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
                self.models[discipline] = safe_load(model_file)
            except Exception as error:
                self.logger.error(f"Failed to load model {model_file}: {error}")

        if not self.models:
            self.logger.warning(f"No models successfully loaded from {self.model_dir}")
        else:
            self.logger.info(f"Successfully loaded {len(self.models)} models: {sorted(list(self.models.keys()))}")

    @property
    def pipeline(self):
        return self.models.get("global")

    def predict_race(self, participants: Union[List[Dict[str, Any]], pd.DataFrame],
                     force_algo: Optional[str] = None) -> tuple[Dict[str, List[float]], str]:
        """
        Performs inference. Full Pipelines handle feature engineering internally.
        Returns a tuple (probabilities_dict, model_version).
        """
        if not self.models:
            return {"win": [], "place": []}, "none"

        try:
            df = participants if isinstance(participants, pd.DataFrame) else pd.DataFrame(participants)
            if df.empty: return {"win": [], "place": []}, "none"

            disc_val = str(df['discipline'].iloc[0]).lower().strip() if 'discipline' in df.columns else "global"

            # Logique de sélection robuste :
            # 1. On cherche discipline_algo (ex: attele_tabnet)
            # 2. Sinon on cherche global_algo (ex: global_tabnet)
            # 3. Sinon on cherche discipline (ex: attele)
            # 4. Sinon on prend 'global' (le HyperStack par défaut)
            
            model_key = None
            if force_algo:
                test_key = f"{disc_val}_{force_algo}"
                global_test_key = f"global_{force_algo}"
                if test_key in self.models:
                    model_key = test_key
                elif global_test_key in self.models:
                    model_key = global_test_key
            
            if not model_key:
                model_key = disc_val if disc_val in self.models else "global"

            model = self.models.get(model_key)
            if not model:
                # Fallback ultime vers le premier modèle disponible si 'global' est absent
                model_key = next(iter(self.models.keys()))
                model = self.models[model_key]

            self.logger.debug(f"Using model: {model_key} for discipline: {disc_val} (force_algo={force_algo})")
            
            # LGBMRanker/TabNet/Hyperstack standard : predict_proba renvoie [P(0), P(1)]
            probabilities = model.predict_proba(df)[:, 1]

            # Inférence pour la place (Top 3)
            place_probs = [0.0] * len(df)
            if hasattr(model, "predict_place"):
                place_probs = model.predict_place(df).tolist()
            elif "ltr" in model_key:
                # Pour les modèles LTR, on peut avoir un score ordinal qu'il faudrait transformer.
                # En attendant, on utilise le même fallback que les autres.
                place_probs = (np.array(probabilities) * 2.1).clip(0, 0.98).tolist()
            else:
                # Fallback mathématique : P(Place) est souvent ~2x P(Gagnant) au PMU
                place_probs = (np.array(probabilities) * 2.2).clip(0, 0.95).tolist()

            return {
                "win": [float(p) for p in probabilities],
                "place": [float(p) for p in place_probs]
            }, model_key

        except Exception as error:
            self.logger.error(f"Error during prediction: {error}", exc_info=True)
            return {"win": [0.0] * len(df), "place": [0.0] * len(df)}, "error"
