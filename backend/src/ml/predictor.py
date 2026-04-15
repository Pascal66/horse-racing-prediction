import joblib
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional
from pathlib import Path

from sklearnex import patch_sklearn
patch_sklearn(verbose=False)

from src.ml.safe_loader import safe_load

pd.set_option('display.max_columns', None)
logger = logging.getLogger('ML.Predictor')

class RacePredictor:
    def __init__(self, model_dir: str = "data") -> None:
        self.logger = logging.getLogger("ML.Predictor")
        self.model_dir = Path(model_dir).resolve()
        self.models: Dict[str, Any] = {}
        self._load_models()

    def _load_models(self) -> None:
        if not self.model_dir.exists(): return
        model_files = list(self.model_dir.glob("model_*.pkl"))
        for model_file in model_files:
            try:
                discipline = model_file.stem.replace("model_", "").lower()
                self.models[discipline] = safe_load(model_file)
            except Exception as error:
                self.logger.error(f"Failed to load model {model_file}: {error}")
        self.logger.info(f"Successfully loaded {len(self.models)} models: {sorted(list(self.models.keys()))}")

    def predict_race(self, participants: Union[List[Dict[str, Any]], pd.DataFrame],
                     force_algo: Optional[str] = None) -> tuple[Dict[str, List[float]], str]:
        if not self.models: return {"win": [], "place": []}, "none"

        try:
            df = participants if isinstance(participants, pd.DataFrame) else pd.DataFrame(participants)
            if df.empty: return {"win": [], "place": []}, "none"

            disc_val = str(df['discipline'].iloc[0]).lower().strip() if 'discipline' in df.columns else "global"
            
            # Normalisation du nom de l'algo (ex: 'ltr_only' -> 'ltr')
            algo_norm = force_algo.replace("_only", "") if force_algo else None
            
            model_key = None
            if algo_norm:
                # 1. Test discipline + suffixe (ex: plat_ltr)
                if f"{disc_val}_{algo_norm}" in self.models:
                    model_key = f"{disc_val}_{algo_norm}"
                # 2. Test suffixe seul (ex: ltr)
                elif algo_norm in self.models:
                    model_key = algo_norm
                # 3. Test global + suffixe (ex: global_ltr)
                elif f"global_{algo_norm}" in self.models:
                    model_key = f"global_{algo_norm}"
            
            if not model_key:
                model_key = disc_val if disc_val in self.models else "global"

            model = self.models.get(model_key)
            if not model:
                model_key = next(iter(self.models.keys()))
                model = self.models[model_key]

            self.logger.info(f"Predicting with model: {model_key} (requested: {force_algo})")
            
            probabilities = model.predict_proba(df)[:, 1]
            place_probs = (np.array(probabilities) * 2.2).clip(0, 0.95).tolist()

            return {
                "win": [float(p) for p in probabilities],
                "place": [float(p) for p in place_probs]
            }, model_key

        except Exception as error:
            self.logger.error(f"Error during prediction: {error}", exc_info=True)
            return {"win": [0.0] * len(df), "place": [0.0] * len(df)}, "error"
