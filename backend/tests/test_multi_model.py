import pytest
from unittest.mock import MagicMock, patch
from backend.src.ml.predictor import RacePredictor

def test_race_predictor_selection():
    """
    Verifies that the RacePredictor selects the correct model based on discipline.
    """
    with patch("joblib.load") as mock_load, \
         patch("pathlib.Path.glob") as mock_glob, \
         patch("pathlib.Path.exists") as mock_exists:

        mock_exists.return_value = True

        # Mock available model files
        mock_file_global = MagicMock()
        mock_file_global.stem = "model_global"
        mock_file_global.suffix = ".pkl"
        mock_file_global.__str__.return_value = "model_global"

        mock_file_trot = MagicMock()
        mock_file_trot.stem = "model_trotting"
        mock_file_trot.suffix = ".pkl"
        mock_file_trot.__str__.return_value = "model_trotting"

        mock_glob.return_value = [mock_file_global, mock_file_trot]

        # Mock the loaded models
        mock_model_global = MagicMock()
        mock_model_trot = MagicMock()

        def side_effect(path):
            if "global" in str(path): return mock_model_global
            if "trotting" in str(path): return mock_model_trot
            return None

        mock_load.side_effect = side_effect

        predictor = RacePredictor("fake_dir")

        # Test Case 1: Specialty model exists
        participants_trot = [{"discipline": "TROTTING", "horse_name": "Horse 1"}]
        predictor.predict_race(participants_trot)
        mock_model_trot.predict_proba.assert_called_once()

        # Test Case 2: Fallback to global
        participants_gallop = [{"discipline": "GALLOP", "horse_name": "Horse 2"}]
        predictor.predict_race(participants_gallop)
        mock_model_global.predict_proba.assert_called_once()

        # Test Case 3: Missing discipline column (fallback to global)
        participants_none = [{"horse_name": "Horse 3"}]
        predictor.predict_race(participants_none)
        assert mock_model_global.predict_proba.call_count == 2
