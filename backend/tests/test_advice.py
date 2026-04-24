import pytest
import asyncio
from unittest.mock import patch, MagicMock
from backend.src.cli.cronJobs import generate_and_send_advice
from datetime import datetime, timezone


class MockRepository:
    def get_race_data_for_ml(self, race_id):
        return [
            {
                "participant_id": 1,
                "race_id": race_id,
                "pmu_number": 1,
                "horse_name": "Test Horse",
                "discipline": "ATTELE",
                "start_timestamp": 1735392000000,
                "reference_odds": 5.0,
                "live_odds": None,
                "live_odds_30mn": None
            }
        ]

    def get_best_model_for_context(self, discipline, month):
        return "mock_model"

    def get_races_by_date(self, date_code):
        return [
            {
                "race_id": 1,
                "meeting_libelle": "Vincennes",
                "race_number": 1,
                "meeting_number": 1
            }
        ]

    def upsert_game_advice(self, advice_data):
        return True


class MockPredictor:
    def __init__(self, *args, **kwargs):
        pass

    def predict_race(self, participants, force_algo=None):
        return {"win": [0.8], "place": [0.9]}, "mock_model_v1"


@pytest.mark.asyncio
async def test_generate_and_send_advice():
    with patch("backend.src.cli.cronJobs.RaceRepository", return_value=MockRepository()), \
            patch("backend.src.cli.cronJobs.RacePredictor", side_effect=MockPredictor), \
            patch("backend.src.cli.cronJobs.send_telegram_message", new_callable=MagicMock) as mock_send:
        # We need to mock asyncio.run(send_telegram_message(...)) if it's called via run_send_telegram
        # but here we test generate_and_send_advice directly which is async.

        await generate_and_send_advice(1)

        assert mock_send.called
        args, kwargs = mock_send.call_args
        message = args[0]
        assert "CONSEIL DE JEU" in message
        assert "Test Horse" in message
        assert "80.0%" in message
