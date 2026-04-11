import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.src.api.main import app, get_repository, ml_models


# --- 1. MOCK CLASSES ---

class MockRaceRepository:
    """Simulates the database interactions with strict adherence to schema.py."""

    def get_races_by_date(self, date_code: str):
        return [
            {
                "race_id": 1,
                "meeting_number": 1,
                "race_number": 1,
                "program_date": "2025-12-28T00:00:00",
                "discipline": "ATTELE",
                "distance_m": 2700,
                "track_type": "C",
                "racetrack_code": "VINC",
                "declared_runners_count": 14,
                "start_timestamp": 1735372800000,
                "timezone_offset": 3600,
                "prize_money": 50000.0,
                "speciality": "TROT"
            }
        ]

    def get_participants_by_race(self, race_id: int):
        return [
            {
                "program_number": 1,
                "horse_name": "Fast Horse",
                "age": 5,
                "sex": "M",
                "jockey_name": "J. Doe",
                "trainer_name": "T. Smith",
                "reference_odds": 5.4,
                "live_odds": 6.0,
                "shoeing_status": "D4",
                "blinkers": "NONE",
                "handicap_value": 0.0,
                "owner_name": "Owner A",
                "finish_rank": 1,
                "incident_code": None
            }
        ]

    def get_daily_data_for_ml(self, date_code: str):
        return [
            # Case 1: Winner (Good Odds, High Edge)
            {
                "participant_id": 101, "race_id": 1, "meeting_number": 1, "race_number": 1, "program_number": 1,
                "horse_name": "Sniper Choice", "reference_odds": 10.0, "live_odds": 10.0, "discipline": "ATTELE",
                "program_date": "2025-12-28"
            },
            # Case 2: Favorite (Odds too low)
            {
                "participant_id": 102, "race_id": 1, "meeting_number": 1, "race_number": 1, "program_number": 2,
                "horse_name": "Low Odds Fav", "reference_odds": 2.0, "live_odds": 2.0, "discipline": "ATTELE",
                "program_date": "2025-12-28"
            },
            # Case 3: Longshot (Odds too high)
            {
                "participant_id": 103, "race_id": 1, "meeting_number": 1, "race_number": 1, "program_number": 3,
                "horse_name": "Longshot", "reference_odds": 50.0, "live_odds": 50.0, "discipline": "ATTELE",
                "program_date": "2025-12-28"
            },
        ]

    def get_race_data_for_ml(self, race_id: int):
        return [
            {"participant_id": 201, "program_number": 1, "horse_name": "Horse A", "reference_odds": 5.0},
            {"participant_id": 202, "program_number": 2, "horse_name": "Horse B", "reference_odds": 10.0}
        ]

    def upsert_predictions(self, predictions):
        return True


class MockPredictor:
    """Simulates the ML Model."""

    # Accept any arguments so it can replace the real RacePredictor(path)
    def __init__(self, *args, **kwargs):
        self.models = {"global": True}

    def predict_race(self, participants):
        count = len(participants)
        if count == 0: return [], "mock"

        # 3 participants = Sniper Test
        if count == 3:
            # sniper_mask = (df['edge'] >= MIN_EDGE) & (df['effective_odds'] >= MIN_ODDS) & (df['effective_odds'] <= MAX_ODDS)
            # Sniper Choice (odds 10.0, prob 0.20) -> Edge = 0.20 - 0.10 = 0.10 (QUALIFIES)
            # Low Odds Fav (odds 2.0, prob 0.51) -> Edge = 0.51 - 0.50 = 0.01 (DISCARDED, MIN_EDGE=0.05)
            # Longshot (odds 50.0, prob 0.05) -> Edge = 0.05 - 1/50 = 0.03 (DISCARDED)
            # Max odds for Sniper is 25.0, so odds 50.0 is DISCARDED by odds filter anyway.
            return [0.20, 0.51, 0.05], "mock"

        # 2 participants = Single Race Prediction
        if count == 2:
            return [0.8, 0.2], "mock"

        return [0.0] * count, "mock"


# --- 2. FIXTURES ---

@pytest.fixture
def client():
    # 1. Override the DB Repository
    app.dependency_overrides[get_repository] = MockRaceRepository

    # 2. PATCH the RacePredictor class and DatabaseManager in main.py
    # When main.py calls RacePredictor(...), it will get our MockPredictor(...) instead.
    # This prevents the real model (and its heavy pickle file) from ever loading.
    with patch("backend.src.api.main.RacePredictor", side_effect=MockPredictor), \
            patch("backend.src.api.main.DatabaseManager") as mock_db:
        mock_db.return_value.initialize_pool.return_value = None
        with TestClient(app) as c:
            yield c

    # Cleanup
    app.dependency_overrides.clear()
    ml_models.clear()


# --- 3. TESTS ---

def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["ml_engine"] == "loaded"


def test_get_races(client):
    response = client.get("/races/28122025")
    assert response.status_code == 200
    data = response.json()
    assert data[0]["discipline"] == "ATTELE"


def test_get_participants(client):
    response = client.get("/races/1/participants")
    assert response.status_code == 200
    data = response.json()
    assert data[0]["jockey_name"] == "J. Doe"


def test_sniper_bets_logic(client):
    response = client.get("/bets/sniper/28122025")
    assert response.status_code == 200
    bets = response.json()

    # Filter for Sniper strategy as Kelly might also produce recommendations
    sniper_bets = [b for b in bets if b["strategy"].startswith("Sniper")]

    assert len(sniper_bets) == 1
    bet = sniper_bets[0]

    assert bet["horse_name"] == "Sniper Choice"
    assert bet["strategy"].startswith("Sniper")
    # Edge is boosted by market signal in main.py: edge = edge + market_signal * 0.1
    # In Mock, market_signal is not present, so it defaults to 0.
    # Edge = 0.20 - (1/10.0) = 0.10
    assert bet["edge"] == pytest.approx(0.10, abs=0.01)


def test_predict_race(client):
    response = client.get("/races/1/predict")
    assert response.status_code == 200
    results = response.json()

    assert results[0]["predicted_rank"] == 1
    assert results[0]["win_probability"] == 0.8