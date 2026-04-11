import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.src.api.main import app, get_repository, ml_models

# --- 1. MOCK CLASSES ---

class MockRaceRepository:
    """Simulates the database interactions with strict adherence to schema.py."""
    
    def get_best_model_for_context(self, discipline, month):
        return "tabnet"

    def upsert_predictions(self, preds):
        return True

    def get_races_by_date(self, date_code: str):
        return [
            {
                "race_id": 1, 
                "race_number": 1, 
                "meeting_number": 1,
                "program_date": "2025-12-28",
                "discipline": "ATTELE",
                "distance_m": 2700,
                "track_type": "C",
                "racetrack_code": "VINCENNES",
                "declared_runners_count": 14,
                "start_timestamp": 1735380000000,
                "timezone_offset": 3600000,
                "prize_money": 10000.0,
                "speciality": "ATTELE"
            }
        ]

    def get_participants_by_race(self, race_id: int):
        return [
            {
                "program_number": 1, 
                "horse_name": "Fast Horse", 
                "jockey_name": "J. Doe",
                "trainer_name": "T. Smith",
                "reference_odds": 5.4,
                "live_odds": 5.2,
                "age": 5,
                "sex": "M",
                "shoeing_status": "D4",
                "blinkers": None,
                "handicap_value": None,
                "owner_name": "Owner A",
                "finish_rank": 1,
                "incident_code": None
            }
        ]

    def get_daily_data_for_ml(self, date_code: str):
        return [
            # Case 1: Winner (Good Odds, High Edge)
            {
                "race_id": 1, "meeting_number": 1, "race_number": 1, "program_number": 1,
                "horse_name": "Sniper Choice", "reference_odds": 10.0, "live_odds": 10.0,
                "participant_id": 1, "discipline": "ATTELE", "program_date": "2025-12-28"
            },
            # Case 2: Favorite (Odds too low)
            {
                "race_id": 1, "meeting_number": 1, "race_number": 1, "program_number": 2,
                "horse_name": "Low Odds Fav", "reference_odds": 1.5, "live_odds": 1.5,
                "participant_id": 2, "discipline": "ATTELE", "program_date": "2025-12-28"
            },
            # Case 3: Longshot (Odds too high)
            {
                "race_id": 1, "meeting_number": 1, "race_number": 1, "program_number": 3,
                "horse_name": "Longshot", "reference_odds": 100.0, "live_odds": 100.0,
                "participant_id": 3, "discipline": "ATTELE", "program_date": "2025-12-28"
            },
        ]

    def get_race_data_for_ml(self, race_id: int):
        return [
            {"program_number": 1, "horse_name": "Horse A", "reference_odds": 5.0, "participant_id": 4, "discipline": "ATTELE", "program_date": "2025-12-28"},
            {"program_number": 2, "horse_name": "Horse B", "reference_odds": 10.0, "participant_id": 5, "discipline": "ATTELE", "program_date": "2025-12-28"}
        ]

class MockPredictor:
    """Simulates the ML Model."""
    
    # Accept any arguments so it can replace the real RacePredictor(path)
    def __init__(self, *args, **kwargs):
        self.pipeline = True 
        self.models = {"tabnet": True}

    def predict_race(self, participants, force_algo=None):
        count = len(participants)
        if count == 0: return {"win": [], "place": []}, "none"
        
        # 3 participants = Sniper Test
        if count == 3:
            # 1. Sniper Choice: Prob 0.20 -> Edge = 0.20 - (1/10) = 0.10 (KEEP)
            # 2. Low Odds Fav: Prob 0.60 -> Edge = 0.60 - (1/1.5) = -0.06 (IGNORE: Edge < 0.05)
            # 3. Longshot: Prob 0.01 -> Edge = 0.01 - (1/100) = 0.0 (IGNORE: Odds > 25)
            return {"win": [0.20, 0.60, 0.01], "place": [0.4, 0.8, 0.05]}, "mock_ver"
            
        # 2 participants = Single Race Prediction
        if count == 2:
            return {"win": [0.8, 0.2], "place": [0.9, 0.4]}, "mock_ver"
            
        return {"win": [0.0] * count, "place": [0.0] * count}, "mock_ver"

# --- 2. FIXTURES ---

@pytest.fixture
def client():
    # 1. Override the DB Repository
    app.dependency_overrides[get_repository] = MockRaceRepository
    
    # 2. PATCH the RacePredictor class in main.py
    # When main.py calls RacePredictor(...), it will get our MockPredictor(...) instead.
    # This prevents the real model (and its heavy pickle file) from ever loading.
    with patch("backend.src.api.main.RacePredictor", side_effect=MockPredictor):
        with patch("backend.src.api.main.DatabaseManager") as mock_db:
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
    assert data[0]["meeting_number"] == 1

def test_get_participants(client):
    response = client.get("/races/1/participants")
    assert response.status_code == 200
    data = response.json()
    assert data[0]["jockey_name"] == "J. Doe"

def test_sniper_bets_logic(client):
    response = client.get("/bets/sniper/28122025")
    assert response.status_code == 200
    bets = response.json()
    
    # Filter only Sniper strategy (exclude Kelly)
    sniper_bets = [b for b in bets if b["strategy"].startswith("Sniper")]

    assert len(sniper_bets) == 1
    bet = sniper_bets[0]
    
    assert bet["horse_name"] == "Sniper Choice"
    assert bet["strategy"].startswith("Sniper")
    assert bet["edge"] == pytest.approx(0.10, abs=0.01)

def test_predict_race(client):
    response = client.get("/races/1/predict")
    assert response.status_code == 200
    results = response.json()
    
    assert results[0]["predicted_rank"] == 1
    assert results[0]["win_probability"] == 0.8
