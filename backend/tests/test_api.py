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
                "race_number": 1, 
                "meeting_number": 1,
                "program_date": "2025-12-28T00:00:00",
                "discipline": "HARNESS",
                "distance_m": 2700,
                "track_type": "DIRT",
                "racetrack_code": "VINCENNES",
                "declared_runners_count": 14,
                "start_timestamp": 1735344000000,
                "timezone_offset": 3600000,
                "prize_money": 50000.0,
                "speciality": "TROT_ATTELE"
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
                "live_odds": 5.5,
                "age": 5,
                "sex": "M",
                "finish_rank": 2,
                "incident_code": "DAI"
            }
        ]

    def get_daily_data_for_ml(self, date_code: str):
        return [
            # Case 1: Winner (Good Odds, High Edge)
            {
                "race_id": 1, "meeting_number": 1, "race_number": 1, "program_number": 1,
                "horse_name": "Sniper Choice", "reference_odds": 10.0, "live_odds": 10.0
            },
            # Case 2: Favorite (Odds too low)
            {
                "race_id": 1, "meeting_number": 1, "race_number": 1, "program_number": 2,
                "horse_name": "Low Odds Fav", "reference_odds": 2.0, "live_odds": 2.0
            },
            # Case 3: Longshot (Odds too high)
            {
                "race_id": 1, "meeting_number": 1, "race_number": 1, "program_number": 3,
                "horse_name": "Longshot", "reference_odds": 50.0, "live_odds": 50.0
            },
        ]

    def get_race_data_for_ml(self, race_id: int):
        return [
            {"program_number": 1, "horse_name": "Horse A", "reference_odds": 5.0, "live_odds": 5.0},
            {"program_number": 2, "horse_name": "Horse B", "reference_odds": 10.0, "live_odds": 10.0}
        ]

class MockPredictor:
    """Simulates the ML Model."""
    
    # Accept any arguments so it can replace the real RacePredictor(path)
    def __init__(self, *args, **kwargs):
        self.pipeline = True
        self.models = {"global": True}

    def predict_race(self, participants):
        # Handle both list of dicts (from repository) and DataFrame (some internal calls)
        import pandas as pd
        if isinstance(participants, pd.DataFrame):
            count = len(participants)
        else:
            count = len(participants)

        if count == 0: return []
        
        # 3 participants = Sniper Test
        if count == 3:
            # 1. Sniper Choice: Prob 0.20 -> Edge = 0.20 - (1/10) = 0.10 (KEEP)
            # Order in get_daily_data_for_ml: 1, 2, 3
            return [0.20, 0.60, 0.05]
            
        # 2 participants = Single Race Prediction
        if count == 2:
            return [0.8, 0.2]
            
        # If count is 1 (get_participants returns 1)
        if count == 1:
            return [1.0]

        return [0.0] * count

# --- 2. FIXTURES ---

@pytest.fixture
def client():
    # 1. Override the DB Repository
    app.dependency_overrides[get_repository] = MockRaceRepository
    
    # 2. PATCH the RacePredictor class in main.py
    # When main.py calls RacePredictor(...), it will get our MockPredictor(...) instead.
    # This prevents the real model (and its heavy pickle file) from ever loading.
    with patch("backend.src.api.main.RacePredictor", side_effect=MockPredictor):
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
    assert response.json()["scheduler"]["status"] == "running"

def test_get_races(client):
    response = client.get("/races/28122025")
    assert response.status_code == 200
    data = response.json()
    assert data[0]["racetrack_code"] == "VINCENNES"

def test_get_participants(client):
    response = client.get("/races/1/participants")
    assert response.status_code == 200
    data = response.json()
    assert data[0]["jockey_name"] == "J. Doe"
    assert data[0]["finish_rank"] == 2
    assert data[0]["incident_code"] == "DAI"

def test_sniper_bets_logic(client):
    response = client.get("/bets/sniper/28122025")
    assert response.status_code == 200
    bets = response.json()
    
    # We have Sniper and Kelly bets now
    sniper_bets = [b for b in bets if "Sniper" in b["strategy"]]
    assert len(sniper_bets) >= 1
    
    bet = sniper_bets[0]
    # In MockPredictor: [0.20, 0.60, 0.05] for [10.0, 2.0, 50.0]
    # Edge1 = 0.2 - 1/10 = 0.1
    # Edge2 = 0.6 - 1/2 = 0.1
    # Both have edge 0.1 and odds >= 2.0. Best is Horse 2 because higher prob.
    assert bet["horse_name"] == "Low Odds Fav"
    assert bet["edge"] == pytest.approx(0.10, abs=0.01)

def test_predict_race(client):
    response = client.get("/races/1/predict")
    assert response.status_code == 200
    results = response.json()
    
    assert results[0]["predicted_rank"] == 1
    assert results[0]["win_probability"] == 0.8