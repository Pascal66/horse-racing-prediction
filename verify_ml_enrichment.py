import os
import sys
import pandas as pd
from unittest.mock import MagicMock, patch

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from src.api.repositories import RaceRepository
from src.ml.features import PmuFeatureEngineer

def test_repo_fetch():
    repo = RaceRepository()
    # Mocking DB connection and response
    with patch("src.core.database.DatabaseManager.get_connection") as mock_get_conn:
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_cursor = mock_conn.cursor.return_value.__enter__.return_value

        # Sample data returning our new columns
        mock_cursor.fetchall.return_value = [{
            'race_id': 1, 'horse_name': 'TEST_HORSE', 'program_number': 1,
            'breed': 'PUR-SANG', 'blinkers': 'OEILLERES', 'handicap_value': 50.5,
            'career_wins_count': 2, 'career_races_count': 10,
            'winnings_victory': 1000.0, 'father_name': 'PAPA'
        }]

        data = repo.get_race_data_for_ml(1)
        print("Repo fetched data keys:", data[0].keys())
        assert 'breed' in data[0]
        assert 'blinkers' in data[0]
        assert data[0]['father_name'] == 'PAPA'

def test_feature_engineering():
    engineer = PmuFeatureEngineer()
    # Include career_winnings to avoid the default "Debutant" logic overwriting races_count
    df = pd.DataFrame([{
        'career_wins_count': 2,
        'career_races_count': 10,
        'career_winnings': 5000,
        'breed': 'PUR-SANG',
        'blinkers': None # Should be filled
    }])

    transformed = engineer.transform(df)
    print("\nTransformed columns sample:", transformed.columns.tolist()[:10])
    win_rate = transformed['career_win_rate'].iloc[0]
    print(f"Career win rate: {win_rate}")

    assert 'career_win_rate' in transformed.columns
    # 2 wins / (10 races + 1) = 2/11
    assert abs(win_rate - (2 / 11)) < 1e-6
    assert transformed['blinkers'].iloc[0] == 'MISSING'
    assert transformed['breed'].iloc[0] == 'PUR-SANG'

if __name__ == "__main__":
    test_repo_fetch()
    test_feature_engineering()
    print("\nVerification successful.")
