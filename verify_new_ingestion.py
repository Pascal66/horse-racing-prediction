import json
import os
import sys
from unittest.mock import MagicMock, patch

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from src.ingestion.participants import ParticipantsIngestor
from src.core.database import DatabaseManager

def test_ingestion():
    # Load sample data
    with open("doc/participants-01042026-4-9.json", "r") as f:
        data = json.load(f)

    participant = data['participants'][0]

    # Mock DB
    with patch("src.ingestion.base.DatabaseManager") as MockDB:
        mock_instance = MockDB.return_value
        mock_conn = mock_instance.get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value.__enter__.return_value

        ingestor = ParticipantsIngestor("01042026")

        # 1. Test Horse Ingestion directly
        mock_cursor.fetchone.side_effect = [(100,)]
        h_id = ingestor._get_or_create_horse(participant)

        horse_sql = mock_cursor.execute.call_args_list[0][0][0]
        horse_params = mock_cursor.execute.call_args_list[0][0][1]

        print("Horse SQL Check:")
        print(f"Params: {horse_params}")

        assert "breed" in horse_sql
        assert horse_params[3] == "PUR-SANG"

        # 2. Test Participant Ingestion
        mock_cursor.execute.reset_mock()
        # In _insert_participant, for participant[0]:
        # 1. _get_or_create_horse calls fetchone
        # 2. trainer calls fetchone
        # 3. driver calls fetchone
        # 4. incident (None) -> no fetchone
        # 5. shoeing (None) -> no fetchone
        # 6. owner calls fetchone

        mock_cursor.fetchone.side_effect = [
            (100,), # horse_id
            (200,), # trainer_id
            (201,), # driver_id
            (500,), # owner_id
        ]

        ingestor._insert_participant(mock_cursor, 99, participant)

        # Find the main participant INSERT query
        p_insert_calls = [call for call in mock_cursor.execute.call_args_list if "INSERT INTO race_participant" in call[0][0]]
        assert len(p_insert_calls) == 1

        p_sql = p_insert_calls[0][0][0]
        p_params = p_insert_calls[0][0][1]

        print("\nParticipant SQL Check:")
        print(f"Params: {p_params}")

        # Verify blinkers (index 18)
        assert p_params[18] == "OEILLERES_CLASSIQUE"
        # Verify winnings_victory (index 24) -> 4950000 / 100 = 49500.0
        assert p_params[24] == 49500.0
        # Verify handicap_value (index 28)
        assert p_params[28] == 53.5
        # Verify owner_id (index 32)
        assert p_params[32] == 500

    print("\nVerification script finished successfully.")

if __name__ == "__main__":
    test_ingestion()
