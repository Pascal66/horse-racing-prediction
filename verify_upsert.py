import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Adjust path to import from backend
sys.path.append(os.getcwd())

from backend.src.ingestion.participants import ParticipantsIngestor

class TestParticipantUpsert(unittest.TestCase):
    def test_upsert_sql_logic(self):
        # Setup
        ingestor = ParticipantsIngestor("01012025")
        mock_cursor = MagicMock()

        # Mocking external entity creation to focus on the INSERT statement
        ingestor._get_or_create_horse = MagicMock(return_value=1)
        ingestor._get_or_create_actor = MagicMock(return_value=10)
        ingestor._get_or_create_incident = MagicMock(return_value=100)
        ingestor._get_or_create_shoeing = MagicMock(return_value=1000)

        participant_data = {
            "numPmu": 5,
            "nom": "BOLERO",
            "age": 7,
            "sexe": "MALE",
            "nombreCourses": 50,
            "gainsParticipant": {"gainsCarriere": 12000000}, # in cents
            "dernierRapportReference": {"rapport": 4.5},
            "dernierRapportDirect": {"rapport": 5.2},
            "musique": "1a2a3a",
            "avisEntraineur": "Good",
            "ordreArrivee": 1,
            "tempsObtenu": 165000,
            "reductionKilometrique": 11.5
        }

        # Execute
        ingestor._insert_participant(mock_cursor, 101, participant_data)

        # Verify
        # Get the call args of execute
        args, _ = mock_cursor.execute.call_args
        sql_query = args[0]

        print("\nGenerated SQL Query:")
        print(sql_query)

        self.assertIn("ON CONFLICT (race_id, pmu_number) DO UPDATE SET", sql_query)
        self.assertIn("live_odds = EXCLUDED.live_odds", sql_query)
        self.assertIn("reduction_km = EXCLUDED.reduction_km", sql_query)

        print("\nVerification SUCCESS: SQL query now includes UPSERT logic.")

if __name__ == "__main__":
    unittest.main()
