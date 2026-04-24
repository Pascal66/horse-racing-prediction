import psycopg2
import os
from pathlib import Path
from dotenv import load_dotenv
from src.core.config import DB_URL

def apply_migrations():
    # Load environment variables
    # env_path = Path(__file__).resolve().parent / ".env"
    # load_dotenv(dotenv_path=env_path)
    #
    db_url = os.getenv("DB_URL")
    # if not db_url:
    #     print("Error: DB_URL not found in .env")
    #     return

    migrations = [
        # Horse table updates
        "ALTER TABLE horse ADD COLUMN IF NOT EXISTS breed VARCHAR(50);",
        "ALTER TABLE horse ADD COLUMN IF NOT EXISTS color VARCHAR(50);",
        "ALTER TABLE horse ADD COLUMN IF NOT EXISTS father_name VARCHAR(100);",
        "ALTER TABLE horse ADD COLUMN IF NOT EXISTS mother_name VARCHAR(100);",
        "ALTER TABLE horse ADD COLUMN IF NOT EXISTS maternal_grandfather_name VARCHAR(100);",

        # Participant table updates
        "ALTER TABLE race_participant ADD COLUMN IF NOT EXISTS blinkers VARCHAR(50);",
        "ALTER TABLE race_participant ADD COLUMN IF NOT EXISTS unraced_indicator BOOLEAN;",
        "ALTER TABLE race_participant ADD COLUMN IF NOT EXISTS career_wins_count SMALLINT;",
        "ALTER TABLE race_participant ADD COLUMN IF NOT EXISTS career_places_count SMALLINT;",
        "ALTER TABLE race_participant ADD COLUMN IF NOT EXISTS career_places_2nd_count SMALLINT;",
        "ALTER TABLE race_participant ADD COLUMN IF NOT EXISTS career_places_3rd_count SMALLINT;",
        "ALTER TABLE race_participant ADD COLUMN IF NOT EXISTS winnings_victory REAL;",
        "ALTER TABLE race_participant ADD COLUMN IF NOT EXISTS winnings_place REAL;",
        "ALTER TABLE race_participant ADD COLUMN IF NOT EXISTS winnings_year_now REAL;",
        "ALTER TABLE race_participant ADD COLUMN IF NOT EXISTS winnings_year_prev REAL;",
        "ALTER TABLE race_participant ADD COLUMN IF NOT EXISTS handicap_value REAL;",
        "ALTER TABLE race_participant ADD COLUMN IF NOT EXISTS handicap_weight REAL;",
        "ALTER TABLE race_participant ADD COLUMN IF NOT EXISTS mount_weight REAL;",
        "ALTER TABLE race_participant ADD COLUMN IF NOT EXISTS allure VARCHAR(30);",
        "ALTER TABLE race_participant ADD COLUMN IF NOT EXISTS owner_id INT REFERENCES racing_actor (actor_id);",

        # Game Advice table
        """
        CREATE TABLE IF NOT EXISTS game_advice (
            advice_id SERIAL PRIMARY KEY,
            race_id INT NOT NULL REFERENCES race(race_id),
            participant_id INT NOT NULL REFERENCES race_participant(participant_id),
            model_version VARCHAR(50),
            strategy VARCHAR(50),
            message TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT uq_race_participant_advice UNIQUE (race_id, participant_id, strategy)
        );
        """
    ]

    try:
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        with conn.cursor() as cur:
            for sql in migrations:
                print(f"Executing: {sql}")
                cur.execute(sql)
        print("Migrations applied successfully.")
        conn.close()
    except Exception as e:
        print(f"Migration failed: {e}")


if __name__ == "__main__":
    apply_migrations()
