-- Migration: Create game_advice table
-- Stores betting recommendations sent to users (e.g., via Telegram).

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
