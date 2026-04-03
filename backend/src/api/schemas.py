from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime

class RaceSummary(BaseModel):
    race_id: int
    meeting_number: int
    race_number: int
    program_date: datetime
    discipline: str
    distance_m: int
    track_type: Optional[str]
    racetrack_code: str
    declared_runners_count: int
    start_timestamp: Optional[int]
    timezone_offset: Optional[int]
    prize_money: Optional[float]
    speciality: Optional[str]

class ParticipantSummary(BaseModel):
    program_number: int
    horse_name: str
    age: int
    sex: str
    jockey_name: Optional[str]
    trainer_name: Optional[str]
    reference_odds: Optional[float]
    live_odds: Optional[float]
    shoeing_status: Optional[str]
    blinkers: Optional[str] = None
    handicap_value: Optional[float] = None
    owner_name: Optional[str] = None

class PredictionResult(BaseModel):
    program_number: int
    horse_name: str
    win_probability: float
    predicted_rank: int

class BetRecommendation(BaseModel):
    race_id: int
    meeting_num: int
    race_num: int
    horse_name: str
    program_number: int
    odds: float
    win_probability: float
    edge: float
    strategy: str

class ModelMetric(BaseModel):
    model_name: str
    algorithm: Optional[str] = "xgboost" # Made optional with default
    segment_type: str
    segment_value: str
    test_month: int
    num_races: int
    logloss: float
    auc: float
    roi: float
    win_rate: float
    avg_odds: float
    updated_at: datetime
