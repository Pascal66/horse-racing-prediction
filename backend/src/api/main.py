"""
Main entry point for the FastAPI application.
"""
import logging
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager
from pathlib import Path
import os

import numpy as np
from fastapi import FastAPI, Depends, HTTPException, status
import pandas as pd

from .schemas import (
    RaceSummary, 
    ParticipantSummary, 
    PredictionResult, 
    BetRecommendation,
    ModelMetric
)
from src.core.database import DatabaseManager
from src.api.repositories import RaceRepository
from src.api.backtest_service import BacktestService
from src.api.kelly_multi_races import analyze_multiple_races
from src.cli.cronJobs import cronjobs, get_scheduler
from src.ml.predictor import RacePredictor

pd.set_option('future.no_silent_downcasting', True)

# --- CONFIGURATION: SNIPER STRATEGY ---
MIN_EDGE = 0.05
MIN_ODDS = 2.0
MAX_ODDS = 25.0

# Logger Configuration
LOG_FILE = "app.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger("API")

ml_models: Dict[str, Optional[RacePredictor]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing ML Pipeline...")
    db = DatabaseManager()
    db.initialize_pool()
    try:
        current_file = Path(__file__).resolve()
        possible_data_path = current_file.parents[3] / "data"
        if not possible_data_path.exists(): possible_data_path = Path("data")
        model_path = Path(os.getenv("MODEL_PATH", possible_data_path))
        ml_models["predictor"] = RacePredictor(str(model_path))
    except Exception as exc:
        logger.error(f"CRITICAL: Failed to load ML model ({exc}).")
        ml_models["predictor"] = None
    yield
    db.close_pool()
    ml_models.clear()

app = FastAPI(title="PMU Predictor API", lifespan=lifespan)
cronjobs()

def get_repository() -> RaceRepository:
    return RaceRepository()

# --- ROUTES ---

@app.get("/", tags=["System"])
def health_check() -> Dict[str, Any]:
    predictor = ml_models.get("predictor")
    model_status = "loaded" if predictor and predictor.models else "failed"
    scheduler = get_scheduler()
    scheduler_status = "running" if scheduler and scheduler.running else "stopped"
    jobs = []
    if scheduler:
        for job in scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": str(job.next_run_time) if job.next_run_time else None
            })

    return {
        "status": "online",
        "ml_engine": model_status,
        "available_models": sorted(list(predictor.models.keys())) if predictor else [],
        "scheduler": {
            "status": scheduler_status,
            "jobs": jobs
        }
    }


@app.get("/logs", tags=["System"])
def get_logs(lines: int = 100) -> Dict[str, List[str]]:
    """Retrieves the last N lines of the application log."""
    if not os.path.exists(LOG_FILE):
        return {"logs": ["Log file not found."]}

    try:
        with open(LOG_FILE, "r") as f:
            all_lines = f.readlines()
            return {"logs": all_lines[-lines:]}
    except Exception as e:
        return {"logs": [f"Error reading logs: {e}"]}


@app.post("/jobs/{job_id}/run", tags=["System"])
def run_job(job_id: str) -> Dict[str, str]:
    """Manually triggers a scheduled job."""
    scheduler = get_scheduler()
    if not scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not available.")

    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    try:
        job.modify(next_run_time=pd.Timestamp.now())
        return {"message": f"Job '{job_id}' triggered successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger job: {e}")

@app.get("/metrics", response_model=List[ModelMetric], tags=["ML Metrics"])
def get_model_metrics(model_name: Optional[str] = None, segment_type: Optional[str] = None, repository: RaceRepository = Depends(get_repository)) -> List[Dict[str, Any]]:
    metrics = repository.get_model_metrics(model_name, segment_type)
    
    sanitized = []
    for m in metrics:
        clean_m = dict(m)
        for field in ['logloss', 'auc', 'roi', 'win_rate', 'avg_odds']:
            val = clean_m.get(field)
            if val == 'N/A' or val == '':
                clean_m[field] = None
            elif isinstance(val, str):
                try: clean_m[field] = float(val)
                except: clean_m[field] = None
        sanitized.append(clean_m)
    return sanitized

@app.get("/races/{date_code}", response_model=List[RaceSummary], tags=["Races"])
def get_races(date_code: str, repository: RaceRepository = Depends(get_repository) ) -> List[Dict[str, Any]]:
    return repository.get_races_by_date(date_code)

@app.get("/races/{race_id}/participants", response_model=List[ParticipantSummary], tags=["Races"])
def get_race_participants(race_id: int, repository: RaceRepository = Depends(get_repository)) -> List[Dict[str, Any]]:
    return repository.get_participants_by_race(race_id)

@app.get("/bets/sniper/{date_code}", response_model=List[BetRecommendation], tags=["Betting"])
def get_sniper_bets(
    date_code: str, 
    algo: Optional[str] = "auto",
    repository: RaceRepository = Depends(get_repository)
) -> List[Dict[str, Any]]:
    predictor = ml_models.get("predictor")
    if predictor is None: raise HTTPException(status_code=503, detail="ML Model unavailable.")

    raw_participants = repository.get_daily_data_for_ml(date_code)
    if not raw_participants: return []

    df_p = pd.DataFrame(raw_participants)
    
    probabilities = []
    place_probabilities = []
    model_versions = []

    # Process race by race to handle "auto" selection per discipline
    for race_id, race_group in df_p.groupby('race_id'):
        selected_algo = algo
        if algo == "auto":
            month = pd.to_datetime(race_group['program_date']).iloc[0].month
            discipline = race_group['discipline'].iloc[0]
            selected_algo = repository.get_best_model_for_context(discipline, month)

        try:
            preds, ver = predictor.predict_race(race_group.to_dict('records'), force_algo=selected_algo)
            probabilities.extend(preds["win"])
            place_probabilities.extend(preds["place"])
            model_versions.extend([ver] * len(race_group))
        except Exception as e:
            logger.error(f"Prediction failed for race {race_id}: {e}")
            probabilities.extend([0.0] * len(race_group))
            place_probabilities.extend([0.0] * len(race_group))
            model_versions.extend(["error"] * len(race_group))

    df = df_p.copy()
    df['win_probability'] = probabilities
    df['place_probability'] = place_probabilities
    df['model_version'] = model_versions

    # Persistence
    try:
        preds_to_save = [
            {
                "participant_id": row["participant_id"],
                "model_version": row["model_version"],
                "proba_winner": row["win_probability"],
                "proba_top3_place": row["place_probability"]
            }
            for _, row in df.iterrows()
        ]
        repository.upsert_predictions(preds_to_save)
    except Exception as e:
        logger.error(f"Failed to persist predictions: {e}")

    df['effective_odds'] = df['live_odds'].combine_first(df.get('live_odds_30mn', pd.Series([None] * len(df))))
    df['effective_odds'] = df['effective_odds'].apply(lambda x: x if x and x > 1.1 else None)
    df['effective_odds'] = df['effective_odds'].fillna(df['reference_odds']).fillna(1.0).clip(lower=1.05)
    df['is_live'] = df['live_odds'].apply(lambda x: True if x and x > 1.1 else False)

    if 'reference_odds' in df.columns:
        df['implied_prob'] = 1.0 / df['reference_odds'].replace(0, np.nan).fillna(100).clip(lower=1.01)
        race_total_prob = df.groupby('race_id')['implied_prob'].transform("sum")
        df['market_sentiment'] = df['implied_prob'] / (race_total_prob + 1e-6)
        df['odds_drop'] = (df['reference_odds'] - df['effective_odds']) / df['reference_odds'].replace(0, np.nan)
        df['market_signal'] = df['odds_drop'].apply(lambda x: x if x > 0 else 0)

    raw_edge = df['win_probability'] - (1 / df['effective_odds'])
    df['edge'] = raw_edge + (df.get('market_signal', 0) * 0.15)

    recommendations = []
    try:
        sniper_mask = (df['edge'] >= MIN_EDGE) & (df['effective_odds'] >= MIN_ODDS) & (df['effective_odds'] <= MAX_ODDS)
        sniper_df = df[sniper_mask]
        seen_bets = set()
        for race_id, group in sniper_df.groupby('race_id'):
            best_bet = group.sort_values('win_probability', ascending=False).iloc[0]
            recommendations.append({
                "race_id": int(best_bet['race_id']), "meeting_num": int(best_bet['meeting_number']),
                "race_num": int(best_bet['race_number']), "horse_name": best_bet['horse_name'],
                "program_number": int(best_bet['program_number']),
                "odds": float(best_bet['effective_odds']),
                "win_probability": float(best_bet['win_probability']),
                "edge": float(best_bet['edge']) + float(best_bet.get('market_signal', 0) * 0.1),
                "strategy": "Sniper" + (" (LIVE)" if best_bet['is_live'] else ""),
                "model_version": best_bet['model_version']
            })
            seen_bets.add((int(best_bet['race_id']), int(best_bet['program_number'])))
    except Exception as e: logger.error(f"Sniper strategy failed: {e}", exc_info=True)

    try:
        df_kelly = df[df['effective_odds'] < 100.0].copy()
        kelly_report = analyze_multiple_races(df_kelly, bankroll=1000.0, kelly_fraction=0.5)
        for course_info in kelly_report['ranking'][:5]:
            race_id = course_info['race_id']
            race_participants = df_kelly[df_kelly['race_id'] == race_id]
            if race_participants.empty: continue
            for prog_num, stake_frac in course_info['fractions'].items():
                if stake_frac > 0.005:
                    p_match = race_participants[race_participants['program_number'].astype(str) == str(prog_num)]
                    if not p_match.empty:
                        p_info = p_match.iloc[0]
                        if (int(p_info['race_id']), int(p_info['program_number'])) in seen_bets: continue
                        recommendations.append({
                            "race_id": int(p_info['race_id']), "meeting_num": int(p_info['meeting_number']),
                            "race_num": int(p_info['race_number']), "horse_name": p_info['horse_name'],
                            "program_number": int(p_info['program_number']), "odds": float(p_info['effective_odds']),
                            "win_probability": float(p_info['win_probability']), 
                            "edge": float(p_info['edge']),
                            "strategy": f"Kelly ({stake_frac:.1%})" + (" (LIVE)" if p_info.get('is_live') else ""),
                            "model_version": p_info['model_version']
                        })
    except Exception as e: logger.error(f"Kelly strategy failed: {e}", exc_info=True)

    return recommendations

@app.get("/races/{race_id}/predict", response_model=List[PredictionResult], tags=["Predictions"])
def predict_race(
    race_id: int, 
    algo: Optional[str] = "auto",
    repository: RaceRepository = Depends(get_repository)
) -> List[Dict[str, Any]]:
    predictor = ml_models.get("predictor")
    if predictor is None: raise HTTPException(status_code=503, detail="ML Model unavailable.")
    raw_participants = repository.get_race_data_for_ml(race_id)
    if not raw_participants: raise HTTPException(status_code=404, detail="Race not found.")

    selected_algo = algo
    if algo == "auto":
        discipline = raw_participants[0]['discipline']
        month = pd.to_datetime(raw_participants[0]['program_date']).month
        selected_algo = repository.get_best_model_for_context(discipline, month)
        logger.info(f"Auto-selected algo: {selected_algo} for {discipline} (month {month})")

    results = []
    try:
        preds_dict, model_version = predictor.predict_race(raw_participants, force_algo=selected_algo)
        win_probs = preds_dict["win"]
        place_probs = preds_dict["place"]
        
        preds_to_save = []
        for index, participant in enumerate(raw_participants):
            results.append({
                "program_number": participant["program_number"], "horse_name": participant["horse_name"],
                "win_probability": win_probs[index], "predicted_rank": 0,
                "model_version": model_version
            })
            preds_to_save.append({
                "participant_id": participant["participant_id"],
                "model_version": model_version,
                "proba_winner": win_probs[index],
                "proba_top3_place": place_probs[index]
            })

        repository.upsert_predictions(preds_to_save)
        results.sort(key=lambda x: x["win_probability"], reverse=True)
        for rank, res in enumerate(results, 1): res["predicted_rank"] = rank
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
    return results

@app.get("/backtest", tags=["ML Metrics"])
def run_backtest(
    force: bool = False,
    repository: RaceRepository = Depends(get_repository)
):
    try:
        service = BacktestService(repository)
        return service.run_backtest(force_update=force)
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Backtest failure.")
