import logging
import asyncio
from datetime import datetime, timedelta, timezone
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger

from src.cli.etl import etl_daily, etl_liveodds
from src.cli.daily_performance_etl import run_daily_performance_etl
from src.cli.telegram_bot import send_telegram_message
from src.api.repositories import RaceRepository
from src.api.backtest_service import BacktestService
from src.ml.predictor import RacePredictor

logger = logging.getLogger("Scheduler")

# Global scheduler instance to allow querying status from API
_scheduler = None


def get_scheduler():
    return _scheduler

async def generate_and_send_advice(race_id):
    """Génère un conseil de jeu basé sur le meilleur modèle et l'envoie via Telegram."""
    try:
        repo = RaceRepository()
        # 1. Récupérer les données de la course
        participants = repo.get_race_data_for_ml(race_id)
        if not participants:
            logger.warning(f"Aucun participant trouvé pour la course {race_id}")
            return

        discipline = participants[0]['discipline']
        month = datetime.fromtimestamp(participants[0]['start_timestamp'] / 1000, tz=timezone.utc).month

        # 2. Identifier le meilleur modèle (algo)
        selected_algo = repo.get_best_model_for_context(discipline, month)
        if not selected_algo:
             # Fallback sur le meilleur modèle global du cache si possible
             service = BacktestService(repo)
             cache = service.run_backtest() # Charge le cache
             trainers = cache.get("trainers", {})
             best_roi = -999
             for model_name, stats in trainers.items():
                 if discipline.lower() in model_name.lower():
                     roi = stats.get("roi", -100)
                     if roi > best_roi:
                         best_roi = roi
                         selected_algo = model_name

        # 3. Prédire
        from pathlib import Path
        import os
        current_file = Path(__file__).resolve()
        possible_data_path = current_file.parents[3] / "data"
        if not possible_data_path.exists(): possible_data_path = Path("data")
        model_path = Path(os.getenv("MODEL_PATH", possible_data_path))

        predictor = RacePredictor(str(model_path))
        preds, model_version = predictor.predict_race(participants, force_algo=selected_algo)

        if not preds["win"]:
            logger.warning(f"Prédictions vides pour la course {race_id}")
            return

        # 4. Sélectionner le meilleur cheval
        import pandas as pd
        import numpy as np
        df = pd.DataFrame(participants)
        # Ensure 'pmu_number' is available, it might be 'program_number' in some contexts
        if 'pmu_number' not in df.columns and 'program_number' in df.columns:
            df['pmu_number'] = df['program_number']

        df['win_probability'] = preds["win"]

        # Calcul de l'edge (besoin des cotes live ou ref)
        df['effective_odds'] = df['live_odds'].combine_first(df.get('live_odds_30mn', pd.Series([None] * len(df))))
        df['effective_odds'] = df['effective_odds'].apply(lambda x: x if x and x > 1.1 else None)
        df['effective_odds'] = df['effective_odds'].fillna(df['reference_odds']).fillna(1.0).clip(lower=1.05)

        df['edge'] = df['win_probability'] - (1 / df['effective_odds'])

        best_horse = df.sort_values('win_probability', ascending=False).iloc[0]

        # 5. Formater le message
        race_info = repo.get_races_by_date(datetime.fromtimestamp(participants[0]['start_timestamp']/1000).strftime("%d%m%Y"))
        race_row = next((r for r in race_info if r['race_id'] == race_id), None)

        meeting_label = race_row['meeting_libelle'] if race_row else "Inconnu"
        r_num = race_row['race_number'] if race_row else "?"
        m_num = race_row['meeting_number'] if race_row else "?"

        message = (
            f"<b>🏇 CONSEIL DE JEU - R{m_num}C{r_num}</b>\n"
            f"📍 {meeting_label} - {discipline}\n\n"
            f"🏆 <b>{best_horse['horse_name']}</b> (#{best_horse['pmu_number']})\n"
            f"📊 Probabilité: {best_horse['win_probability']:.1%}\n"
            f"💰 Cote estimée: {best_horse['effective_odds']:.2f}\n"
            f"📈 Edge: {best_horse['edge']:.2f}\n\n"
            f"🤖 Modèle: <code>{model_version}</code>"
        )

        # 6. Sauvegarder en BDD
        advice_data = {
            "race_id": race_id,
            "participant_id": int(best_horse['participant_id']),
            "model_version": model_version,
            "strategy": "BestModel",
            "message": message
        }
        repo.upsert_game_advice(advice_data)

        # 7. Envoyer Telegram
        await send_telegram_message(message)

    except Exception as e:
        logger.error(f"Erreur lors de la génération du conseil pour {race_id}: {e}", exc_info=True)

def run_send_telegram(race_id):
    """Wrapper synchrone pour exécuter la tâche de conseil Telegram."""
    try:
        asyncio.run(generate_and_send_advice(race_id))
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du job Telegram pour {race_id}: {e}")

def schedule_race_better(race_data_list):
    """
    Planifie dynamiquement les relevés de cotes pour une liste de courses.
    race_data_list: Liste de dict contenant {'id': str, 'start_time': int (ms timestamp)}
    """
    global _scheduler
    if _scheduler is None:
        return

    # Offsets demandés : -6h, -1h, -30m, -5m, 0m, +1m
    offsets = [
        # timedelta(hours=-6),
        # timedelta(hours=-1),
        # timedelta(minutes=-30),
        timedelta(minutes=-15), # Début de la cristallisation des enjeux
        timedelta(minutes=-5),  # Sentiment final / "Smart Money"
        # timedelta(minutes=2)  # Vérification post-départ (clôture des masses)
    ]

    for race in race_data_list:
        race_id = race['id']
        start_ts = race['start_time']
        if not start_ts:
            continue

        # Convert PMU ms timestamp to UTC datetime
        start_time = datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc)

        for offset in offsets:
            run_time = start_time + offset
            # On ne planifie que si l'heure est dans le futur
            if run_time > datetime.now(timezone.utc):
                job_id = f"better_{race_id}_{int(offset.total_seconds())}"
                _scheduler.add_job(
                    run_send_telegram,
                    trigger=DateTrigger(run_date=run_time),
                    args=[race_id],
                    id=job_id,
                    name=f"Better {race_id} @ {offset}",
                    replace_existing=True
                )

def schedule_race_updates(race_data_list):
    """
    Planifie dynamiquement les relevés de cotes pour une liste de courses.
    race_data_list: Liste de dict contenant {'id': str, 'start_time': int (ms timestamp)}
    """
    global _scheduler
    if _scheduler is None:
        return

    # Offsets demandés : -6h, -1h, -30m, -5m, 0m, +1m
    offsets = [
        timedelta(hours=-6),
        timedelta(hours=-1),
        timedelta(minutes=-30),
        timedelta(minutes=-15),
        timedelta(minutes=5),
        timedelta(minutes=10)
    ]

    for race in race_data_list:
        race_id = race['id']
        start_ts = race['start_time']
        if not start_ts:
            continue

        # Convert PMU ms timestamp to UTC datetime
        start_time = datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc)

        for offset in offsets:
            run_time = start_time + offset
            # On ne planifie que si l'heure est dans le futur
            if run_time > datetime.now(timezone.utc):
                job_id = f"odds_{race_id}_{int(offset.total_seconds())}"
                _scheduler.add_job(
                    etl_liveodds,
                    trigger=DateTrigger(run_date=run_time),
                    args=[race_id],  # etl_liveodds accepte maintenant l'ID
                    id=job_id,
                    name=f"Live Odds {race_id} @ {offset}",
                    replace_existing=True
                )


def cronjobs():
    """
    Initializes and starts the background scheduler for ETL tasks.
    # tasks.scan.triggers=daily:6h30
    # tasks.scan.dayoffsets=-1,0,1
    # tasks.odds.triggers=-6h, -1h, -30m,-5m, 0m, +1m*
    # tasks.better.triggers=-25m, -15m
    # tasks.finish.triggers=+30m,+1h*
    """
    global _scheduler
    if _scheduler is not None:
        return

    _scheduler = BackgroundScheduler()
    scheduler = _scheduler

    # Define common dates for the daily ETL
    # Note: These are recalculated each time the job runs if we pass them as arguments,
    # but since etl_daily takes them as fixed strings, we wrap it.

    def daily_job_wrapper():
        # Ingestion pour hier, aujourd'hui et demain
        start_date = (datetime.today() - timedelta(days=1)).strftime("%d%m%Y")
        end_date = (datetime.today() + timedelta(days=1)).strftime("%d%m%Y")
        logger.info(f"Triggering daily ETL for range {start_date} to {end_date}")

        # 1. Exécuter l'ingestion de base
        races = etl_daily(start_date, end_date)

        # 2. Planifier les relevés de cotes spécifiques si des courses sont retournées
        if races:
            logger.info(f"Scheduling dynamic updates for {len(races)} races.")
            schedule_race_updates(races)
            schedule_race_better(races)

    # 1. Daily Ingestion at 06:30 AM
    scheduler.add_job(
        daily_job_wrapper,
        CronTrigger(hour=6, minute=30),
        id="daily_etl",
        name="Daily PMU Data Ingestion",
        replace_existing=True
    )

    # 2. Live Odds update every 15 minutes
    scheduler.add_job(
        etl_liveodds,
        IntervalTrigger(minutes=15),
        id="live_odds",
        name="Live Odds Update",
        replace_existing=True
    )

    # 3. Daily Performance figer à 01h15
    scheduler.add_job(
        run_daily_performance_etl,
        CronTrigger(hour=1, minute=15),
        id="daily_performance",
        name="Daily Performance Snapshot",
        replace_existing=True
    )

    scheduler.start()
    logger.info("Background Scheduler started. Jobs scheduled: Daily ETL (06:30) and Live Odds (15m).")
