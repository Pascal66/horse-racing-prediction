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

        # 1. Initialisation et récupération des stats ROI (Backtest)
        first = participants[0]
        discipline = first['discipline']
        start_ts = first.get('start_timestamp') or first.get('start_time')
        if not start_ts:
            logger.error(f"Impossible de générer le conseil pour {race_id} : timestamp absent.")
            return

        from pathlib import Path
        import os
        import pandas as pd
        model_path = os.getenv("MODEL_PATH", Path(__file__).resolve().parents[3] / "data")
        predictor = RacePredictor(str(model_path))
        
        service = BacktestService(repo)
        backtest = service.run_backtest()
        trainer_stats = backtest.get("trainers", {})

        # 2. Générer les prédictions pour tous les algos (Logique "Pronostics Comparés")
        model_results = []
        for algo in ["tabnet", "ltr", "hyperstack", "gpt"]:
            preds, m_ver = predictor.predict_race(participants, force_algo=algo)
            if not preds or not preds.get("win"): continue

            # Matching du ROI (réplication de find_best_roi_match du frontend)
            roi = -99.0
            if m_ver in trainer_stats:
                roi = trainer_stats[m_ver].get("roi", -99.0)
            else:
                m_parts = set(m_ver.lower().split('_'))
                for k, v in trainer_stats.items():
                    k_parts = set(k.lower().split('_'))
                    if m_parts.issubset(k_parts) or k_parts.issubset(m_parts):
                        roi = v.get("roi", -99.0)
                        break
            
            df = pd.DataFrame(participants)
            df['pmu_number'] = df.get('pmu_number', df.get('program_number'))
            df['win_probability'] = preds["win"]
            # Calcul de la cote effective (live ou ref)
            df['eff_odds'] = df['live_odds'].combine_first(df.get('live_odds_30mn', pd.Series([None]*len(df))))
            df['eff_odds'] = df['eff_odds'].fillna(df['reference_odds']).fillna(1.0).clip(lower=1.05)
            
            # Récupération du Top 3 pour ce modèle
            top_3_df = df.sort_values('win_probability', ascending=False).head(3)
            model_results.append({
                "algo": algo, 
                "roi": roi, 
                "top_3": top_3_df.to_dict('records'), 
                "version": m_ver
            })

        if not model_results: return
        
        # 3. Trier par ROI pour identifier l'expert du moment
        model_results.sort(key=lambda x: x["roi"], reverse=True)

        # 4. Formater le message (Top 3 experts)
        race_date_str = datetime.fromtimestamp(start_ts / 1000).strftime("%d%m%Y")
        race_hour_str = datetime.fromtimestamp(start_ts / 1000).strftime("%H:%M")
        race_info = repo.get_races_by_date(race_date_str)
        race_row = next((r for r in race_info if r['race_id'] == race_id), None)
        m_num = race_row['meeting_number'] if race_row else "?"
        r_num = race_row['race_number'] if race_row else "?"
        meeting_label = race_row['meeting_libelle'] if race_row else "Inconnu"

        message = f"<b>R{m_num}C{r_num}</b> {discipline} {race_hour_str}\n\n"
        
        for idx, res in enumerate(model_results[:3], 1):
            emoji = ["1️⃣", "2️⃣", "3️⃣"][idx-1]
            color = "🟢" if res['roi'] > 0 else ("⚪" if res['roi'] == -99.0 else "🔴")
            # Ligne de titre concise : Emoji + Nom Algo + Pastille ROI
            message += f"{emoji} <b>{res['algo'].upper()}</b> {color}\n"
            
            # Ligne de pronostics : Numéro (Prob%) séparés par des points médians
            # preds_str = []
            odds_str = []
            for h in res['top_3']:
                num = h.get('pmu_number') or h.get('program_number')
                # preds_str.append(f"#{num} ({h['win_probability']:.1%})")
                odds_str.append(f"#{num}<b> ({h['eff_odds']:.2f})</b>")
            
            message += " · ".join(odds_str) + "\n"

        # 5. Sauvegarder le meilleur conseil en BDD
        top = model_results[0]
        repo.upsert_game_advice({
            "race_id": race_id,
            "participant_id": int(top['top_3'][0]['participant_id']),
            "model_version": top['version'],
            "strategy": "ConsensusROI",
            "message": message
        })

        # 6. Envoyer via Telegram
        await send_telegram_message(message)

    except Exception as e:
        logger.error(f"Erreur lors de la génération du conseil pour {race_id}: {e}", exc_info=True)


def etl_liveodds_and_update_cache(race_id: int = None):
    """
    Ingest data and update the backtest cache.
    """
    try:
        # 1. Ingest results and dividends
        etl_liveodds(race_id=race_id)

        # 2. Update the cache file
        repo = RaceRepository()
        service = BacktestService(repo)
        service.update_today_etl()
        logger.info(f"Cache updated after ingestion for race {race_id if race_id else 'global'}")
    except Exception as e:
        logger.error(f"Failed to update cache after ingestion: {e}")


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
        timedelta(minutes=-16),  # Début de la cristallisation des enjeux
        timedelta(minutes=-6),  # Sentiment final / "Smart Money"
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

    # Offsets demandés : -6h, -1h, -30m, -15m, 5m, 10m, 20m, 40m, 60m
    offsets = [
        timedelta(hours=-6),
        timedelta(hours=-1),
        timedelta(minutes=-30),
        timedelta(minutes=-15),
        timedelta(minutes=5),
        timedelta(minutes=10),
        timedelta(minutes=20),
        timedelta(minutes=40),
        timedelta(minutes=60)
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
                # Use wrapper to update cache after ingestion
                target_func = etl_liveodds_and_update_cache if offset.total_seconds() > 0 else etl_liveodds

                _scheduler.add_job(
                    target_func,
                    trigger=DateTrigger(run_date=run_time),
                    args=[race_id],
                    id=job_id,
                    name=f"Update {race_id} @ {offset}",
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

        # 3. Refresh the backtest cache (global 12-month stats)
        try:
            repo = RaceRepository()
            service = BacktestService(repo)
            service.run_backtest(force_update=True)
            logger.info("Global backtest cache refreshed after daily ETL.")
        except Exception as e:
            logger.error(f"Failed to refresh global backtest cache: {e}")

    # 1. Daily Ingestion at 06:30 AM
    scheduler.add_job(
        daily_job_wrapper,
        CronTrigger(hour=6, minute=30),
        id="daily_etl",
        name="Daily PMU Data Ingestion",
        replace_existing=True
    )

    # 2. Live Odds & Cache update every 15 minutes
    scheduler.add_job(
        etl_liveodds_and_update_cache,
        IntervalTrigger(minutes=15),
        id="live_odds",
        name="Live Odds & Cache Update",
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
