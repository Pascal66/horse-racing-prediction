import logging
from datetime import datetime, timedelta, timezone
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger

from src.cli.etl import etl_daily, etl_liveodds
from src.cli.daily_performance_etl import run_daily_performance_etl

logger = logging.getLogger("Scheduler")

# Global scheduler instance to allow querying status from API
_scheduler = None


def get_scheduler():
    return _scheduler


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
        timedelta(minutes=-5),
        timedelta(minutes=0),
        timedelta(minutes=5)
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
