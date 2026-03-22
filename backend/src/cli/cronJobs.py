import logging
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from backend.src.cli.etl import etl_daily, etl_liveodds

logger = logging.getLogger("Scheduler")

# Global scheduler instance to allow querying status from API
_scheduler = None

def get_scheduler():
    return _scheduler

def cronjobs():
    """
    Initializes and starts the background scheduler for ETL tasks.
    """
    global _scheduler
    if _scheduler is not None:
        return

    _scheduler = BackgroundScheduler()
    scheduler = _scheduler

    # Define common dates for the daily ETL
    def daily_job_wrapper():
        start_date = (datetime.today() - timedelta(days=2)).strftime("%d%m%Y")
        end_date = (datetime.today() + timedelta(days=1)).strftime("%d%m%Y")
        logger.info(f"Triggering daily ETL for range {start_date} to {end_date}")
        etl_daily(start_date, end_date)

    # 1. Daily Ingestion at 07:30 AM
    scheduler.add_job(
        daily_job_wrapper,
        CronTrigger(hour=7, minute=30),
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

    scheduler.start()
    logger.info("Background Scheduler started. Jobs scheduled: Daily ETL (07:30) and Live Odds (15m).")
