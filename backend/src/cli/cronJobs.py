import logging
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.cli.etl import etl_daily, etl_liveodds

logger = logging.getLogger("Scheduler")

def cronjobs():
    """
    Initializes and starts the background scheduler for ETL tasks.
    # tasks.scan.triggers=daily:7h30
    # tasks.scan.dayoffsets=-1,0,1
    # tasks.odds.triggers=-6h,-15m,-3m,0m,+1m*
    # tasks.better.triggers=-5m
    # tasks.finish.triggers=+30m,+1h*
    """
    scheduler = BackgroundScheduler()

    # Define common dates for the daily ETL
    # Note: These are recalculated each time the job runs if we pass them as arguments,
    # but since etl_daily takes them as fixed strings, we wrap it.

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
