import logging
import datetime
from typing import Optional
from src.api.repositories import RaceRepository
from src.api.backtest_service import BacktestService

logger = logging.getLogger("DailyPerformanceETL")

def run_daily_performance_etl(target_date: Optional[datetime.date] = None):
    """
    Calcule et fige les performances de la journée passée.
    Audit les prédictions manquantes.
    """
    if target_date is None:
        target_date = datetime.date.today() - datetime.timedelta(days=1)

    logger.info(f"Starting Daily Performance ETL for {target_date}")

    repo = RaceRepository()
    service = BacktestService(repo)

    # 1. Calcul des stats
    results = service.get_period_stats(target_date, target_date)

    trainers = results.get("trainers", {})
    audit = results.get("audit", {})

    # 2. Audit
    missing_races = audit.get("missing_races", [])
    if missing_races:
        logger.warning(f"AUDIT: {len(missing_races)} races are finished but MISSING predictions: {missing_races}")
    else:
        logger.info("AUDIT: No missing predictions for finished races.")

    # 3. Préparation des données pour la table ml_daily_performance
    performances_to_save = []

    for model_version, data in trainers.items():
        for discipline, disc_stats in data.get("disciplines", {}).items():
            for bet_type, stats in disc_stats.items():
                performances_to_save.append({
                    "performance_date": target_date,
                    "model_version": model_version,
                    "discipline": discipline,
                    "bet_type": bet_type,
                    "nb_bets": stats["nb_bets"],
                    "nb_wins": stats["nb_wins"],
                    "roi": stats["roi"],
                    "avg_odds": stats["avg_odds"]
                })

    if performances_to_save:
        logger.info(f"Saving {len(performances_to_save)} performance records to database.")
        success = repo.upsert_daily_performance(performances_to_save)
        if success:
            logger.info("Daily performance ETL completed successfully.")
        else:
            logger.error("Failed to save daily performance records.")
    else:
        logger.warning(f"No performance data found for {target_date}.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_daily_performance_etl()
