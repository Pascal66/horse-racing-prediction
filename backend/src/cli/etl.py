import argparse
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Any

# Import BaseIngestor for Type Hinting
from src.ingestion.base import BaseIngestor
from src.ingestion.program import ProgramIngestor
from src.ingestion.participants import ParticipantsIngestor
from src.ingestion.performances import PerformancesIngestor
from src.ingestion.rapports import ReportsIngestor

# Constants for date formatting to avoid magic strings
DATE_FORMAT_INPUT = "%d%m%Y"  # e.g., 05112025
DATE_FORMAT_LOG = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | ORCHESTRATOR | %(message)s",
    datefmt=DATE_FORMAT_LOG
)
logger = logging.getLogger("Orchestrator")


def generate_date_range(start_str: str, end_str: str) -> List[str]:
    """
    Generates a list of date strings between start and end inclusive.

    Args:
        start_str (str): Start date in DDMMYYYY format.
        end_str (str): End date in DDMMYYYY format.

    Returns:
        List[str]: A list of strings representing dates in DDMMYYYY format.

    Exits:
        sys.exit(1): If date format is invalid or start date is after end date.
    """
    try:
        start = datetime.strptime(start_str, DATE_FORMAT_INPUT)
        end = datetime.strptime(end_str, DATE_FORMAT_INPUT)
    except ValueError:
        logger.error(f"Invalid date format. Please use DDMMYYYY (e.g., 05112025).")
        sys.exit(1)

    if start > end:
        logger.error("Start date cannot be after end date.")
        sys.exit(1)

    date_list = []
    current = start
    while current <= end:
        date_list.append(current.strftime(DATE_FORMAT_INPUT))
        current += timedelta(days=1)

    return date_list


def process_date(date_code: str, ingestion_type: str, race_id: int = None) -> Any:
    """
    Executes the ingestion pipeline for a specific single date.

    Args:
        date_code (str): The date to process (DDMMYYYY).
        ingestion_type (str): The category of data to ingest
                              ('program', 'participants', 'performances', 'reports', 'all').
        race_id (int, optional): Specific race ID to ingest.
    """
    logger.info(f"=== Starting Ingestion for Date: {date_code} ===")

    # FIX FOR MYPY: Explicitly type the list as generic BaseIngestor
    # This prevents Mypy from assuming the list is ONLY for ProgramIngestor
    ingestors: List[BaseIngestor] = []

    # Conditional logic preserves the original behavior
    if ingestion_type in ["program", "all"]:
        ingestors.append(ProgramIngestor(date_code))

    if ingestion_type in ["participants", "all"]:
        ingestors.append(ParticipantsIngestor(date_code))

    if ingestion_type in ["performances", "all"]:
        ingestors.append(PerformancesIngestor(date_code))

    if ingestion_type in ["reports", "all"]:
        ingestors.append(ReportsIngestor(date_code))

    results = {}
    for ingestor in ingestors:
        ingestor_name = ingestor.__class__.__name__
        try:
            # Pass race_id to all ingestors that support it
            if race_id and not isinstance(ingestor, ProgramIngestor):
                res = ingestor.ingest(race_id=race_id)
            else:
                res = ingestor.ingest()
            results[ingestor_name] = res
        except Exception as e:
            # Capturing the class name for clearer error logging
            logger.exception(f"Ingestion failed for {ingestor_name} on {date_code}: {e}")
            results[ingestor_name] = None

    # Priority return: if ProgramIngestor was run, return its list of races
    if "ProgramIngestor" in results and results["ProgramIngestor"]:
        return results["ProgramIngestor"]

    # Otherwise return the last result in the dictionary
    return list(results.values())[-1] if results else None


def main() -> None:
    """
    Main entry point for the CLI Orchestrator.
    """
    parser = argparse.ArgumentParser(description="PMU Ingestion Orchestrator (ETL CLI)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--date",
        help="Single date code (e.g., 05112025)"
    )
    group.add_argument(
        "--range",
        nargs=2,
        metavar=('START', 'END'),
        help="Date range (e.g., 01012025 05012025)"
    )

    parser.add_argument(
        "--type",
        required=True,
        choices=["program", "participants", "performances", "reports", "all"],
        help="Type of data to ingest"
    )

    args = parser.parse_args()

    dates_to_process: List[str] = []

    if args.date:
        dates_to_process = [args.date]
    elif args.range:
        start_date, end_date = args.range
        dates_to_process = generate_date_range(start_date, end_date)

    total_days = len(dates_to_process)
    logger.info(f"Job started. Processing {total_days} day(s). Mode: {args.type}")

    # Enumerate starting at 1 for user-friendly progress logging
    for i, date_code in enumerate(dates_to_process, 1):
        logger.info(f"Progress: [{i}/{total_days}] Processing {date_code}")
        process_date(date_code, args.type)

    logger.info("All jobs completed.")


def etl_daily(start_date, end_date) -> List[dict]:
    dates_to_process = generate_date_range(start_date, end_date)
    total_days = len(dates_to_process)
    logger.info(f"Job started. Processing {total_days} day(s). Mode: all")

    all_races = []
    # Enumerate starting at 1 for user-friendly progress logging
    for i, date_code in enumerate(dates_to_process, 1):
        logger.info(f"Progress: [{i}/{total_days}] Processing {date_code}")
        # Note: If multiple dates are processed, all_races will contain
        # the races of the LAST date if "all" is used and ProgramIngestor is last.
        # But in daily_job_wrapper, start_date/end_date usually cover only today/tomorrow.
        res = process_date(date_code, "all")
        if isinstance(res, list):
            all_races.extend(res)

    logger.info("All jobs completed.")
    return all_races


def etl_liveodds(race_id: int = None, start_hour=7, end_hour=21) -> None:
    NOW = datetime.now()
    TODAY = datetime.today()

    # If a specific race_id is provided, we ignore the hours restriction
    if race_id is None:
        FROM = NOW.replace(hour=start_hour, minute=0)
        TO = NOW.replace(hour=end_hour, minute=30)
        if NOW < FROM or NOW > TO:
            return

    date_code = TODAY.strftime("%d%m%Y")

    # Use "participants" ingestion type for regular updates
    # if it's the 15m interval task (race_id is None),
    # it will update all races of today.
    # If race_id is provided, it updates only that race.
    process_date(date_code, "participants", race_id=race_id)


if __name__ == "__main__":
    main()
