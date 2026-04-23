import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
import os
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from backend.src.api.repositories import RaceRepository
from backend.src.ml.predictor import RacePredictor
from backend.src.core.database import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("Backfill")


def backfill(date_start: str, date_end: str, model_dir: str):
    repo = RaceRepository()

    try:
        predictor = RacePredictor(model_dir)
    except Exception as e:
        logger.error(f"Failed to load predictor: {e}")
        return

    start_dt = datetime.strptime(date_start, "%d%m%Y")
    end_dt = datetime.strptime(date_end, "%d%m%Y")

    # current_dt = start_dt
    # while current_dt <= end_dt:
    #     date_code = current_dt.strftime("%d%m%Y")
    #     logger.info(f"Processing date: {date_code}")
    #
    #     participants = repo.get_daily_data_for_ml(date_code)
    #     if not participants:
    #         logger.info(f"No participants found for {date_code}")
    #     else:
    #         try:
    #             probabilities, model_version = predictor.predict_race(participants)
    #
    #             preds_to_save = []
    #             for idx, p in enumerate(participants):
    #                 preds_to_save.append({
    #                     "participant_id": p["participant_id"],
    #                     "model_version": model_version,
    #                     "proba_winner": probabilities[idx]
    #                 })
    #
    #             if repo.upsert_predictions(preds_to_save):
    #                 logger.info(
    #                     f"Successfully backfilled {len(preds_to_save)} predictions for {date_code} using model {model_version}")
    #             else:
    #                 logger.error(f"Failed to upsert predictions for {date_code}")
    #
    #         except Exception as e:
    #             logger.error(f"Error processing {date_code}: {e}")
    #
    #     current_dt += timedelta(days=1)


    algos = ["ltr", "hyperstack", "tabnet", "gpt"]
    current_dt = start_dt
    while current_dt <= end_dt:
        date_code = current_dt.strftime("%d%m%Y")
        month = current_dt.month
        participants = repo.get_daily_data_for_ml(date_code)

        if participants:
            df_p = pd.DataFrame(participants)
            for race_id, race_group in df_p.groupby('race_id'):
                discipline = race_group['discipline'].iloc[0]

                # Choix du meilleur algo pour ce contexte spécifique (Optionnel, on peut tout backfiller)
                best_algo = repo.get_best_model_for_context(discipline, month)

                for algo in algos:
                    try:
                        # Si on veut limiter au meilleur, on décommente :
                        # if algo != best_algo: continue

                        preds_dict, model_ver = predictor.predict_race(race_group.to_dict('records'), force_algo=algo)
                        if model_ver == "none": continue

                        preds_to_save = [{
                            "participant_id": p["participant_id"],
                            "model_version": f"{algo}_{model_ver}",
                            "proba_winner": preds_dict["win"][i],
                            "proba_top3_place": preds_dict["place"][i]
                        } for i, p in enumerate(race_group.to_dict('records'))]

                        repo.upsert_predictions(preds_to_save)
                    except Exception as e:
                        logger.error(f"Error {date_code} Race {race_id} Algo {algo}: {e}")

        current_dt += timedelta(days=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill predictions for historical data.")
    parser.add_argument("--start", required=True, help="Start date DDMMYYYY")
    parser.add_argument("--end", required=True, help="End date DDMMYYYY")
    parser.add_argument("--model-dir", default="data", help="Directory containing models")

    args = parser.parse_args()

    # Initialize DB pool
    db = DatabaseManager()
    db.initialize_pool()

    try:
        backfill(args.start, args.end, args.model_dir)
    finally:
        db.close_pool()
