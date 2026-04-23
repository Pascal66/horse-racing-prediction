import os
import logging

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, Engine
from src.core.config import DB_URL

class DataLoader:
    """
    Robust Data Manager using SQLAlchemy.
    Responsible for loading raw data (Participants + History) and performing the initial merge.
    """

    def __init__(self) -> None:
        """
        Initializes the database connection using environment variables.
        """

        self.logger = logging.getLogger("ML.Loader")
        if not DB_URL:
            raise ValueError("DB_URL is missing in the .env file")
        self.engine: Engine = create_engine(DB_URL)

    def get_training_data(self) -> pd.DataFrame:
        self.logger.info("Loading SQL data...")
        try:
            with self.engine.connect() as connection:
                # 1. Query principale enrichie avec les stats de carrière détaillées
                query_main = """
                SELECT
                    rp.participant_id, rp.race_id, rp.horse_id,
                    rp.finish_rank,
                    CASE WHEN rp.finish_rank = 1 THEN 1 ELSE 0 END AS is_winner,
                    dp.program_date, rm.meeting_code, rm.meeting_type, rm.audience,
                    r.discipline, r.distance_m, r.track_type, r.terrain_label,
                    r.declared_runners_count, r.penetrometer,
                    h.birth_year, h.sex,
                    rp.pmu_number, rp.age, 
                    rp.career_winnings, rp.career_races_count,
                    rp.career_wins_count, rp.career_places_count, 
                    rp.career_places_2nd_count, rp.career_places_3rd_count,
                    rp.winnings_victory, rp.winnings_place, 
                    rp.winnings_year_now, rp.winnings_year_prev,
                    rp.reference_odds, rp.live_odds,
                    ls.code AS shoeing_status,
                    j.actor_name AS jockey_name,
                    t.actor_name AS trainer_name,
                    rp.driver_jockey_id -- Gardé temporairement pour le merge duo
                FROM race_participant rp
                JOIN race r ON rp.race_id = r.race_id
                JOIN race_meeting rm ON r.meeting_id = rm.meeting_id
                JOIN daily_program dp ON rm.program_id = dp.program_id
                JOIN horse h ON rp.horse_id = h.horse_id
                LEFT JOIN lookup_shoeing ls ON rp.shoeing_id = ls.shoeing_id
                LEFT JOIN racing_actor j ON rp.driver_jockey_id = j.actor_id
                LEFT JOIN racing_actor t ON rp.trainer_id = t.actor_id
                WHERE rp.finish_rank IS NOT NULL
                -- Filtrer sur les courses pariables
                AND rm.audience IN ('NATIONAL')
                --, 'REGIONAL')  
                -- ou uniquement NATIONAL si les régionales ne sont pas disponibles
                """
                main_df = pd.read_sql(query_main, connection)
                main_df['program_date'] = pd.to_datetime(main_df['program_date'])

                # 2. Historique enrichi avec temporalité
                self.logger.info("Calculating historical statistics...")
                # query_history = """
                # SELECT horse_id, race_date, discipline, distance_m, finish_place, finish_status, reduction_km, prize_money,
                #        ROW_NUMBER() OVER (PARTITION BY horse_id ORDER BY race_date DESC) AS race_recency_rank
                # FROM horse_race_history
                # """
                query_history = """
                WITH ranked_history AS (
                    SELECT
                        hrh.horse_id,
                        hrh.race_date,
                        hrh.discipline,
                        hrh.distance_m,
                        hrh.finish_place,
                        hrh.finish_status,
                        hrh.reduction_km,
                        hrh.prize_money,
                        -- Utilisation de l'audience officielle (NATIONAL, REGIONAL, LOCAL)
                        -- Priorité à rm.audience, fallback sur prize_money pour l'historique externe
                        COALESCE(rm.audience, 
                            CASE 
                                WHEN hrh.prize_money >= 15000 THEN 'NATIONAL'
                                WHEN hrh.prize_money >= 4000  THEN 'REGIONAL'
                                ELSE 'LOCAL'
                            END
                        ) AS estimated_level,
                        ROW_NUMBER() OVER (
                            PARTITION BY hrh.horse_id ORDER BY hrh.race_date DESC
                        ) AS race_recency_rank
                    FROM horse_race_history hrh
                    -- Jointure via Date + Discipline car race_id est absent de l'historique
                    LEFT JOIN (
                        SELECT DISTINCT dp2.program_date, r2.discipline, rm2.audience
                        FROM race r2
                        JOIN race_meeting rm2 ON r2.meeting_id = rm2.meeting_id
                        JOIN daily_program dp2 ON rm2.program_id = dp2.program_id
                    ) rm ON hrh.race_date = rm.program_date AND hrh.discipline = rm.discipline
                )
                SELECT * FROM ranked_history
                """
                history_df = pd.read_sql(query_history, connection)
                # debug seuils
                # 0.33    15500.0
                # 0.66    23000.0
                # 0.90    46000.0
                # print(history_df['prize_money'].describe())
                # print(history_df['prize_money'].quantile([0.33, 0.66, 0.90]))

                # 3. Duo Jockey/Cheval
                query_jockey_horse = """
                WITH jh_history AS (
                    SELECT rp.horse_id, rp.driver_jockey_id, dp.program_date,
                           CASE WHEN rp.finish_rank = 1 THEN 1 ELSE 0 END AS is_winner,
                           rp.finish_rank
                    FROM race_participant rp
                    JOIN race r ON rp.race_id = r.race_id
                    JOIN race_meeting rm ON r.meeting_id = rm.meeting_id
                    JOIN daily_program dp ON rm.program_id = dp.program_id
                    WHERE rp.finish_rank IS NOT NULL
                )
                SELECT cur.horse_id, cur.driver_jockey_id, cur.program_date,
                       COUNT(hist.is_winner) AS duo_total_races,
                       COALESCE(SUM(hist.is_winner), 0) AS duo_wins,
                       COALESCE(AVG(hist.is_winner), 0) AS duo_win_rate,
                       COALESCE(AVG(hist.finish_rank), 99) AS duo_avg_rank,
                       COALESCE(MIN(hist.finish_rank), 99) AS duo_best_rank
                FROM jh_history cur
                LEFT JOIN jh_history hist ON hist.horse_id = cur.horse_id 
                     AND hist.driver_jockey_id = cur.driver_jockey_id 
                     AND hist.program_date < cur.program_date
                GROUP BY cur.horse_id, cur.driver_jockey_id, cur.program_date
                """
                jh_df = pd.read_sql(query_jockey_horse, connection)
                jh_df['program_date'] = pd.to_datetime(jh_df['program_date'])

                # Merges
                horse_stats, discipline_aff = self._compute_horse_stats(history_df)
                final_df = pd.merge(main_df, horse_stats, on='horse_id', how='left')
                final_df = pd.merge(final_df, discipline_aff, on=['horse_id', 'discipline'], how='left')
                
                # Merge duo sur horse + jockey + DATE (plus robuste que race_id)
                final_df = pd.merge(final_df, jh_df, on=['horse_id', 'driver_jockey_id', 'program_date'], how='left')

                # Cleanup
                final_df = final_df.drop(columns=['driver_jockey_id'])
                #final_df.sort_values('program_date', inplace=True)
                # final_df.sort_values(['horse_id', 'program_date'], inplace=True)
                # for col in ['hist_avg_speed', 'career_winnings', 'duo_win_rate']:
                #     final_df[col] = final_df.groupby('horse_id')[col].shift(1).expanding().mean().values
                final_df = final_df.sort_values('program_date').reset_index(drop=True)
                return final_df

        except Exception as error:
            self.logger.error(f"Error loading data: {error}")
            raise

    def _compute_horse_stats(self, history_df: pd.DataFrame):
        history_df = history_df.copy()
        history_df['finish_status'] = history_df['finish_status'].replace(r'\N', None)
        bad_statuses = ['ARRETE', 'DEROBE', 'DISQUALIFIE', 'RESTE_AU_POTEAU', 'TOMBE', 'INCONNU']
        history_df['is_clean_run'] = (~history_df['finish_status'].str.upper().isin(bad_statuses)).astype(int)

        career = history_df.groupby('horse_id').agg(
            hist_avg_speed=('reduction_km', 'mean'),
            hist_earnings=('prize_money', 'sum'),
            hist_pct_clean_runs=('is_clean_run', 'mean'),
            hist_races=('horse_id', 'count')
        ).reset_index()

        recent_3 = history_df[history_df['race_recency_rank'] <= 3].groupby('horse_id').agg(
            avg_speed_last_3=('reduction_km', 'mean')
        ).reset_index()

        last_race = history_df[history_df['race_recency_rank'] == 1][['horse_id', 'race_date']].rename(columns={'race_date': 'last_race_date'})
        
        distance_stats = history_df.groupby('horse_id').agg(hist_avg_distance=('distance_m', 'mean')).reset_index()

        discipline_aff = history_df.groupby(['horse_id', 'discipline']).agg(
            n_races_disc=('is_clean_run', 'count'),
            pct_clean_on_discipline=('is_clean_run', 'mean')
        ).reset_index()
        total_r = history_df.groupby('horse_id').size().reset_index(name='total')
        discipline_aff = discipline_aff.merge(total_r, on='horse_id')
        discipline_aff['pct_races_on_discipline'] = discipline_aff['n_races_disc'] / discipline_aff['total']

        horse_stats = career.merge(recent_3, on='horse_id', how='left') \
                            .merge(last_race, on='horse_id', how='left') \
                            .merge(distance_stats, on='horse_id', how='left')

        # Stats séparées par niveau estimé
        national_hist = history_df[history_df['estimated_level'] == 'NATIONAL']
        national_stats = national_hist.groupby('horse_id').agg(
            hist_avg_speed_national=('reduction_km', 'mean'),
            n_races_national=('finish_place', 'count'),
            hist_win_rate_national=('finish_place', lambda x: (x == 1).mean()),
        ).reset_index()

        horse_stats = horse_stats.merge(national_stats, on='horse_id', how='left')

        # Feature de progression : ratio vitesse national vs globale
        horse_stats['national_speed_ratio'] = (
                horse_stats['hist_avg_speed_national'] /
                horse_stats['hist_avg_speed'].replace(0, np.nan)
        ).fillna(1.0)

        # Expérience au niveau national (0 = débutant, 1 = très expérimenté)
        horse_stats['national_experience_rate'] = (
                horse_stats['n_races_national'].fillna(0) /
                horse_stats['hist_races'].replace(0, 1)
        ).clip(0, 1)

        return horse_stats, discipline_aff[['horse_id', 'discipline', 'pct_races_on_discipline', 'pct_clean_on_discipline']]
