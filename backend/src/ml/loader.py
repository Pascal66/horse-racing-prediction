import os
import logging
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
                # 1. Query principale — ajout penetrometer et prize_money
                # ajout windspeed et meeting_type (DIURNE, SEMINOCTURNE, NOCTURNE)
                query_main = """
                SELECT
                    rp.participant_id, rp.race_id, rp.horse_id,
                    rp.finish_rank, rp.driver_jockey_id,
                    CASE WHEN rp.finish_rank = 1 THEN 1 ELSE 0 END AS is_winner,
                    dp.program_date, rm.racetrack_code, rm.weather_temperature, rm.weather_wind, rm.weather_windspeed,
                    rm.meeting_type,
                    r.race_number, r.discipline, r.distance_m, r.track_type, r.terrain_label,
                    r.declared_runners_count, r.penetrometer, r.prize_money AS race_prize_money,
                    h.birth_year, h.sex,
                    rp.pmu_number, rp.age, rp.career_winnings, rp.career_races_count, rp.trainer_advice,
                    rp.reference_odds, rp.live_odds,
                    ls.code AS shoeing_status,
                    j.actor_name AS jockey_name,
                    t.actor_name AS trainer_name
                FROM race_participant rp
                JOIN race r ON rp.race_id = r.race_id
                JOIN race_meeting rm ON r.meeting_id = rm.meeting_id
                JOIN daily_program dp ON rm.program_id = dp.program_id
                JOIN horse h ON rp.horse_id = h.horse_id
                LEFT JOIN lookup_shoeing ls ON rp.shoeing_id = ls.shoeing_id
                LEFT JOIN racing_actor j ON rp.driver_jockey_id = j.actor_id
                LEFT JOIN racing_actor t ON rp.trainer_id = t.actor_id
                WHERE rp.finish_rank IS NOT NULL
                """
                main_df = pd.read_sql(query_main, connection)
                main_df['program_date'] = pd.to_datetime(main_df['program_date'])

                # 2. Historique enrichi avec temporalité
                self.logger.info("Calculating historical statistics...")
                query_history = """
                WITH ranked_history AS (
                    SELECT
                        horse_id,
                        race_date,
                        discipline,
                        distance_m,
                        finish_place,
                        finish_status,
                        reduction_km,
                        prize_money,
                        ROW_NUMBER() OVER (
                            PARTITION BY horse_id ORDER BY race_date DESC
                        ) AS race_recency_rank
                    FROM horse_race_history
                )
                SELECT * FROM ranked_history
                """
                history_df = pd.read_sql(query_history, connection)
                query_jockey_horse = """
                WITH jockey_horse_history AS (
                    SELECT
                        rp.horse_id,
                        rp.driver_jockey_id,
                        rp.race_id,
                        dp.program_date,
                        CASE WHEN rp.finish_rank = 1 THEN 1 ELSE 0 END AS is_winner,
                        rp.finish_rank,
                        rp.reference_odds
                    FROM race_participant rp
                    JOIN race r          ON rp.race_id     = r.race_id
                    JOIN race_meeting rm ON r.meeting_id   = rm.meeting_id
                    JOIN daily_program dp ON rm.program_id = dp.program_id
                    WHERE rp.finish_rank IS NOT NULL
                      AND rp.driver_jockey_id IS NOT NULL
                )
                -- Self-join : pour chaque participation, on compte les courses
                -- du même duo AVANT cette date
                SELECT
                    cur.horse_id,
                    cur.driver_jockey_id,
                    cur.race_id,
                    COUNT(hist.race_id)          AS duo_total_races,
                    COALESCE(SUM(hist.is_winner), 0)  AS duo_wins,
                    COALESCE(AVG(hist.is_winner), 0)  AS duo_win_rate,
                    COALESCE(AVG(hist.finish_rank), 99) AS duo_avg_rank,
                    COALESCE(MIN(hist.finish_rank), 99) AS duo_best_rank,
                    COALESCE(AVG(hist.reference_odds), 0) AS duo_avg_odds,
                    MAX(hist.program_date)       AS duo_last_date
                FROM jockey_horse_history cur
                LEFT JOIN jockey_horse_history hist
                    ON  hist.horse_id         = cur.horse_id
                    AND hist.driver_jockey_id = cur.driver_jockey_id
                    AND hist.program_date     < cur.program_date  -- strictement passé
                GROUP BY cur.horse_id, cur.driver_jockey_id, cur.race_id
                """
                jh_df = pd.read_sql(query_jockey_horse, connection)

                self.logger.info("Merging datasets...")
                # horse_stats = self._compute_horse_stats(history_df)
                # final_df = pd.merge(main_df, horse_stats, on='horse_id', how='left')
                horse_stats, discipline_affinity = self._compute_horse_stats(history_df)
                final_df = pd.merge(main_df, horse_stats, on='horse_id', how='left')

                # Affinité discipline : merge sur horse_id + discipline courante
                final_df = pd.merge(
                    final_df,
                    discipline_affinity,
                    on=['horse_id', 'discipline'],
                    how='left'
                )
                final_df['pct_races_on_discipline'] = final_df['pct_races_on_discipline'].fillna(0.0)

                # Valeurs par défaut pour nouveaux chevaux
                fill_defaults = {
                    'hist_races': 0, 'hist_earnings': 0,
                    'avg_speed_last_3': 0, 'avg_rank_last_3': 99,
                    'earnings_last_3': 0, 'days_since_last_race': 365,
                    'hist_pct_clean_runs': 1.0,
                    'pct_races_on_discipline': 0.0,
                    'pct_races_on_distance_range': 0.0,
                }
                for col, val in fill_defaults.items():
                    if col in final_df.columns:
                        final_df[col] = final_df[col].fillna(val)

                final_df = pd.merge(
                    final_df,
                    jh_df[['horse_id', 'driver_jockey_id', 'race_id',
                           'duo_total_races', 'duo_wins', 'duo_win_rate',
                           'duo_avg_rank', 'duo_best_rank']],
                    on=['horse_id', 'driver_jockey_id', 'race_id'],
                    how='left'
                )
                # Valeurs par défaut — duo jamais vu ensemble
                final_df['duo_total_races'] = final_df['duo_total_races'].fillna(0)
                final_df['duo_wins'] = final_df['duo_wins'].fillna(0)
                final_df['duo_win_rate'] = final_df['duo_win_rate'].fillna(0.0)
                final_df['duo_avg_rank'] = final_df['duo_avg_rank'].fillna(99)
                final_df['duo_best_rank'] = final_df['duo_best_rank'].fillna(99)

                # Supprimer driver_jockey_id après merge (pas une feature utile brute)
                final_df = final_df.drop(columns=['driver_jockey_id'])

                return final_df.sort_values('program_date')

        except Exception as error:
            self.logger.error(f"Critical error while loading data: {error}")
            raise

    def _compute_horse_stats(self, history_df: pd.DataFrame): # -> pd.DataFrame:
        """Calcule toutes les statistiques historiques par cheval."""
        # --- ÉTAPE 1 : is_clean_run en premier, tout le reste en dépend ---
        history_df = history_df.copy()  # évite le SettingWithCopyWarning
        # --- Fiabilité : % de courses "propres" (sans disqualification/galop) ---
        # Statuts qui indiquent une course "anormale" ou non terminée proprement
        # finish_status NULL ou 'OK' = course normale
        history_df['finish_status'] = history_df['finish_status'].replace(r'\N', None)
        bad_statuses = ['ARRETE', 'DEROBE', 'DISQUALIFIE', 'RESTE_AU_POTEAU', 'TOMBE', 'INCONNU']
        history_df['is_clean_run'] = (
            ~history_df['finish_status'].str.upper().isin(bad_statuses)
        ).astype(int)

        # --- Carrière complète ---
        career = history_df.groupby('horse_id').agg(
            hist_races=('finish_place', 'count'),
            hist_avg_rank=('finish_place', 'mean'),
            hist_avg_speed=('reduction_km', 'mean'),
            hist_best_speed=('reduction_km', 'min'),
            hist_earnings=('prize_money', 'sum'),
            hist_pct_clean_runs = ('is_clean_run', 'mean'),
        ).reset_index()
        career['hist_avg_speed'] = career['hist_avg_speed'].fillna(1.20)

        # --- Forme récente : 3 dernières courses ---
        recent_3 = history_df[history_df['race_recency_rank'] <= 3]
        recent_stats = recent_3.groupby('horse_id').agg(
            avg_speed_last_3=('reduction_km', 'mean'),
            avg_rank_last_3=('finish_place', 'mean'),
            earnings_last_3=('prize_money', 'sum'),
        ).reset_index()

        # --- Fiabilité récente : 5 dernières courses ---
        recent_5 = history_df[history_df['race_recency_rank'] <= 5]
        reliability_recent = recent_5.groupby('horse_id').agg(
            pct_clean_last_5=('is_clean_run', 'mean')
        ).reset_index()

        # --- Dernière course (fraîcheur) ---
        last_race = history_df[history_df['race_recency_rank'] == 1][
            ['horse_id', 'race_date']
        ].rename(columns={'race_date': 'last_race_date'})

        # --- Affinité distance (distance actuelle dans la plage historique ±200m) ---
        distance_stats = history_df.groupby('horse_id').agg(
            hist_avg_distance=('distance_m', 'mean'),
            hist_std_distance=('distance_m', 'std'),
        ).reset_index()

        # --- Affinité discipline (retournée séparément pour merge horse+discipline) ---
        discipline_counts = history_df.groupby(
            ['horse_id', 'discipline']
        ).agg(
            n_races_discipline=('finish_place', 'count'),
            pct_clean_on_discipline=('is_clean_run', 'mean'),
        ).reset_index()
        total_counts = history_df.groupby('horse_id').size().reset_index(name='total')
        discipline_affinity = discipline_counts.merge(total_counts, on='horse_id')
        discipline_affinity['pct_races_on_discipline'] = (
                discipline_affinity['n_races_discipline'] / discipline_affinity['total']
        )
        discipline_affinity = discipline_affinity[
            ['horse_id', 'discipline', 'pct_races_on_discipline', 'pct_clean_on_discipline']
        ]

        # --- Merge final horse_stats ---
        horse_stats = career \
            .merge(recent_stats, on='horse_id', how='left') \
            .merge(reliability_recent, on='horse_id', how='left') \
            .merge(last_race, on='horse_id', how='left') \
            .merge(distance_stats, on='horse_id', how='left')

        return horse_stats, discipline_affinity
    #
    # def _get_training_data_old(self) -> pd.DataFrame:
    #     """
    #     Extracts and merges participant data with aggregated historical statistics.
    #
    #     Returns:
    #         pd.DataFrame: The merged dataset sorted by program date.
    #     """
    #     self.logger.info("Loading SQL data...")
    #
    #     try:
    #         with self.engine.connect() as connection:
    #             # 1. Main Dataset (Optimized Query)
    #             # Note: SQL Column aliases are preserved to ensure downstream compatibility.
    #             query_main = """
    #             SELECT
    #                 rp.participant_id, rp.race_id, rp.horse_id,
    #                 rp.finish_rank,
    #                 CASE WHEN rp.finish_rank = 1 THEN 1 ELSE 0 END AS is_winner,
    #                 dp.program_date, rm.racetrack_code, rm.weather_temperature, rm.weather_wind,
    #                 r.race_number, r.discipline, r.distance_m, r.track_type, r.terrain_label, r.declared_runners_count,
    #                 h.birth_year, h.sex,
    #                 rp.pmu_number, rp.age, rp.career_winnings, rp.career_races_count, rp.trainer_advice,
    #                 rp.reference_odds, rp.live_odds,
    #                 ls.code AS shoeing_status,
    #                 j.actor_name AS jockey_name,
    #                 t.actor_name AS trainer_name
    #             FROM race_participant rp
    #             JOIN race r ON rp.race_id = r.race_id
    #             JOIN race_meeting rm ON r.meeting_id = rm.meeting_id
    #             JOIN daily_program dp ON rm.program_id = dp.program_id
    #             JOIN horse h ON rp.horse_id = h.horse_id
    #             LEFT JOIN lookup_shoeing ls ON rp.shoeing_id = ls.shoeing_id
    #             LEFT JOIN racing_actor j ON rp.driver_jockey_id = j.actor_id
    #             LEFT JOIN racing_actor t ON rp.trainer_id = t.actor_id
    #             WHERE rp.finish_rank IS NOT NULL
    #             """
    #             main_df = pd.read_sql(query_main, connection)
    #             main_df['program_date'] = pd.to_datetime(main_df['program_date'])
    #
    #             # 2. History Dataset & Aggregation
    #             self.logger.info("Calculating historical statistics... V2")
    #             query_history = """
    #             SELECT
    #                 horse_id, finish_place, reduction_km, prize_money,
    #                 race_date,
    #                 ROW_NUMBER() OVER (
    #                     PARTITION BY horse_id ORDER BY race_date DESC
    #                 ) AS race_recency_rank
    #             FROM horse_race_history
    #             SELECT
    #                 horse_id,
    #                 finish_place,
    #                 reduction_km,
    #                 prize_money,
    #                 race_date,
    #                 race_recency_rank
    #             FROM ranked_history
    #             """
    #             history_df = pd.read_sql(query_history, connection)
    #             # Stats carrière complète (comme avant)
    #             career_stats = history_df.groupby('horse_id').agg(
    #                 hist_races=('finish_place', 'count'),
    #                 hist_avg_rank=('finish_place', 'mean'),
    #                 hist_avg_speed=('reduction_km', 'mean'),
    #                 hist_best_speed=('reduction_km', 'min'),
    #                 hist_earnings=('prize_money', 'sum'),
    #             ).reset_index()
    #             # Stats forme récente (3 dernières courses)
    #             recent = history_df[history_df['race_recency_rank'] <= 3]
    #             recent_stats = recent.groupby('horse_id').agg(
    #                 avg_speed_last_3=('reduction_km', 'mean'),
    #                 avg_rank_last_3=('finish_place', 'mean'),
    #                 earnings_last_3=('prize_money', 'sum'),
    #             ).reset_index()
    #             # Dernière course (pour days_since_last_race)
    #             last_race = history_df[history_df['race_recency_rank'] == 1][
    #                 ['horse_id', 'race_date']
    #             ].rename(columns={'race_date': 'last_race_date'})
    #
    #             # Optimized statistics calculation
    #             horse_stats = history_df.groupby('horse_id').agg({
    #                 'finish_place': ['count', 'mean'],
    #                 'reduction_km': ['mean', 'min'],
    #                 'prize_money': 'sum'
    #             }).reset_index()
    #
    #             # Merge tout ensemble V2
    #             horse_stats = career_stats \
    #                 .merge(recent_stats, on='horse_id', how='left') \
    #                 .merge(last_race, on='horse_id', how='left')
    #
    #             # Flatten MultiIndex columns and rename V2
    #             # horse_stats.columns = [
    #             #     'horse_id', 'hist_races', 'hist_avg_rank',
    #             #     'hist_avg_speed', 'hist_best_speed', 'hist_earnings'
    #             # ]
    #
    #             # Default value for speed (1.20 = slow/standard context)
    #             horse_stats['hist_avg_speed'] = horse_stats['hist_avg_speed'].fillna(1.20)
    #
    #             # 3. Merge
    #             self.logger.info("Merging datasets...")
    #             final_df = pd.merge(main_df, horse_stats, on='horse_id', how='left')
    #
    #             # Basic handling of NULLs post-merge (for "new" horses not in history)
    #             final_df['hist_races'] = final_df['hist_races'].fillna(0)
    #             final_df['hist_earnings'] = final_df['hist_earnings'].fillna(0)
    #
    #             return final_df.sort_values('program_date')
    #
    #     except Exception as error:
    #         self.logger.error(f"Critical error while loading data: {error}")
    #         raise error