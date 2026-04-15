"""
Database Access Layer.
Handles all SQL interactions for retrieving race and participant data.
"""
import datetime as dt
import logging
from typing import List, Dict, Any, Optional
import psycopg2.extras

from src.core.database import DatabaseManager

logger = logging.getLogger(__name__)

# Constants
DEFAULT_HISTORICAL_SPEED = 1.20  # Default kilometer reduction if history is missing

class RaceRepository:
    """
    Repository for accessing Race and Participant data from the PostgreSQL database.
    """

    def __init__(self) -> None:
        self.db_manager = DatabaseManager()

    def get_model_metrics(self, model_name: Optional[str] = None, segment_type: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM ml_model_metrics WHERE 1=1"
        params = []
        if model_name:
            query += " AND model_name = %s"
            params.append(model_name)
        if segment_type:
            query += " AND segment_type = %s"
            params.append(segment_type)
        
        query += " ORDER BY model_name, segment_type, segment_value, test_month"

        conn = self.db_manager.get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, tuple(params))
                return cur.fetchall()
        except Exception as exc:
            logger.error(f"Error fetching model metrics: {exc}")
            return []
        finally:
            self.db_manager.release_connection(conn)

    def get_backtest_data(self, date_start: Optional[dt.date] = None, date_end: Optional[dt.date] = None) -> List[Dict[str, Any]]:
        query = """
            SELECT 
                r.race_id,
                r.discipline,
                dp.program_date,
                rp.participant_id,
                rp.pmu_number AS program_number,
                rp.finish_rank,
                rp.reference_odds,
                rp.live_odds,
                rp.live_odds_30mn,
                p.model_version,
                p.proba_winner,
                p.proba_top3_place,
                rb.bet_type,
                br.combination,
                br.dividend_per_1e
            FROM race r
            JOIN daily_program dp ON r.meeting_id IN (SELECT meeting_id FROM race_meeting rm WHERE program_id = dp.program_id AND rm.audience = 'NATIONAL')
            JOIN race_participant rp ON r.race_id = rp.race_id
            LEFT JOIN prediction p ON rp.participant_id = p.participant_id
            LEFT JOIN race_bet rb ON r.race_id = rb.race_id
            LEFT JOIN bet_report br ON rb.bet_id = br.bet_id
            WHERE 1=1
        """
        params = []
        if date_start:
            query += " AND dp.program_date >= %s"
            params.append(date_start)
        else:
            query += " AND dp.program_date >= CURRENT_DATE - INTERVAL '1 year'"

        if date_end:
            query += " AND dp.program_date <= %s"
            params.append(date_end)

        query += """
              AND rp.finish_rank IS NOT NULL 
              AND (rb.bet_family IN ('Simple', 'Couple', 'Trio'))
            ORDER BY dp.program_date DESC, r.race_id, rp.pmu_number;
        """
        conn = self.db_manager.get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, tuple(params))
                return cur.fetchall()
        except Exception as exc:
            logger.error(f"Error fetching backtest data: {exc}", exc_info=True)
            return []
        finally:
            self.db_manager.release_connection(conn)

    def get_races_by_date(self, date_code: str) -> List[Dict[str, Any]]:
        try:
            target_date = dt.datetime.strptime(date_code, "%d%m%Y").date()
        except ValueError:
            return []

        query = """
            SELECT 
                r.race_id, rm.meeting_number, r.race_number, dp.program_date,
                r.discipline, r.distance_m, r.track_type,
                rm.meeting_code, rm.meeting_libelle, rm.meeting_type, rm.weather_windspeed,
                r.declared_runners_count, r.start_timestamp, r.timezone_offset, r.prize_money, r.speciality,
                r.racetrack_libelle
            FROM race r
            JOIN race_meeting rm ON r.meeting_id = rm.meeting_id
            JOIN daily_program dp ON rm.program_id = dp.program_id
            WHERE dp.program_date = %s AND rm.audience = 'NATIONAL'
            ORDER BY rm.meeting_number, r.race_number;
        """
        conn = self.db_manager.get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, (target_date,))
                return cur.fetchall()
        except Exception as exc:
            logger.error(f"Error fetching races for date {date_code}: {exc}")
            return []
        finally:
            self.db_manager.release_connection(conn)

    def get_participants_by_race(self, race_id: int) -> List[Dict[str, Any]]:
        query = """
            SELECT 
                rp.pmu_number AS program_number, h.horse_name, rp.age, h.sex,
                d.actor_name AS jockey_name, t.actor_name AS trainer_name, 
                rp.reference_odds, rp.live_odds, rp.live_odds_30mn,
                ls.code AS shoeing_status, rp.blinkers, rp.handicap_value,
                o.actor_name AS owner_name, rp.finish_rank, li.code AS incident_code,
                dp.program_date, r.discipline
            FROM race_participant rp
            JOIN horse h ON rp.horse_id = h.horse_id
            JOIN race r ON rp.race_id = r.race_id            
            JOIN race_meeting rm ON r.meeting_id = rm.meeting_id
            JOIN daily_program dp ON rm.program_id = dp.program_id
            LEFT JOIN lookup_shoeing ls ON rp.shoeing_id = ls.shoeing_id
            LEFT JOIN lookup_incident li ON rp.incident_id = li.incident_id
            LEFT JOIN racing_actor d ON rp.driver_jockey_id = d.actor_id
            LEFT JOIN racing_actor t ON rp.trainer_id = t.actor_id
            LEFT JOIN racing_actor o ON rp.owner_id = o.actor_id
            WHERE rp.race_id = %s
            ORDER BY rp.pmu_number;
        """
        conn = self.db_manager.get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, (race_id,))
                return cur.fetchall()
        except Exception as exc:
            logger.error(f"Error fetching participants for race {race_id}: {exc}")
            return []
        finally:
            self.db_manager.release_connection(conn)

    def get_race_data_for_ml(self, race_id: int) -> List[Dict[str, Any]]:
        query = """
            WITH horse_stats AS (
                SELECT horse_id, COUNT(*) as hist_races, AVG(finish_place) as hist_avg_rank,
                       AVG(reduction_km) as hist_avg_speed, SUM(prize_money) as hist_earnings
                FROM horse_race_history
                WHERE horse_id IN (SELECT horse_id FROM race_participant WHERE race_id = %s)
                GROUP BY horse_id
            )
            SELECT 
                rp.participant_id, rp.race_id, rp.pmu_number AS program_number, h.horse_name,
                dp.program_date, r.distance_m, r.declared_runners_count,
                rm.meeting_code, rm.meeting_libelle, rm.meeting_type, rm.weather_windspeed,
                r.discipline, r.track_type, rm.weather_wind, rm.weather_temperature, r.terrain_label,
                rp.age, rp.career_winnings, rp.career_races_count, h.birth_year,
                rp.reference_odds, rp.live_odds, rp.live_odds_30mn,
                COALESCE(hs.hist_avg_speed, %s) as hist_avg_speed, COALESCE(hs.hist_earnings, 0) as hist_earnings,
                COALESCE(hs.hist_races, 0) as hist_races, ls.code AS shoeing_status,
                h.sex, d.actor_name AS jockey_name, t.actor_name AS trainer_name, r.racetrack_libelle
            FROM race_participant rp
            JOIN race r ON rp.race_id = r.race_id
            JOIN race_meeting rm ON r.meeting_id = rm.meeting_id
            JOIN daily_program dp ON rm.program_id = dp.program_id
            JOIN horse h ON rp.horse_id = h.horse_id
            LEFT JOIN horse_stats hs ON rp.horse_id = hs.horse_id
            LEFT JOIN lookup_shoeing ls ON rp.shoeing_id = ls.shoeing_id
            LEFT JOIN racing_actor d ON rp.driver_jockey_id = d.actor_id
            LEFT JOIN racing_actor t ON rp.trainer_id = t.actor_id
            WHERE rp.race_id = %s
            ORDER BY rp.pmu_number;
        """
        conn = self.db_manager.get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, (race_id, DEFAULT_HISTORICAL_SPEED, race_id))
                return cur.fetchall()
        except Exception as exc:
            logger.error(f"Database error in get_race_data_for_ml for race {race_id}: {exc}")
            return []
        finally:
            self.db_manager.release_connection(conn)

    def get_daily_data_for_ml(self, date_code: str) -> List[Dict[str, Any]]:
        try:
            target_date = dt.datetime.strptime(date_code, "%d%m%Y").date()
        except ValueError:
            return []

        query = """
            WITH horse_stats AS (
                SELECT horse_id, COUNT(*) as hist_races, AVG(finish_place) as hist_avg_rank,
                       AVG(reduction_km) as hist_avg_speed, SUM(prize_money) as hist_earnings
                FROM horse_race_history
                WHERE horse_id IN (
                    SELECT rp.horse_id FROM race_participant rp
                    JOIN race r ON rp.race_id = r.race_id
                    JOIN daily_program dp ON r.meeting_id IN (SELECT meeting_id FROM race_meeting rm WHERE program_id = dp.program_id)
                    WHERE dp.program_date = %s
                )
                GROUP BY horse_id
            )
            SELECT 
                rp.participant_id, rp.race_id, rm.meeting_number, r.race_number,
                rp.pmu_number AS program_number, h.horse_name, dp.program_date, 
                r.distance_m, r.declared_runners_count,
                rm.meeting_code, rm.meeting_libelle, rm.meeting_type, rm.weather_windspeed,
                r.discipline, r.track_type, rm.weather_wind, rm.weather_temperature, r.terrain_label,
                rp.age, rp.career_winnings, rp.career_races_count, h.birth_year,
                rp.reference_odds, rp.live_odds, rp.live_odds_30mn,
                COALESCE(hs.hist_avg_speed, %s) as hist_avg_speed, COALESCE(hs.hist_earnings, 0) as hist_earnings,
                COALESCE(hs.hist_races, 0) as hist_races, ls.code AS shoeing_status,
                h.sex, d.actor_name AS jockey_name, t.actor_name AS trainer_name, r.racetrack_libelle
            FROM race_participant rp
            JOIN race r ON rp.race_id = r.race_id
            JOIN race_meeting rm ON r.meeting_id = rm.meeting_id
            JOIN daily_program dp ON rm.program_id = dp.program_id
            JOIN horse h ON rp.horse_id = h.horse_id
            LEFT JOIN horse_stats hs ON rp.horse_id = hs.horse_id
            LEFT JOIN lookup_shoeing ls ON rp.shoeing_id = ls.shoeing_id
            LEFT JOIN racing_actor d ON rp.driver_jockey_id = d.actor_id
            LEFT JOIN racing_actor t ON rp.trainer_id = t.actor_id
            WHERE dp.program_date = %s AND rm.audience = 'NATIONAL'
            ORDER BY rp.race_id, rp.pmu_number;
        """
        conn = self.db_manager.get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, (target_date, DEFAULT_HISTORICAL_SPEED, target_date))
                return cur.fetchall()
        except Exception as exc:
            logger.error(f"Database error in get_daily_data_for_ml for date {target_date}: {exc}")
            return []
        finally:
            self.db_manager.release_connection(conn)

    def upsert_predictions(self, predictions: List[Dict[str, Any]]) -> bool:
        """
        Saves prediction results to the database.
        SECURITY: Does NOT update if the race is already finished (finish_rank exists).
        """
        if not predictions:
            return True

        # Query checking if participant already has a finish_rank
        query = """
            INSERT INTO prediction (participant_id, model_version, proba_winner, proba_top3_place)
            SELECT %s, %s, %s, %s
            WHERE NOT EXISTS (
                SELECT 1 FROM race_participant 
                WHERE participant_id = %s AND finish_rank IS NOT NULL
            )
            ON CONFLICT (participant_id, model_version) DO UPDATE SET
                proba_winner = EXCLUDED.proba_winner,
                proba_top3_place = EXCLUDED.proba_top3_place,
                created_at = CURRENT_TIMESTAMP;
        """

        params = [(p['participant_id'], p['model_version'], p['proba_winner'], p.get('proba_top3_place'), p['participant_id']) for p in predictions]

        conn = self.db_manager.get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    psycopg2.extras.execute_batch(cur, query, params)
            return True
        except Exception as exc:
            logger.error(f"Error upserting predictions: {exc}")
            return False
        finally:
            self.db_manager.release_connection(conn)

    def upsert_daily_performance(self, performances: List[Dict[str, Any]]) -> bool:
        if not performances:
            return True
        query = """
            INSERT INTO ml_daily_performance (performance_date, model_version, discipline, bet_type, nb_bets, nb_wins, roi, avg_odds)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (performance_date, model_version, discipline, bet_type) DO UPDATE SET
                nb_bets = EXCLUDED.nb_bets,
                nb_wins = EXCLUDED.nb_wins,
                roi = EXCLUDED.roi,
                avg_odds = EXCLUDED.avg_odds;
        """
        params = [(p['performance_date'], p['model_version'], p['discipline'], p['bet_type'], p['nb_bets'], p['nb_wins'], p['roi'], p['avg_odds']) for p in performances]
        conn = self.db_manager.get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    psycopg2.extras.execute_batch(cur, query, params)
            return True
        except Exception as exc:
            logger.error(f"Error upserting daily performance: {exc}")
            return False
        finally:
            self.db_manager.release_connection(conn)

    def get_daily_performance(self, performance_date: dt.date) -> List[Dict[str, Any]]:
        query = "SELECT * FROM ml_daily_performance WHERE performance_date = %s"
        conn = self.db_manager.get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, (performance_date,))
                return cur.fetchall()
        except Exception as exc:
            logger.error(f"Error fetching daily performance for {performance_date}: {exc}")
            return []
        finally:
            self.db_manager.release_connection(conn)

    def get_best_model_for_context(self, discipline: str, month: int) -> Optional[str]:
        query = """
             SELECT algorithm FROM ml_model_metrics 
             WHERE segment_value = %s AND test_month = %s AND segment_type = 'discipline_month'
             ORDER BY roi DESC LIMIT 1;
         """
        conn = self.db_manager.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(query, (discipline.upper(), month))
                res = cur.fetchone()
                return res[0] if res else None
        except Exception as exc:
            logger.error(f"Error fetching best model: {exc}")
            return None
        finally:
            self.db_manager.release_connection(conn)
