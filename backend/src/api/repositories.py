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
        """
        Retrieves ML model performance metrics from the database.
        """
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

    def get_races_by_date(self, date_code: str) -> List[Dict[str, Any]]:
        """
        Retrieves all races scheduled for a specific date.
        """
        try:
            target_date = dt.datetime.strptime(date_code, "%d%m%Y").date()
        except ValueError:
            logger.warning(f"Invalid date format provided: {date_code}")
            return []

        query = """
            SELECT 
                r.race_id, 
                rm.meeting_number, 
                r.race_number, 
                dp.program_date,
                r.discipline, 
                r.distance_m, 
                r.track_type,
                rm.racetrack_code, rm.racetrack_libelle, rm.meeting_type, rm.weather_windspeed,
                r.declared_runners_count,
                r.start_timestamp,
                r.timezone_offset,
                r.prize_money,
                r.speciality
            FROM race r
            JOIN race_meeting rm ON r.meeting_id = rm.meeting_id
            JOIN daily_program dp ON rm.program_id = dp.program_id
            WHERE dp.program_date = %s
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
        """
        Retrieves basic participant details for a specific race.
        """
        query = """
            SELECT 
                rp.pmu_number AS program_number, 
                h.horse_name, 
                rp.age,
                h.sex,
                d.actor_name AS jockey_name, 
                t.actor_name AS trainer_name, 
                rp.reference_odds,
                rp.live_odds,
                ls.code AS shoeing_status,
                rp.blinkers,
                rp.handicap_value,
                o.actor_name AS owner_name,
                ls.code AS shoeing_status
            FROM race_participant rp
            JOIN horse h ON rp.horse_id = h.horse_id
            LEFT JOIN lookup_shoeing ls ON rp.shoeing_id = ls.shoeing_id
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
        """
        Extracts comprehensive feature sets for the Machine Learning pipeline.
        """
        query = """
            WITH horse_stats AS (
                SELECT 
                    horse_id,
                    COUNT(*) as hist_races,
                    AVG(finish_place) as hist_avg_rank,
                    AVG(reduction_km) as hist_avg_speed,
                    SUM(prize_money) as hist_earnings
                FROM horse_race_history
                WHERE horse_id IN (SELECT horse_id FROM race_participant WHERE race_id = %s)
                GROUP BY horse_id
            )
            
            SELECT 
                rp.race_id, 
                rp.pmu_number AS program_number, 
                h.horse_name,
                dp.program_date, 
                r.distance_m, 
                r.declared_runners_count,
                rm.racetrack_code, rm.racetrack_libelle, rm.meeting_type, rm.weather_windspeed,
                r.discipline, 
                r.track_type,
                rm.weather_wind,
                rm.weather_temperature, 
                r.terrain_label,
                rp.age, 
                rp.career_winnings, 
                rp.career_races_count, 
                h.birth_year,
                rp.reference_odds, 
                rp.live_odds,
                COALESCE(hs.hist_avg_speed, %s) as hist_avg_speed, 
                COALESCE(hs.hist_earnings, 0) as hist_earnings,
                COALESCE(hs.hist_races, 0) as hist_races,
                ls.code AS shoeing_status,
                h.sex,
                d.actor_name AS jockey_name,
                t.actor_name AS trainer_name
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
        """
        Batch retrieval for all races on a specific day.
        """
        try:
            target_date = dt.datetime.strptime(date_code, "%d%m%Y").date()
        except ValueError:
            logger.warning(f"Invalid date format: {date_code}")
            return []

        query = """
            WITH horse_stats AS (
                SELECT 
                    horse_id,
                    COUNT(*) as hist_races,
                    AVG(finish_place) as hist_avg_rank,
                    AVG(reduction_km) as hist_avg_speed,
                    SUM(prize_money) as hist_earnings
                FROM horse_race_history
                WHERE horse_id IN (
                    SELECT rp.horse_id 
                    FROM race_participant rp
                    JOIN race r ON rp.race_id = r.race_id
                    JOIN race_meeting rm ON r.meeting_id = rm.meeting_id
                    JOIN daily_program dp ON rm.program_id = dp.program_id
                    WHERE dp.program_date = %s
                )
                GROUP BY horse_id
            )
            
            SELECT 
                rp.race_id, 
                rm.meeting_number,
                r.race_number,
                rp.pmu_number AS program_number, 
                h.horse_name,
                dp.program_date, 
                r.distance_m, 
                r.declared_runners_count,
                rm.racetrack_code, rm.racetrack_libelle, rm.meeting_type, rm.weather_windspeed,
                r.discipline, 
                r.track_type,
                rm.weather_wind, 
                rm.weather_temperature, 
                r.terrain_label,
                rp.age, 
                rp.career_winnings, 
                rp.career_races_count, 
                h.birth_year,
                rp.reference_odds, 
                rp.live_odds,
                COALESCE(hs.hist_avg_speed, %s) as hist_avg_speed,
                COALESCE(hs.hist_earnings, 0) as hist_earnings,
                COALESCE(hs.hist_races, 0) as hist_races,
                ls.code AS shoeing_status,
                h.sex,
                d.actor_name AS jockey_name,
                t.actor_name AS trainer_name
            FROM race_participant rp
            JOIN race r ON rp.race_id = r.race_id
            JOIN race_meeting rm ON r.meeting_id = rm.meeting_id
            JOIN daily_program dp ON rm.program_id = dp.program_id
            JOIN horse h ON rp.horse_id = h.horse_id
            LEFT JOIN horse_stats hs ON rp.horse_id = hs.horse_id
            LEFT JOIN lookup_shoeing ls ON rp.shoeing_id = ls.shoeing_id
            LEFT JOIN racing_actor d ON rp.driver_jockey_id = d.actor_id
            LEFT JOIN racing_actor t ON rp.trainer_id = t.actor_id
            WHERE dp.program_date = %s
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
