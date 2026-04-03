import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

pd.set_option('future.no_silent_downcasting', True)

class RaceContextEncoder(BaseEstimator, TransformerMixin):
    """
    Transforms raw horse features into contextual features relative to the race.
    """
    def __init__(self, group_col="race_id", feature_cols=None):
        self.group_col = group_col
        self.feature_cols = feature_cols
        # Si feature_cols est passé explicitement, on l'utilise tel quel.
        # Sinon on définit la liste par défaut ici dans __init__,
        # PAS dans transform() pour éviter la mutation d'état.
        self.feature_cols = feature_cols or [
            'reference_odds', 'career_winnings', 'horse_age_at_race',
            'hist_avg_speed', 'hist_earnings',
            'avg_speed_last_3',
            'days_since_last_race',
            'hist_pct_clean_runs',
            'duo_win_rate',      # ← nouveau
            'duo_avg_rank',      # ← nouveau (rank relatif dans la course)
            'duo_confidence',    # ← nouveau
            'duo_best_rank',  # Added duo_best_rank for contextualization
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # if self.feature_cols is None:
            # self.feature_cols = [
            #     'reference_odds', 'career_winnings', 'horse_age_at_race',
            #     'hist_avg_speed', 'hist_earnings'
            # ] # Mutation de l'état interne lors du transform !
        # modifies self.feature_cols lors du transform, ce qui veut dire qu'après le premier appel, le comportement change si quelqu'un passe feature_cols=None explicitement. C'est un anti-pattern sklearn — transform ne doit jamais muter l'état.
        #     self.feature_cols = self.feature_cols or [
        #         'reference_odds', 'career_winnings', 'horse_age_at_race',
        #         'hist_avg_speed', 'hist_earnings', 'avg_speed_last_3_races',
        #         'avg_speed_last_3',       # nouveau
        #         'days_since_last_race',   # nouveau
        #         'hist_pct_clean_runs',    # nouveau
        #     ]
        # Plus de mutation ici — feature_cols est déjà défini dans __init__
        feature_cols = self.feature_cols  # lecture seule

        #J cols_to_use = [c for c in feature_cols if c in df.columns]
        if self.group_col not in df.columns:
            return df

        grouped = df.groupby(self.group_col)
        for col in feature_cols: #J cols_to_use:
            # Ensure the column exists before trying to group/transform
            if col not in df.columns:
                df[col] = np.nan # Fill with NaN, imputer will handle it later
            # Stats per race
            mean_val = grouped[col].transform("mean")
            std_val = grouped[col].transform("std")
            
            # Relative, Z-score and Rank features
            df[f"{col}_rel_race"] = df[col] - mean_val
            df[f"{col}_z_race"] = (df[col] - mean_val) / (std_val + 1e-6)
            df[f"{col}_rank_race"] = grouped[col].rank(pct=True)

        # PMU Market Sentiment (Implied Probability)
        if 'reference_odds' in df.columns:
            df['implied_prob'] = 1.0 / df['reference_odds'].replace(0, np.nan).fillna(100).clip(lower=1.01)
            race_total_prob = grouped['implied_prob'].transform("sum")
            df['market_sentiment'] = df['implied_prob'] / (race_total_prob + 1e-6)
            
        return df

class PmuFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Intelligent Scikit-learn Transformer.
    Integrates advanced business logic:
    - Debutant handling
    - Relative financial ratios
    - Intra-race rankings
    - Statistical imputation learned during fit().
    """

    def __init__(self):
        # Stats learned during fit
        self.learned_stats_ = {} 
        self.cat_fill_value_ = 'MISSING'

    def fit(self, X: pd.DataFrame, y=None) -> 'PmuFeatureEngineer':
        """Learns global statistics for imputation."""
        df = X.copy()
        
        # 1. Learn global average temperature
        if 'weather_temperature' in df.columns:
            self.learned_stats_['weather_temperature'] = df['weather_temperature'].mean()
            
        # 2. Learn average reference odds (for nulls)
        if 'reference_odds' in df.columns:
            self.learned_stats_['reference_odds'] = df['reference_odds'].mean()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies transformations without data leakage."""
        df = X.copy()
        
        # --- 1. Dates & Ages ---
        if 'program_date' in df.columns:
            df['program_date'] = pd.to_datetime(df['program_date'])
            df['race_month'] = df['program_date'].dt.month
            df['race_day_of_week'] = df['program_date'].dt.dayofweek
            
            # Robust age calculation
            if 'birth_year' in df.columns:
                # Fallback to 'age' column if birth_year is missing, otherwise calculate
                current_year = df['program_date'].dt.year
                calculated_age = current_year - df['birth_year']
                df['horse_age_at_race'] = calculated_age.fillna(df['age'])
            else:
                df['horse_age_at_race'] = df['age']

        # Fraîcheur
        # Ensure 'days_since_last_race' is always present
        if 'last_race_date' in df.columns and 'program_date' in df.columns:
            df['last_race_date'] = pd.to_datetime(df['last_race_date'], errors='coerce')
            df['days_since_last_race'] = (
                    df['program_date'] - df['last_race_date']
            ).dt.days.fillna(365).clip(lower=0)
        else:
            df['days_since_last_race'] = 365 # Default if dates are missing

        # Ratio forme récente vs carrière
        # Ensure 'speed_form_ratio' is always present
        if 'avg_speed_last_3' in df.columns and 'hist_avg_speed' in df.columns:
            df['speed_form_ratio'] = (
                    df['avg_speed_last_3'] /
                    df['hist_avg_speed'].replace(0, np.nan)
            ).fillna(1.0)
        else:
            df['speed_form_ratio'] = 1.0 # Default if speeds are missing

        # Affinité distance : cheval dans sa plage habituelle ?
        # Ensure 'distance_delta' and 'is_unusual_distance' are always present
        if 'distance_m' in df.columns and 'hist_avg_distance' in df.columns:
            df['distance_delta'] = (
                    df['distance_m'] - df['hist_avg_distance']
            ).fillna(0).abs()
            # Flag si la distance est inhabituelle (> 400m de sa moyenne)
            df['is_unusual_distance'] = (df['distance_delta'] > 400).astype(int)
        else:
            df['distance_delta'] = 0.0 # Default
            df['is_unusual_distance'] = 0 # Default

        # Spécialiste discipline
        # Ensure 'is_specialist' is always present
        if 'pct_races_on_discipline' in df.columns:
            df['is_specialist'] = (df['pct_races_on_discipline'] > 0.7).astype(int)
        else:
            df['is_specialist'] = 0 # Default

        # --- 2. Intelligent Imputation ---
        # Temperature (By Racetrack if possible, otherwise Global learned)
        if 'weather_temperature' in df.columns:
            # First fill by track mean within this batch (if available)
            df['weather_temperature'] = df['weather_temperature'].fillna(
                df.groupby('racetrack_code')['weather_temperature'].transform('mean')
            )
            # Fallback to learned global value
            df['weather_temperature'] = df['weather_temperature'].fillna(
                self.learned_stats_.get('weather_temperature', 15.0)
            )

        # Categorical Filling
        cat_cols = ['racetrack_code', 'discipline', 'track_type', 'sex', 
                   'shoeing_status', 'jockey_name', 'trainer_name', 'terrain_label', 'meeting_type',
                   'breed', 'color', 'blinkers', 'allure',
                   'father_name', 'mother_name', 'maternal_grandfather_name']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna(self.cat_fill_value_).astype(str)

        # --- 3. Business Feature Engineering ---
        
        # A. Odds Logic
        if 'reference_odds' in df.columns:
            df['is_odds_missing'] = df['reference_odds'].isnull().astype(int)
            # Impute with average of the current RACE first
            if 'race_id' in df.columns:
                race_avg_odds = df.groupby('race_id')['reference_odds'].transform('mean')
                df['reference_odds'] = df['reference_odds'].fillna(race_avg_odds)
            
            # Final Fallback
            df['reference_odds'] = df['reference_odds'].fillna(
                self.learned_stats_.get('reference_odds', 10.0)
            )

            # Odds rank within the race (Powerful predictor)
            if 'race_id' in df.columns:
                df['odds_rank_in_race'] = df.groupby('race_id')['reference_odds'].rank(ascending=True, method='min')
        else: # Ensure these are present even if reference_odds is missing
            df['is_odds_missing'] = 1
            df['reference_odds'] = self.learned_stats_.get('reference_odds', 10.0)
            df['odds_rank_in_race'] = 0.5 # Neutral rank

        # B. Finances & Performance
        if 'career_winnings' in df.columns:
            df['career_winnings'] = df['career_winnings'].fillna(0)
            df['career_races_count'] = df['career_races_count'].fillna(0)
            
            # "Debutant" Flag
            df['is_debutant'] = (df['career_races_count'] == 0).astype(int)
            
            # Winnings per race
            df['winnings_per_race'] = df['career_winnings'] / (df['career_races_count'] + 1)
            
            if 'race_id' in df.columns:
                # Is this the richest horse in the race?
                df['winnings_rank_in_race'] = df.groupby('race_id')['career_winnings'].rank(ascending=False, method='min')
                
                # Ratio vs Race Average (Contextual)
                race_avg_earnings = df.groupby('race_id')['career_winnings'].transform('mean')
                df['relative_winnings'] = df['career_winnings'] / (race_avg_earnings + 1)
        else: # Ensure defaults if career_winnings is missing
            df['career_winnings'] = 0
            df['career_races_count'] = 0
            df['is_debutant'] = 1
            df['winnings_per_race'] = 0
            df['winnings_rank_in_race'] = 0.5
            df['relative_winnings'] = 1.0

        # C. Forme récente
        if 'last_race_date' in df.columns and 'program_date' in df.columns:
            df['program_date'] = pd.to_datetime(df['program_date'])
            df['last_race_date'] = pd.to_datetime(df['last_race_date'])
            df['days_since_last_race'] = (
                    df['program_date'] - df['last_race_date']
            ).dt.days.fillna(365)  # 365 = jamais couru ou inconnu

        if 'avg_speed_last_3_races' in df.columns:
            df['avg_speed_last_3_races'] = df['avg_speed_last_3_races'].fillna(
                df['hist_avg_speed'].fillna(0)  # fallback sur moyenne carrière
            )
            # Ratio forme récente vs moyenne carrière
            df['speed_form_ratio'] = df['avg_speed_last_3_races'] / (
                df['hist_avg_speed'].replace(0, np.nan).fillna(1)
            )

        # D. Mouvement de cote (market drift)
        if 'live_odds' in df.columns and 'reference_odds' in df.columns:
            live = pd.to_numeric(df['live_odds'], errors='coerce')
            ref = pd.to_numeric(df['reference_odds'], errors='coerce')

            # Ratio : <1 = raccourcissement (favori du marché), >1 = dérive
            df['odds_drift_ratio'] = live / ref.replace(0, np.nan)

            # Version log pour lisser les extrêmes
            df['odds_drift_log'] = np.log(df['odds_drift_ratio'].clip(lower=0.1))

            # Flag raccourcissement fort (signal positif historiquement)
            df['is_drifting_in'] = (df['odds_drift_ratio'] < 0.85).astype(int)

            df['odds_drift_ratio'] = df['odds_drift_ratio'].fillna(1.0)
            df['odds_drift_log'] = df['odds_drift_log'].fillna(0.0)
        else:  # Ensure defaults if odds are missing
            df['odds_drift_ratio'] = 1.0
            df['odds_drift_log'] = 0.0
            df['is_drifting_in'] = 0

            # odds_log — actuellement calculé dans le trainer, doit être ici
        if 'reference_odds' in df.columns:
            df['odds_log'] = np.log1p(
                pd.to_numeric(df['reference_odds'], errors='coerce').fillna(20)
            )
        else:
            df['odds_log'] = np.log1p(20)  # Default

            # Fraîcheur
        if 'last_race_date' in df.columns and 'program_date' in df.columns:
            df['last_race_date'] = pd.to_datetime(df['last_race_date'], errors='coerce')
            df['days_since_last_race'] = (
                    df['program_date'] - df['last_race_date']
            ).dt.days.clip(lower=0).fillna(365)
        else:
            # À l'inférence si last_race_date absent : valeur neutre
            if 'days_since_last_race' not in df.columns:
                df['days_since_last_race'] = 365

            # Forme récente
        if 'avg_speed_last_3' not in df.columns:
            df['avg_speed_last_3'] = df.get('hist_avg_speed', pd.Series(0, index=df.index))
        if 'hist_avg_speed' in df.columns:
            df['speed_form_ratio'] = (
                    df['avg_speed_last_3'] /
                    df['hist_avg_speed'].replace(0, np.nan)
            ).fillna(1.0)
        else:
            df['speed_form_ratio'] = 1.0

            # Fiabilité — valeurs neutres si absentes (inférence sans historique)
        for col, default in [
            ('hist_pct_clean_runs', 1.0),
            ('pct_clean_on_discipline', 1.0),
            ('pct_races_on_discipline', 0.0),
        ]:
            if col not in df.columns:
                df[col] = default

            # Spécialiste discipline
        if 'pct_races_on_discipline' in df.columns:
            df['is_specialist'] = (df['pct_races_on_discipline'] > 0.7).astype(int)

        # "Business Feature Engineering"

        # E. Duo jockey×cheval - Ensure all duo_ features are present
        # These features are expected to come from the raw data, but we provide defaults
        # if they are missing, so subsequent steps don't fail.
        if 'duo_total_races' in df.columns:
            df['duo_total_races'] = df.get('duo_total_races', 0).fillna(0)
            df['duo_win_rate'] = df.get('duo_win_rate', 0.0).fillna(0.0)
            df['duo_avg_rank'] = df.get('duo_avg_rank', 0.0).fillna(0.0)
            df['duo_confidence'] = df.get('duo_confidence', 0.0).fillna(0.0)
            df['duo_best_rank'] = df.get('duo_best_rank', 0.0).fillna(0.0)  # Ensure this is present

            # Flag : duo expérimenté (5+ courses ensemble)
            df['duo_is_experienced'] = (df['duo_total_races'] >= 5).astype(int)
            # Flag : duo gagnant (win_rate > moyenne discipline)
            df['duo_is_winner'] = (df['duo_win_rate'] > 0.15).astype(int)
            # Confiance dans le duo — log pour éviter les extrêmes
            df['duo_confidence'] = np.log1p(df['duo_total_races']) * df['duo_win_rate']

        # Ensure penetrometer is always present and numeric
        if 'penetrometer' not in df.columns:
            df['penetrometer'] = np.nan  # Will be imputed by SimpleImputer

        # Fiabilité — valeurs neutres si absentes (inférence sans historique)
        for col, default in [
            ('hist_pct_clean_runs', 1.0),
            ('pct_clean_on_discipline', 1.0),
            ('pct_races_on_discipline', 0.0),
            ('hist_avg_speed', 0.0),  # Ensure hist_avg_speed is present for speed_form_ratio
            ('avg_speed_last_3', 0.0),  # Ensure avg_speed_last_3 is present for speed_form_ratio
        ]:
            if col not in df.columns:
                df[col] = default

        # Derived stats from enriched data
        if 'career_wins_count' in df.columns and 'career_races_count' in df.columns:
            df['career_win_rate'] = (df['career_wins_count'] / (df['career_races_count'] + 1)).fillna(0)

        if 'career_places_count' in df.columns and 'career_races_count' in df.columns:
            df['career_place_rate'] = (df['career_places_count'] / (df['career_races_count'] + 1)).fillna(0)

        # Dernière étape de transform() — garantit toutes les colonnes attendues
        from src.ml.feature_config import FEATURE_DEFAULTS
        for col, default in FEATURE_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = df[col].fillna(default)

        return df