import pandas as pd
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn(verbose=False)  # avant tous les imports sklearn
from sklearn.base import BaseEstimator, TransformerMixin

pd.set_option('future.no_silent_downcasting', True)

import logging
logger = logging.getLogger('sklearnex')
logger.setLevel(logging.WARNING)

import os
os.environ["SKLEARNEX_VERBOSE"] = "WARNING"

def build_ltr_target(df: pd.DataFrame) -> pd.Series:
    """
    Construit la target ordinale LTR depuis finish_rank.
    Utilisé uniquement pendant l'entraînement.
    0 = non placé (6ème et au-delà)
    1 = 5ème, 2 = 4ème, 3 = 3ème, 4 = 2ème, 5 = 1er
    """
    def rank_to_score(r):
        if pd.isna(r): return 0
        try:
            r = int(float(r))
            mapping = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
            return mapping.get(r, 0)
        except (ValueError, TypeError):
            return 0

    return df['finish_rank'].apply(rank_to_score).astype(int)

class RaceContextEncoder(BaseEstimator, TransformerMixin):
    """
    Transforms raw horse features into contextual features relative to the race.
    """
    def __init__(self, group_col="race_id", feature_cols=None):
        self.group_col = group_col
        self.feature_cols = feature_cols or [
            'reference_odds', 'career_winnings', 'horse_age_at_race',
            'hist_avg_speed', 'avg_speed_last_3',
            'days_since_last_race', 'hist_pct_clean_runs',
            'duo_win_rate', 'duo_avg_rank', 'duo_confidence'
        ]

    def fit(self, X, y=None): return self

    def transform(self, X):
        df = X.copy()
        if self.group_col not in df.columns: return df

        grouped = df.groupby(self.group_col)
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = np.nan
            
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
        if 'reference_odds' in df.columns:
            self.learned_stats_['reference_odds'] = df['reference_odds'].mean()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies transformations without data leakage."""
        df = X.copy()
        
        # 1. Dates & Fraîcheur
        if 'program_date' in df.columns:
            df['program_date'] = pd.to_datetime(df['program_date'])
            df['race_month'] = df['program_date'].dt.month
            
            if 'birth_year' in df.columns:
                df['horse_age_at_race'] = df['program_date'].dt.year - df['birth_year']
                df['horse_age_at_race'] = df['horse_age_at_race'].fillna(df.get('age', 5))

            if 'last_race_date' in df.columns:
                df['last_race_date'] = pd.to_datetime(df['last_race_date'], errors='coerce')
                df['days_since_last_race'] = (df['program_date'] - df['last_race_date']).dt.days.fillna(365).clip(lower=0)

        # 2. Forme & Affinité
        if 'avg_speed_last_3' in df.columns and 'hist_avg_speed' in df.columns:
            df['speed_form_ratio'] = (df['avg_speed_last_3'] / df['hist_avg_speed'].replace(0, np.nan)).fillna(1.0)
        
        if 'distance_m' in df.columns and 'hist_avg_distance' in df.columns:
            df['distance_delta'] = (df['distance_m'] - df['hist_avg_distance']).fillna(0).abs()
            df['is_unusual_distance'] = (df['distance_delta'] > 400).astype(int)

        if 'pct_races_on_discipline' in df.columns:
            df['is_specialist'] = (df['pct_races_on_discipline'] > 0.7).astype(int)

        # 3. Market Drift
#        if 'live_odds' in df.columns and 'reference_odds' in df.columns:
#            live = pd.to_numeric(df['live_odds'], errors='coerce')
#            ref = pd.to_numeric(df['reference_odds'], errors='coerce')
#            df['odds_drift_ratio'] = (live / ref.replace(0, np.nan)).fillna(1.0)
#            df['odds_drift_log'] = np.log(df['odds_drift_ratio'].clip(lower=0.1))
        
        if 'reference_odds' in df.columns:
            df['reference_odds_log'] = np.log1p(pd.to_numeric(df['reference_odds'], errors='coerce').fillna(20))
            if 'race_id' in df.columns:
                df['odds_rank_in_race'] = df.groupby('race_id')['reference_odds'].rank(ascending=True, method='min')

        # 4. Carrière & Duo
        if 'career_winnings' in df.columns:
            df['is_debutant'] = (df['career_races_count'].fillna(0) == 0).astype(int)
            if 'race_id' in df.columns:
                # Is this the richest horse in the race?
                df['winnings_rank_in_race'] = df.groupby('race_id')['career_winnings'].rank(ascending=False, method='min')
                race_avg_e = df.groupby('race_id')['career_winnings'].transform('mean')
                df['relative_winnings'] = df['career_winnings'] / (race_avg_e + 1)

        # E. Duo jockey×cheval - Ensure all duo_ features are present
        if 'duo_total_races' in df.columns:
            df['duo_confidence'] = np.log1p(df['duo_total_races'].fillna(0)) * df['duo_win_rate'].fillna(0)
            df['duo_is_experienced'] = (df['duo_total_races'].fillna(0) >= 5).astype(int)

        # Progression de niveau : le cheval monte-t-il en compétition ?
        if 'n_races_national' in df.columns and 'hist_races' in df.columns:
            # Proportion de courses nationales dans la carrière
            df['national_experience_rate'] = (
                    df['n_races_national'].fillna(0) /
                    df['hist_races'].replace(0, 1)
            )

            # Flag : cheval en montée de niveau (peu d'expérience nationale
            # mais bonnes perfs en régional)
            df['is_rising_horse'] = (
                    (df['national_experience_rate'] < 0.3) &  # peu de nationales
                    (df['hist_pct_clean_runs'] > 0.8)  # fiable en général
            ).astype(int)

        # Imputation finale Neutre (Source feature_config)
        from src.ml.feature_config import FEATURE_DEFAULTS, CATEGORICAL_FEATURES
        for col in CATEGORICAL_FEATURES:
            if col in df.columns: df[col] = df[col].fillna(self.cat_fill_value_).astype(str)
        
        for col, default in FEATURE_DEFAULTS.items():
            if col not in df.columns: df[col] = default
            else: df[col] = df[col].fillna(default)
            
        return df
