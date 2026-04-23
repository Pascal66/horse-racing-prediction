# src/ml/feature_config.py
"""
Source unique de vérité pour toutes les features du pipeline.
Modifier ici se propage automatiquement aux trainers et au predictor.
"""

CATEGORICAL_FEATURES = [
    'meeting_code', 'discipline', 'track_type', 'sex',
    'shoeing_status', 'jockey_name', 'trainer_name', 'terrain_label', 'meeting_type',
    'breed', 'color', 'blinkers', 'allure',
    'father_name', 'mother_name', 'maternal_grandfather_name'
]

NUMERICAL_FEATURES = [
    # Cheval — carrière
    'horse_age_at_race', 'career_winnings', 'career_races_count',
    'relative_winnings',
    'winnings_rank_in_race', 'odds_rank_in_race', 'reference_odds',
    'is_debutant',
    'career_wins_count', 'career_places_count', 'career_places_2nd_count', 'career_places_3rd_count',
    'winnings_victory', 'winnings_place', 'winnings_year_now', 'winnings_year_prev',
    'career_win_rate', 'career_place_rate',
    # Course
    'distance_m', 'declared_runners_count', 'race_month', 'penetrometer', 'weather_windspeed',
    'handicap_value', 'handicap_weight', 'mount_weight',
    # Historique
    'hist_avg_speed', 'hist_earnings', 'hist_pct_clean_runs',
    # Forme récente
    'avg_speed_last_3', 'days_since_last_race', 'speed_form_ratio',
    # Marché
    'odds_log', 'odds_drift_ratio', 'odds_drift_log',
    # Discipline / distance
    'pct_races_on_discipline', 'pct_clean_on_discipline',
    'distance_delta', 'is_unusual_distance', 'is_specialist',
    # Duo jockey×cheval
    'duo_total_races', 'duo_win_rate', 'duo_avg_rank',
    'duo_best_rank', 'duo_confidence', 'duo_is_experienced',
    'national_speed_ratio', 'national_experience_rate',
]

CONTEXTUAL_FEATURES = [
    # Générées par RaceContextEncoder — ne pas mettre dans numerical_features
       'market_sentiment',
       'reference_odds_rel_race', 'reference_odds_rank_race',
        'hist_avg_speed_z_race',
        'career_winnings_rank_race',
        'avg_speed_last_3_z_race',
        'days_since_last_race_z_race',
        'duo_win_rate_z_race', 'duo_avg_rank_z_race', 'duo_confidence_z_race',
]

EXTRA_FEATURES = [
    'proba_tabnet',  'ltr_proba', # optionnel — présent seulement si modèle TabNet et/ou ltr sont disponibles
]

# Valeurs par défaut à l'inférence quand une colonne est absente
# "neutre" = ne donne aucun avantage ni désavantage
FEATURE_DEFAULTS = {
    # Carrière
    'horse_age_at_race':        5,
    'career_winnings':          0,
    'career_races_count':       0,
    'relative_winnings':        1.0,
    'winnings_rank_in_race':    5,
    'odds_rank_in_race':        5,
    'reference_odds':           10.0,
    'is_debutant':              0,
    'career_wins_count':        0,
    'career_places_count':      0,
    'career_places_2nd_count':  0,
    'career_places_3rd_count':  0,
    'winnings_victory':         0,
    'winnings_place':           0,
    'winnings_year_now':        0,
    'winnings_year_prev':       0,
    'career_win_rate':          0.0,
    'career_place_rate':        0.0,
    # Course
    'distance_m':               2000,
    'declared_runners_count':   10,
    'race_month':               6,
    'penetrometer':             3.5,
    'handicap_value':           0,
    'handicap_weight':          0,
    'mount_weight':             0,
    # Historique
    'hist_avg_speed':           1.20,
    'hist_earnings':            0,
    'hist_pct_clean_runs':      1.0,
    # Forme récente
    'avg_speed_last_3':         0,
    'days_since_last_race':     365,
    'speed_form_ratio':         1.0,
    # Marché
    'odds_log':                 2.3,
    'odds_drift_ratio':         1.0,
    'odds_drift_log':           0.0,
    # Discipline
    'pct_races_on_discipline':  0.0,
    'pct_clean_on_discipline':  1.0,
    'distance_delta':           0,
    'is_unusual_distance':      0,
    'is_specialist':            0,
    # Duo
    'duo_total_races':          0,
    'duo_win_rate':             0.0,
    'duo_avg_rank':             99,
    'duo_best_rank':            99,
    'duo_confidence':           0.0,
    'duo_is_experienced':       0,
    # Contextuel
    'market_sentiment':         0.1,
    'reference_odds_rel_race':  0.0,
    'reference_odds_rank_race': 0.5,
    'hist_avg_speed_z_race':    0.0,
    'career_winnings_rank_race':0.5,
    'duo_win_rate_z_race':      0.0,
    'duo_avg_rank_z_race':      0.0,
    'duo_confidence_z_race':    0.0,
    # Optionnel
    'proba_tabnet':             0.5,
    "proba_ltr":                0.5,
    'national_speed_ratio':     1.0,
    'national_experience_rate': 0.0,
}