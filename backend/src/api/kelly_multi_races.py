from .Kelly_Formula import kelly_method2_corrected, kelly_dutching_strategy
import pandas as pd

def analyze_multiple_races(df: pd.DataFrame, bankroll=1000.0, kelly_fraction=0.5):
    """
    Analyse plusieurs courses avec la méthode de Kelly à partir d'un DataFrame de participants.
    Utilise 'live_odds' et 'win_probability'.

    :param df: pd.DataFrame - colonnes attendues: ['race_id', 'program_number', 'win_probability', 'live_odds']
    :param bankroll: float - capital total disponible
    :param kelly_fraction: float - facteur Kelly (ex: 0.5 pour Half Kelly)
    :return: dict - rapport complet des courses et allocation optimale du bankroll
    """

    # Groupement par course
    races_groups = df.groupby('race_id')
    
    report = {
        'courses': {},
        'ranking': [],
        'total_positive_delta': 0.0,
        'bankroll_allocation': {},
    }

    for race_id, group in races_groups:
        # Préparation des dictionnaires pour Kelly
        # On utilise le program_number comme identifiant de cheval dans la course
        probabilities = group.set_index('program_number')['win_probability'].to_dict()
        odds = group.set_index('program_number')['live_odds'].to_dict()

        # Nettoyage synchronisé : on ne garde que si les deux sont valides et > 0
        common_keys = set(probabilities.keys()) & set(odds.keys())
        probabilities = {k: v for k, v in probabilities.items() if k in common_keys and pd.notnull(v) and v > 0}
        odds = {k: max(v, 1.01) for k, v in odds.items() if k in common_keys and pd.notnull(v) and v > 0}

        if not probabilities or not odds:
            continue

        # Déterminer automatiquement la méthode adaptée :
        # Si la somme des probabilités ≈ 1 → simple gagnant → Dutching.
        total_prob = sum(probabilities.values())
        
        # Note: Dans un contexte de prédiction ML, la somme peut ne pas être exactement 1 
        # mais devrait être proche si le modèle est bien calibré par course.
        if 0.90 <= total_prob <= 1.10:
            fractions, delta = kelly_dutching_strategy(probabilities, odds, bankroll_fraction=kelly_fraction)
            method_used = 'Dutching'
        else:
            fractions, delta = kelly_method2_corrected(probabilities, odds, kelly_fraction)
            method_used = 'Thorp/Benter'

        total_bet_fraction = sum(fractions.values()) if fractions else 0.0
        # expected_return: delta par rapport à la fraction mise
        expected_return = delta / total_bet_fraction if total_bet_fraction > 0 else 0.0

        if delta > 0:
            report['courses'][race_id] = {
                'fractions': fractions, # {program_number: fraction_du_bankroll_alloué_à_cette_course}
                'delta': delta,
                'total_bet_fraction': total_bet_fraction,
                'expected_return': expected_return,
                'method': method_used,
                'race_id': race_id
            }

    # Filtrer les courses profitables (delta > 0)
    profitable_ids = list(report['courses'].keys())

    if not profitable_ids:
        return report

    total_positive_delta = sum(info['delta'] for info in report['courses'].values())
    report['total_positive_delta'] = total_positive_delta

    # Classement des courses par rentabilité relative (delta / mise)
    report['ranking'] = sorted(
        report['courses'].values(),
        key=lambda x: x['expected_return'],
        reverse=True
    )

    # Allocation du capital (Kelly global / Diversification)
    # On distribue le bankroll proportionnellement au delta de chaque course
    for race_id, info in report['courses'].items():
        weight = info['delta'] / total_positive_delta
        allocated_budget = bankroll * weight
        report['bankroll_allocation'][race_id] = {
            'weight': weight,
            'allocated': allocated_budget
        }

    return report
