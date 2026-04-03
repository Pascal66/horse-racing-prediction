
def kelly_method2_corrected(probabilities, odds, kelly_fraction=1.0):
    """
    Calcule les fractions du budget à miser sur chaque cheval en utilisant la méthode de Kelly pour paris multiples simultanés.
    Cette méthode identifie d'abord le sous-ensemble optimal de chevaux sur lesquels parier,
    puis calcule les fractions de mise exactes pour ce sous-ensemble.
    (Basé sur l'algorithme de Thorp/Benter)

    :param probabilities: dict - clé: identifiant cheval, valeur: probabilité estimée (Pi)
    :param odds: dict - clé: identifiant cheval, valeur: rapport (Ri)
    :param kelly_fraction: float - Levier à appliquer (ex: 1.0 pour Full Kelly, 0.5 pour Half-Kelly)
    :return: dict - fractions du budget à miser (Fi), delta (écart espérance gain - espérance perte)
    """
    # Étape 1 : Calculer le score de sélection pour tous les chevaux.
    # Le score Pi - 1/(Ri-1) est la dérivée du log de la croissance du capital.
    # On ne garde que les chevaux avec un score > 0.
    selection_scores = {}
    for horse in probabilities:
        Pi = probabilities[horse]
        # Utilisation de .get() pour éviter KeyError si le nettoyage amont a échoué
        Ri = odds.get(horse)
        if Ri and Ri > 1:
            score = Pi - 1.0 / (Ri - 1.0)
            if score > 0:
                selection_scores[horse] = score

    if not selection_scores:
        return {}, 0.0

    # Trier les chevaux par score décroissant.
    C_set = sorted(selection_scores, key=selection_scores.get, reverse=True)

    # Étape 2 : Déterminer le sous-ensemble optimal de chevaux.
    # On commence avec tous les chevaux à score positif et on retire itérativement
    # le cheval avec le score le plus bas jusqu'à ce que toutes les fractions calculées soient positives.
    while len(C_set) > 0:
        sum_p = sum(probabilities[h] for h in C_set)
        sum_d = sum(1.0 / (odds[h] - 1.0) for h in C_set)

        if sum_d == 0: # Évite la division par zéro
             C_set.pop()
             continue

        # C_hat est une constante pour le calcul des fractions pour un ensemble donné de chevaux
        C_hat = (sum_p - 1.0) / sum_d + 1.0

        # La condition pour une fraction positive est Pi*Ri > C_hat
        # Si le cheval avec le score le plus bas (le dernier de la liste triée) a une fraction positive,
        # alors tous les autres en auront une aussi.
        last_horse = C_set[-1]
        if probabilities[last_horse] * odds[last_horse] > C_hat:
            # Le sous-ensemble est optimal, on peut sortir.
            break
        else:
            # Le dernier cheval a une fraction négative, on le retire et on recommence.
            C_set.pop()

    if not C_set:
        return {}, 0.0

    # Étape 3 : Calculer les fractions finales pour le sous-ensemble optimal C_set
    fractions = {}
    for horse_id in C_set:
        Pi = probabilities[horse_id]
        Ri = odds[horse_id]
        # Formule de la fraction : Fi = (Pi * Ri - C_hat) / (Ri - 1)
        Fi = (Pi * Ri - C_hat) / (Ri - 1.0)
        fractions[horse_id] = Fi * kelly_fraction # Application du levier

    delta = calculate_delta(fractions, probabilities, odds)

    return fractions, delta


def calculate_delta(fractions, probabilities, odds):
    """
    Calcule l'écart entre espérance de gain totale et espérance de perte totale.

    :param fractions: dict - fractions du budget à miser (Fi)
    :param probabilities: dict - probabilités estimées (Pi)
    :param odds: dict - rapports (Ri)
    :return: float - delta (écart espérance gain - espérance perte)
    """
    if not fractions:
        return 0.0

    esperance_gain = 0.0
    esperance_perte = 0.0

    for horse, Fi in fractions.items():
        Pi = probabilities[horse]
        Ri = odds[horse]

        # Espérance de gain pour ce cheval : Pi * (Fi * (Ri-1))
        gain_cheval = Pi * (Fi * (Ri - 1.0))
        esperance_gain += gain_cheval

        # Espérance de perte pour ce cheval : (1-Pi) * Fi
        perte_cheval = (1.0 - Pi) * Fi
        esperance_perte += perte_cheval

    # Delta = Espérance de gain totale - Espérance de perte totale
    delta = esperance_gain - esperance_perte
    return delta


def calculate_esperance_details(fractions, probabilities, odds):
    """
    Calcule les détails de l'espérance de gain et de perte pour chaque cheval.

    :param fractions: dict - fractions du budget à miser (Fi)
    :param probabilities: dict - probabilités estimées (Pi)
    :param odds: dict - rapports (Ri)
    :return: dict - détails par cheval et totaux
    """
    details = {
        'par_cheval': {},
        'total_gain': 0.0,
        'total_perte': 0.0,
        'delta': 0.0
    }

    if not fractions:
        return details

    for horse, Fi in fractions.items():
        Pi = probabilities[horse]
        Ri = odds[horse]

        # Calcul des valeurs pour ce cheval
        gain_cheval = Pi * (Fi * (Ri - 1.0))
        perte_cheval = (1.0 - Pi) * Fi
        delta_cheval = gain_cheval - perte_cheval

        # Stockage des détails
        details['par_cheval'][horse] = {
            'fraction': Fi,
            'gain': gain_cheval,
            'perte': perte_cheval,
            'delta': delta_cheval
        }

        # Accumulation des totaux
        details['total_gain'] += gain_cheval
        details['total_perte'] += perte_cheval

    details['delta'] = details['total_gain'] - details['total_perte']
    return details


def kelly_dutching_strategy(probabilities, odds, bankroll_fraction=1.0):
    """
    Stratégie optimale pour le "Simple Gagnant" (événements mutuellement exclusifs).
    Cette méthode (Dutching + Kelly) est mathématiquement supérieure à la méthode 2
    pour une course unique où un seul cheval peut gagner.

    :param probabilities: dict - probabilités estimées (Pi)
    :param odds: dict - rapports (Ri)
    :param bankroll_fraction: float - Levier (ex: 0.5 pour Half-Kelly)
    :return: dict - fractions du budget à miser
    """
    # 1. Identifier les chevaux avec une espérance positive (Value Bet)
    selection = [h for h in probabilities if probabilities[h] * odds.get(h, 0) > 1.0]

    if not selection:
        return {}, 0.0

    # 2. Calculer les métriques du "Dutching" (Pari combiné)
    # Probabilité que L'UN des chevaux sélectionnés gagne
    total_prob_winning = sum(probabilities[h] for h in selection)

    # Cote équivalente du portefeuille (si on répartit les mises pour un gain constant)
    sum_inv_odds = sum(1.0 / odds[h] for h in selection)
    if sum_inv_odds == 0: return {}, 0.0
    effective_odds = 1.0 / sum_inv_odds

    # 3. Appliquer Kelly sur ce "Super Pari"
    # f = (p * o - 1) / (o - 1)
    # Ici p = total_prob_winning, o = effective_odds
    if effective_odds <= 1: return {}, 0.0 # Arbitrage négatif ou nul

    kelly_fraction_total = (total_prob_winning * effective_odds - 1.0) / (effective_odds - 1.0)

    # Application du levier (Half-Kelly, etc.)
    total_stake = max(0.0, kelly_fraction_total * bankroll_fraction)

    if total_stake <= 0:
        return {}, 0.0

    # 4. Répartir la mise totale pour garantir l'équité des gains (Dutching)
    fractions = {}
    for horse in selection:
        # La part de chaque cheval est inversement proportionnelle à sa cote
        share = (1.0 / odds[horse]) / sum_inv_odds
        fractions[horse] = total_stake * share

    delta = calculate_delta(fractions, probabilities, odds)
    return fractions, delta

# Exemple d'utilisation avec les données de l'article
if __name__ == "__main__":
    # Données : probabilités aléatoires et rapports (cotes)
    probabilities = { #Pi
        1: 0.10, #0.05,
        2: 0.15,
        3: 0.05, #0.08,
        4: 0.12, #0.05,
        5: 0.08, #0.10,
        6: 0.07, #0.08,
        7: 0.13, #0.01,
        8: 0.30, #0.02
    }

    odds = { #Ri
        1: 84, #31.0,
        2: 59, #5.3,
        3: 118, #12.0,
        4: 9.6, #7.3,
        5: 59, #7.9,
        6: 21, #20.0,
        7: 2.7, #18.0,
        8: 1.6, #9.6
    }

    print("=== STRATÉGIE 1 : KELLY MULTI-PARIS (APPROXIMATION) ===")
    fractions_m2, delta_m2 = kelly_method2_corrected(probabilities, odds, kelly_fraction=0.5)

    # Affichage des résultats
    if fractions_m2:
        total_bet = sum(fractions_m2.values())
        print("Répartition (Méthode 2 - Thorp) :")
        print("-" * 60)

        # Affichage par cheval
        for horse in sorted(fractions_m2.keys()):
            Fi = fractions_m2[horse]
            print(f"  Cheval {horse}: {Fi:.2%} (soit {Fi * 100:.2f}€ pour un budget de 100€)")

        print("-" * 60)
        print(f"Total misé: {total_bet:.2%} (soit {total_bet * 100:.2f}€)")
        print(f"Delta (E[gain] - E[perte]): {delta_m2:.4f}")
        print("\nDétails des calculs d'espérance :")
        print("-" * 60)

        # Calcul des détails
        details = calculate_esperance_details(fractions_m2, probabilities, odds)

        # Affichage des détails par cheval
        for horse, data in sorted(details['par_cheval'].items()):
            print(f"Cheval {horse}:")
            print(f"  Fraction: {data['fraction']:.4f}")
            print(
                f"  E[gain]: {data['gain']:.6f} = {probabilities[horse]} * ({data['fraction']:.4f} * ({odds[horse]}-1))")
            print(f"  E[perte]: {data['perte']:.6f} = (1-{probabilities[horse]}) * {data['fraction']:.4f}")
            print(f"  Delta cheval: {data['delta']:.6f}")
            print()

        print("-" * 60)
        print(f"TOTAL E[gain]: {details['total_gain']:.6f}")
        print(f"TOTAL E[perte]: {details['total_perte']:.6f}")
        print(f"DELTA TOTAL: {details['delta']:.6f}")

        # Calcul de l'espérance de gain net par euro misé et pour 100€
        esperance_par_euro = delta_m2 / total_bet if total_bet > 0 else 0.0
        esperance_100euros = delta_m2 * 100.0

        print("\nInterprétation :")
        print(f"Pour 1€ misé, on peut espérer un gain net de {esperance_par_euro:.4f}€")
        print(f"Pour un budget de 100€ (mise totale: {total_bet * 100:.2f}€), ")
        print(f"l'espérance de gain net est de {esperance_100euros:.2f}€")

    else:
        print("Aucun cheval ne répond aux critères de la formule de Kelly.")

    print("\n" + "="*60 + "\n")

    print("=== STRATÉGIE 2 : KELLY DUTCHING (OPTIMAL POUR SIMPLE GAGNANT) ===")
    fractions_dutch, delta_dutch = kelly_dutching_strategy(probabilities, odds, bankroll_fraction=0.5)

    if fractions_dutch:
        total_bet_d = sum(fractions_dutch.values())
        print("Répartition (Dutching) :")
        for horse in sorted(fractions_dutch.keys()):
            Fi = fractions_dutch[horse]
            print(f"  Cheval {horse}: {Fi:.2%} (soit {Fi * 100:.2f}€)")
        
        print(f"Total misé: {total_bet_d:.2%}")
        print(f"Delta Total: {delta_dutch:.4f}")
        print("Note: Cette méthode garantit un profit égal quel que soit le gagnant parmi la sélection.")
    else:
        print("Pas d'opportunité de Dutching rentable.")