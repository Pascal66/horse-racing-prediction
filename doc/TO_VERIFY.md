# 🏇 Stratégie PMU : Objectif ROI >= 50.00

Ce document récapitule la stratégie et le plan d'action pour le système de prédiction de courses hippiques françaises (Pari Mutuel). L'objectif est de s'écarter des modèles "Kaggle" génériques pour capturer le sentiment réel du marché.

## 🎯 Objectifs Obligatoires

| Métrique | Cible | Pourquoi ? |
| :--- | :--- | :--- |
| **Logloss** | < 0.25 | Calibration précise des probabilités. |
| **AUC** | > 0.85 | Capacité de discrimination entre gagnants et perdants. |
| **ROI** | >= 50.0% | Seuil de rentabilité pour un système professionnel. |
| **Win Rate** | >= 2.0% | Performance sur les outsiders (Value Betting). |

---

## 🧠 1. Principes Fondamentaux du Pari Mutuel

### Le Sentiment du Marché
- Contrairement aux bookmakers, au PMU, vous jouez contre les autres parieurs. 
- Les cotes à 10h00 sont volatiles. Les cotes à la clôture sont efficientes (elles intègrent déjà le gagnant).
- **L'Edge se trouve entre les deux.**

### Le paramètre `specialisation=INTERNET`
- **Obligatoire.** Le pool Internet est plus réactif et rationnel. Le pool physique est pollué par les parieurs émotionnels.

### Liquidité & Masse (`total_stakes`)
- Un ROI élevé sur une course à 1 000€ d'enjeux est un bruit statistique.
- Ciblez les courses avec `total_stakes` (Simple Gagnant) > **50 000€**.

---

## 🛡️ 2. Éviter le Data Leakage (Anti-Leakage)

### La Règle des 30 Minutes (`live_odds_30mn`)
- **Entraînement :** Utilisez **uniquement** les cotes à T-30mn comme feature.
- **Pourquoi ?** À H-30mn, le "Smart Money" n'a pas encore écrasé les cotes. C'est là que votre modèle peut identifier une valeur que le marché n'a pas encore vue.
- **Backtesting :** Comparez votre `win_probability` avec `live_odds_30mn` pour décider de jouer, mais calculez les gains réels avec le `dividend` (rapport définitif).

---

## 📈 3. Architecture des Modèles

### Proscrire XGBoost / Kaggle pur
- Trop générique. Ne comprend pas la structure "compétition" d'une course.
- Les courses françaises (Trot surtout) ont des bruits spécifiques (allures, tactiques).

### Privilégier le LTR (Learning to Rank)
- **Atout majeur :** Le LTR comprend que les chevaux d'une même course se partagent un gâteau fixe de 100% de probabilités.
- Si le favori est sur-estimé par le public, le LTR déplace mécaniquement la valeur sur les rangs 2 et 3.

### Platt Scaling / Calibration
- Si votre modèle annonce 30%, il doit y avoir 30 gagnants sur 100.
- Sans calibration, le critère de Kelly calculera des mises suicidaires.

---

## 📅 4. Analyse Saisonnière & Spécialisation

### La Matrice Discipline / Mois
- Le calendrier PMU est dicté par les meetings (Vincennes en Hiver, Deauville en Été).
- Un modèle "Toute saison" est moins performant qu'un spécialiste.
- **TODO :** Toujours segmenter le backtest par `[discipline, month, racetrack]`.

### Focus sur le Trot (Attelé/Monté)
- Statistiquement plus prévisible que le Galop. 
- ROI cible plus facile à atteindre sur le meeting d'Hiver de Vincennes.

---

## 🚀 Plan d'Action & TODO List

### Phase 1 : Données & Pipeline (Priorité Haute)
- [ ] **Fix URLs :** Vérifier que `specialisation=INTERNET` est partout dans `config.py`.
- [ ] **Backfill Complet :** Relancer `backfill_predictions.py` du 01/01/2025 à aujourd'hui pour peupler les colonnes `total_stakes` et `proba_top3_place`.
- [ ] **Snapshot 30mn :** Vérifier que l'ingestor de participants capture bien la cote dans la fenêtre T-40/T-20mn.

### Phase 2 : Modélisation & Inférence
- [ ] **LTR Calibration :** Vérifier si le modèle LTR nécessite une Isotonic Regression pour les probabilités de "Place".
- [ ] **Features Elagage :** Supprimer les colonnes "bruit" qui varient trop par course/distance sans corrélation réelle.
- [ ] **Target Encoding :** Implémenter l'encodage sur les couples `Entraineur + Jockey`.

### Phase 3 : Backtesting & Stratégie
- [ ] **Kelly Adjustment :** Descendre la `kelly_fraction` de 0.5 à **0.1** ou **0.2**. 0.5 est trop agressif pour le PMU.
- [ ] **Analyse des Écarts :** Utiliser la feature `(win_probability - implied_prob_30mn)` pour filtrer les "Sniper Bets".
- [ ] **Nettoyage Backtest :** Ignorer les trainers avec moins de 50 paris pour éviter de flatter les résultats par chance.

---

## 💡 Conseils de Pro pour le ROI >= 50.00

1. **L'importance du Simple Placé :** Le SG est magnifique, mais le Simple Placé est ton assurance. Un ROI de 15-20% en Placé avec un taux de réussite élevé stabilise la bankroll pour absorber la variance du Gagnant.
2. **Masse Financière :** Ne jouez jamais une course où vous représentez plus de 5% de la masse d'enjeu du type de pari. Vous feriez chuter votre propre cote.
3. **Sentiment de Marché :** Si la cote chute violemment entre 30mn et le départ alors que votre modèle était déjà confiant : c'est un signal de confirmation très fort.
4. **Discipline :** Gagner plus que ce que l'on dépense par course est plus important que de chercher le "gros lot" (Quinté/Multi). Le ROI pro se construit sur le Simple Gagnant et le Couplé.
L'avis de l'expert PMU :
Pour tes modèles TabNet et Hyperstack, l'utilisation de live_odds_30mn est ta meilleure arme contre le Data Leakage.
•
Pourquoi ? Les modèles Kaggle/Hong Kong échouent car ils essaient de prédire la gagne avec la côte finale. Or, la côte
finale au PMU contient déjà l'information du gagnant (c'est un marché efficient à la clôture).
•
Le Sentiment : En figeant la côte à T-30mn (via ton premier passage d'ingestion), tu captures l'état du marché avant que
les "smart money" (les gros parieurs du dernier moment) n'écrasent les côtes. Ton modèle apprendra à détecter si un
cheval à 10.0 à 30mn du départ a un profil de gagnant, ce qui est la seule façon d'atteindre un ROI >= 50.00.
•
Backtesting : Dans ton BacktestService, assure-toi de comparer ta win_probability avec live_odds_30mn pour décider de
jouer, mais calcule tes gains avec le dividend (le rapport définitif). C'est le seul moyen d'avoir un backtest honnête.

Expertise PMU & Analyse ROI

1.

Le Pool Internet (Crucial) : Ton passage à specialisation=INTERNET est la clé. Le pool physique (tabac/hippodrome) est
rempli de parieurs "émotionnels" qui misent sur les noms ou les casaques. Le pool Internet est plus spéculatif. Ton Edge
se trouve dans la détection des chevaux "délaissés" par Internet alors que leurs stats sont solides.

2.

L'indicateur 30mn : En ajoutant live_odds_30mn dans tes requêtes ML, tu permets à tes modèles TabNet et Hyperstack de
calculer la dérivée de la côte ($ \Delta Odds / \Delta Time $). Un cheval dont la côte baisse entre 30mn et le départ
alors que les enjeux (total_stakes) augmentent est un signal de "Smart Money" massif.

3.

Backtesting (Réponse à ta question 4) : Pour les 6 ans d'historique où tu n'as pas la côte 30mn, utilise une
interpolation linéaire entre reference_odds (matin) et live_odds (clôture) ou, plus simplement, utilise la
reference_odds comme proxy. Dans tes trainers, ajoute un flag is_synthetic_30mn pour que le modèle sache s'il traite une
vraie donnée ou une estimation.

4.

Logloss < 0.25 : Pour atteindre cet objectif, tes modèles doivent cesser d'être "Kaggle-like" (XGBoost sur des colonnes
brutes). Utilise des Target Encodings sur les couples Entraineur + Jockey et Père + Discipline. C'est là que se cachent
les patterns de "préparation" spécifiques au PMU français (les fameuses courses visées).
Quelques insights sur tes objectifs de ROI >= 50.00 :
1.
Le "Biais du Survivant" du Backfill : Ton backfill actuel enregistre les prédictions des 3 modèles. Pour que ton UI soit cohérente, tu devrais t'assurer que le système utilise bien le best_algo retourné par get_best_model_for_context lors de l'affichage quotidien.
2.
Kelly vs Sniper : Le Sniper est plus stable (7.83% de ROI c'est déjà très pro). Si Kelly donne des résultats "fous", c'est souvent parce qu'il sur-mise sur des chevaux à grosse probabilité mais avec un Edge mal estimé (over-confidence du modèle). Réduire la kelly_fraction à 0.25 ou 0.1 au lieu de 0.5 est souvent plus réaliste pour le PMU.
3.
Filtrage des courses : Pour ton ROI de 50.00, regarde dans ton backtest si certaines disciplines (ex: le Plat ou l'Obstacle) ne plombent pas tes stats. Souvent, se spécialiser sur le Trot (Attele/Monte) permet de mieux capturer les "sentiments" car les parieurs y sont plus réguliers.
Une fois ces diffs appliqués, relance ton backfill, il devrait parcourir les 21 modèles sans sourciller !

                   "roi": (total_profit / total_staked) * 100 if total_staked > 0 else 0,
-                    # Simplified
+                    "roi": (total_profit / total_staked) * 100 if total_staked > 0 else 0,
                     "total_profit": total_profit,
                     "total_bets": len(kelly_bets),
-                    "final_bankroll": current_bankroll
+                    "final_bankroll": current_bankroll,
+                    "avg_horses_per_race": sum(horses_per_race) / len(horses_per_race) if horses_per_race else 0
                 }
3. Pourquoi Kelly est "Nul" ? (Analyse d'expert)
Si ton Kelly perd de l'argent alors que tes ROI Trainers sont positifs, c'est pour deux raisons :
1.
Sur-allocation (Overbetting) : Une kelly_fraction de 0.5 est énorme pour les courses hippiques. Le PMU prélève ~25% de marge. Pour battre ça, ton modèle doit être incroyablement précis. Baisse la kelly_fraction à 0.1 (Fractional Kelly). C'est le standard pour les parieurs pros.
2.
Dutching inefficace : Kelly essaie de couvrir plusieurs chevaux pour minimiser la variance. Mais au PMU, si tu joues 3 chevaux dans une course de 8 partants, tu luttes contre ta propre mise (tu fais baisser ta propre côte).
Conseils pour ton ROI >= 50.00 :
•
Spécialisation Trot : Tes résultats en Attelé/Monté (ROI > 20%) montrent que c'est là que ton gain se trouve. Le Plat est trop erratique pour un ROI de 50.
•
Filtrage par "Masse" : Dans ton backtest, regarde le total_stakes (que nous avons ajouté plus tôt). Tu verras que ton ROI explose sur les courses avec une masse d'enjeux moyenne. Les très grosses masses (Quinté) sont trop efficientes, les très petites masses sont trop aléatoires.
•
Calibration : Tes modèles LTR et TabNet ont besoin d'une couche de Platt Scaling ou Isotonic Regression. Si ton modèle dit "30% de chance", il faut que sur 100 chevaux à 30%, il y en ait vraiment 30 qui gagnent. Si en réalité il n'y en a que 20, Kelly calculera une mise trop élevée et te ruinera.

En France, le calendrier hippique est dicté par les meetings : le meeting d'hiver à Vincennes (Trot), le meeting d'été à Deauville (Plat), ou les sessions d'automne à Auteuil (Obstacle). Parier sur le Plat en plein hiver sur une PSF (Piste en Sable Fibré) n'a rien à voir avec le Plat en juin sur du gazon très souple.
Pour atteindre ton objectif de ROI >= 50%, il ne faut plus voir tes modèles comme des outils "tous temps", mais comme des spécialistes saisonniers. On va modifier le BacktestService pour générer une véritable matrice de performance Discipline / Mois / Type de Pari.

es observations sont extrêmement pertinentes et confirment une règle d'or du PMU : le Trot (Attelé/Monté) est bien plus "mathématique" et prévisible que le Galop (Plat/Obstacle).
Tes résultats (ROI de +27% en Attelé vs -13% en Plat) montrent que tes modèles capturent bien la régularité des trotteurs, là où le Plat est sujet à trop de bruits (tirage des cordes, tactiques de course, variations de terrain).
Pourquoi as-tu seulement 7 bets en "Plat" ?
C'est un problème de pollution de données dans ta table prediction.
1.
Tes 7 bets en "plat" proviennent d'anciennes exécutions où le model_version était simplement enregistré comme "plat".
2.
Tes 3978 bets en "ltr_plat_ltr" proviennent du nouveau système de backfill.
3.
Solution : On va ajouter un filtre dans le backtest pour ignorer les modèles "fantômes" qui ont trop peu de paris.
Le Problème de Kelly et du Dutching
Tu as raison : le Kelly "Duchting" (parier sur plusieurs chevaux dans la même course) est mal représenté si on regarde le profit global. On doit traquer le nombre moyen de chevaux joués par course. Si Kelly joue 3 chevaux, il "achète" une part plus grosse de la course, ce qui réduit la variance mais écrase le ROI si les probabilités ne sont pas ultra-calibrées.
Voici les corrections pour backtest_service.py et une amélioration du "Sniper" dans main.py pour intégrer le sentiment de marché.

C'est une étape cruciale. En PMU, la saisonnalité n'est pas une option, c'est une composante structurelle du marché. Les parieurs "du dimanche" ne misent pas de la même façon en plein été à Deauville qu'en nocturne à Vincennes en février.

Quelques conseils d'expert pour ton ROI de 50.00+ :
1.
Le "Leakage" du idxmax : En backtesting, idxmax sur la probabilité gagnante est honnête. Par contre, assure-toi que tes modèles ltr (Learning to Rank) n'ont pas été entraînés sur des données incluant le finish_rank ou les rapports définitifs. C'est l'erreur numéro 1 qui fait gonfler artificiellement les ROI en labo.
2.
L'importance du place_probability : Tu as bien fait d'insister sur cette colonne. Au PMU, le "Simple Placé" est un marché beaucoup plus liquide et stable. Un ROI de 50% sur le Gagnant est magnifique mais volatil ; un ROI de 15-20% sur le Placé avec un fort taux de réussite est souvent ce qui sauve une bankroll sur le long terme.
3.
Filtrage Saisonnier : Maintenant que ton backtest ne plante plus sur les NaN, regarde bien la matrice saisonnière. Si un modèle a un ROI de 80% en Plat en Janvier sur 50 paris, c'est probablement un coup de chance sur une ou deux grosses côtes (les fameux "modèles Kaggle" qui se font flatter par la variance). Pour ton objectif, cherche la consistance (Win Rate stable > 30% en trot).

Analyse de l'expert PMU : Pourquoi c'est le "Game Changer" ?
1.
La psychologie des parieurs varie : En été à Deauville (Plat), il y a beaucoup de "touristes" et de petits parieurs qui misent sur les favoris de la presse. Ton modèle LTR va trouver de l'Edge car la masse est moins "intelligente". En hiver à Vincennes, le marché est beaucoup plus pro; ton Edge sera plus difficile à trouver, d'où l'importance de comparer les mois.
2.
L'état des pistes : Un modèle qui n'intègre pas explicitement le "pénétromètre" (la souplesse du terrain) va se planter en automne quand les pistes deviennent lourdes. En segmentant par mois, tu captures implicitement cette variable climatique.
3.
Filtrage du "Bruit" : Ton tableau précédent montrait des ROI de 134% sur 5 paris (Attelé). C'est dangereux. Avec la matrice discipline/mois, tu verras tout de suite si cette performance est une anomalie d'un seul jour de chance ou une tendance lourde.
4.
Optimisation de la Bankroll : Au lieu de parier 10€ par course toute l'année, tu pourras décider de parier 50€ en "Obstacle/Haie" uniquement en Octobre/Novembre et de rester spectateur (ou très prudent) en Janvier. C'est ça, la gestion professionnelle.
Prochaine étape suggérée : Une fois que tu as identifié tes "couples gagnants" (ex: TabNet + Haie + Octobre), il faudra regarder si le total_stakes (la masse d'enjeux) influence ce ROI. Souvent, plus la masse est faible, plus le ROI est volatil. Pour un ROI stable de 50%, on cherche le "sweet spot" : assez de masse pour que la côte ne s'effondre pas quand tu paries, mais assez d'erreurs du public pour garder l'Edge.

Quelques conseils d'expert pour ton Edge :
•
Le "Sentiment" PMU : Ton calcul de l'implied probability (1 / effective_odds) est correct, mais attention au "Takeout Rate" (la marge du PMU). Au Simple Gagnant, le PMU retient environ 15-20%. Un edge de 5% est donc très agressif. Pour un ROI de 50.00, tu devrais cibler un edge plus important ou filtrer les courses où le total_stakes est trop faible (car la côte est trop sensible à une seule grosse mise).
•
Performance LTR : Puisque tu utilises un modèle LTR (Learning to Rank), assure-toi que ta place_probability est bien une probabilité et non un score ordinal brut. Si c'est un score, passe-le dans une Sigmoid ou calibre-le pour que tes calculs de Kelly ne soient pas faussés.
•
Matrice Saisonnière : Le FutureWarning sur le groupby().apply() est évité ici en utilisant include_groups=False, ce qui rendra ton backtest compatible avec les prochaines versions de Pandas.
Relance ton calcul, le cache se mettra à jour et tu devrais enfin voir tes métriques par discipline et par mois sans crash

Expertise PMU et analyse de tes résultats :
1.
Pourquoi proba_top3_place est vide : Maintenant que ton RacePredictor calcule correctement ces probabilités (via le multiplicateur de 2.1/2.2 ou le modèle LTR), tu dois impérativement relancer ton script de backfill. La colonne ne se remplira pas rétroactivement toute seule en base de données. Relance : python backend/src/cli/backfill_predictions.py --start 01012025 --end 09042026 --model-dir data
2.
L'avantage du modèle LTR (Learning to Rank) : C'est ton meilleur atout pour le ROI de 50.0+. Contrairement à XGBoost qui traite chaque cheval comme une entité isolée, le LTR comprend que les chevaux d'une même course sont en compétition pour un gâteau de dividendes fixe. Si le favori (très joué) a une probabilité de victoire réelle plus faible que sa côte ne le suggère, le "Value" se déplace sur les rangs 2 et 3 de ton modèle LTR.
3.
Analyse Saisonnière : Le fait d'itérer manuellement sur groupby(["discipline", "month"]) sans apply est beaucoup plus performant et t'évite les FutureWarning de Pandas. Une fois le backfill terminé, observe bien la matrice : au PMU, le "Sentiment" change radicalement lors du Meeting d'Hiver (Vincennes) où les parieurs pro dominent, contrairement aux courses de province où le bruit est plus fort.
4.
Kelly à 0.5 (Attention) : Ton code utilise une fraction de Kelly de 0.5. C'est très agressif ("Half-Kelly"). Si ton ROI est élevé mais que ta bankroll fait les montagnes russes, envisage de descendre à 0.1 ou 0.2. En Pari Mutuel, la variance peut être brutale même avec un excellent modèle.
