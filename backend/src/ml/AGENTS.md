La règle à retenir pour la suite
Toute feature qui doit exister à l'inférence doit être calculée dans un transformer du Pipeline. @loader.py enrichit les
données d'entraînement avec des agrégats historiques — mais à l'inférence l'orchestrateur doit fournir ces colonnes ou
le Pipeline doit avoir un fallback gracieux. Pour les colonnes loader comme avg_speed_last_3 et hist_pct_clean_runs,
deux options : soit l'orchestrateur les calcule avant d'appeler predict_race (en faisant une mini-requête SQL sur
l'historique du cheval), soit PmuFeatureEngineer a des valeurs par défaut comme ci-dessus. La deuxième option est plus
simple mais perd de l'information à l'inférence.

La règle générale pour les pickles sklearn
Toute classe sérialisée dans un .pkl doit être importable depuis un chemin de module stable au moment du chargement. Les
classes définies dans __main__ ou dans un script lancé directement ne survivent pas au rechargement. Le bon réflexe : si
une classe finit dans un Pipeline sauvegardé, elle appartient à src/ml/ (ex: RaceContextEncoder dans features.py).

Règles de Training et Évaluation (Crucial pour éviter les résultats aberrants)
1. ÉVALUATION OUT-OF-SAMPLE : L'évaluation finale des performances (ROI, Win Rate) doit TOUJOURS se faire sur le jeu de test (`test_df`), jamais sur le jeu d'entraînement. Utiliser `train_df` pour les métriques de décision métier conduit à un overfitting massif (ex: ROI > 50% illogique).
2. DÉFINITION WIN RATE : Le `win_rate` doit représenter le pourcentage de courses où le cheval ayant la plus haute probabilité a gagné (nb_gagnants / nb_courses * 100). Ne pas le confondre avec le profit moyen par pari.
3. CONSISTENCE DES FEATURES : Pour éviter les `UserWarning` de LGBM/XGBoost sur les noms de colonnes, maintenir l'objet `X` en tant que `pd.DataFrame` tout au long du Pipeline, y compris dans les boucles de cross-validation (OOF).
4. ARCHITECTURE PIPELINE : Intégrer `RaceContextEncoder` (dans `features.py`) directement dans le Pipeline Scikit-learn. Cela permet à `predictor.py` de traiter les données brutes sans avoir à recalculer manuellement les rangs et moyennes par course (market sentiment).
5. TARGETS : Toujours entraîner une target "global" avant de passer aux disciplines spécifiques pour avoir un modèle de repli robuste.

src/ml/
  models.py        ← HyperStackModel (partagé, pas de dépendances ML circulaires)
  tabnet_utils.py  ← TabNetEnsembleWrapper
  features.py      ← PmuFeatureEngineer, RaceContextEncoder
  safe_loader.py   ← safe_load (dépend de tous les précédents, importé en dernier)
  loader.py        ← DataLoader
  predictor.py     ← RacePredictor (importe safe_loader + models)
  trainer_hyperstack.py  ← importe safe_loader + models
  trainer_tabnet_gpu.py  ← importe tabnet_utils + features