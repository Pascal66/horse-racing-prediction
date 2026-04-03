import logging
import sys
from pathlib import Path

import numpy as np
import joblib
import optuna
import pandas as pd
from category_encoders import CatBoostEncoder

from sklearnex import patch_sklearn

from src.ml.tabnet_bridge import TabNetBridge

patch_sklearn(verbose=False)  # avant tous les imports sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss ,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.core.config import DB_URL
from src.core.database import DatabaseManager
from src.ml.features import PmuFeatureEngineer, RaceContextEncoder
from src.ml.loader import DataLoader
# from src.ml.predictor import HyperStackModel

from src.ml.models import HyperStackModel      # ← plus depuis predictor lui-même
from src.ml.feature_config import (
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    CONTEXTUAL_FEATURES, EXTRA_FEATURES
)

# Set up logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
import os
os.environ["SKLEARNEX_VERBOSE"] = "WARNING"

# ------------------------------------------------
# Base models
# Problème 3 — l'espace de recherche est sous-exploité
# Plusieurs hyperparamètres importants manquent, surtout pour les petits datasets comme Haie/Steeple où la régularisation est critique :# ------------------------------------------------

def build_base_models(trial):
    models = {}

    models["xgb"] = XGBClassifier(
        n_estimators=trial.suggest_int("xgb_n_estimators", 300, 1200),
        max_depth=trial.suggest_int("xgb_max_depth", 3, 7),
        learning_rate=trial.suggest_float("xgb_lr", 0.005, 0.1, log=True),
        subsample=trial.suggest_float("xgb_sub", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("xgb_col", 0.6, 1.0),
        min_child_weight=trial.suggest_int("xgb_min_child", 1, 20),  # nouveau — clé pour petits datasets
        reg_alpha=trial.suggest_float("xgb_alpha", 1e-4, 10.0, log=True),   # nouveau
        reg_lambda=trial.suggest_float("xgb_lambda", 1e-4, 10.0, log=True), # nouveau
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42
    )

    models["lgbm"] = LGBMClassifier(
        n_estimators=trial.suggest_int("lgbm_n_estimators", 300, 1200),
        max_depth=trial.suggest_int("lgbm_max_depth", 3, 8),
        learning_rate=trial.suggest_float("lgbm_lr", 0.005, 0.1, log=True),
        num_leaves=trial.suggest_int("lgbm_leaves", 15, 80),
        min_child_samples=trial.suggest_int("lgbm_min_child", 5, 50),  # nouveau
        reg_alpha=trial.suggest_float("lgbm_alpha", 1e-4, 10.0, log=True),  # nouveau
        reg_lambda=trial.suggest_float("lgbm_lambda", 1e-4, 10.0, log=True), # nouveau
        subsample=trial.suggest_float("lgbm_sub", 0.6, 1.0),                 # nouveau
        colsample_bytree=trial.suggest_float("lgbm_col", 0.6, 1.0),          # nouveau
        n_jobs=-1,
        random_state=42,
        verbosity=-1
    )

    models["cat"] = CatBoostClassifier(
        iterations=trial.suggest_int("cat_iter", 300, 1000),
        depth=trial.suggest_int("cat_depth", 4, 8),
        learning_rate=trial.suggest_float("cat_lr", 0.005, 0.1, log=True),
        l2_leaf_reg=trial.suggest_float("cat_l2", 1.0, 20.0),   # nouveau
        bagging_temperature=trial.suggest_float("cat_bag", 0.0, 1.0),  # nouveau
        verbose=False,
        random_seed=42
    )

    return models

# ------------------------------------------------
# OOF stacking
# ------------------------------------------------

def generate_oof(models, X, y, folds):
    """
    Generates Out-of-Fold predictions.
    We keep X as a DataFrame to preserve feature names for GBM models.
    """
    n_models = len(models)
    oof = np.zeros((len(X), n_models))
    trained_models = []

    # If X is numpy, but we have feature names, convert to DF to avoid warnings
    feature_names = None
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns

    y_vals = y.values if hasattr(y, "values") else y

    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame to maintain feature names.")

    for m_id, (name, model) in enumerate(models.items()):
        preds = np.zeros(len(X))
        fold_models = []
        for train_idx, val_idx in folds:
            # Slicing
            if isinstance(X, pd.DataFrame):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_train, X_val = X[train_idx], X[val_idx]
                if feature_names is not None:
                    X_train = pd.DataFrame(X_train, columns=feature_names)
                    X_val = pd.DataFrame(X_val, columns=feature_names)

            y_train = y_vals[train_idx]

            m = clone(model)
            m.fit(X_train, y_train)

            # Prediction
            preds[val_idx] = m.predict_proba(X_val)[:, 1]
            fold_models.append(m)
            
        oof[:, m_id] = preds
        trained_models.append((name, fold_models))
    return oof, trained_models

# ------------------------------------------------
# Optuna objective
# Problème 1 — Optuna optimise le mauvais critère
# La fonction objective optimise le log_loss OOF du stacker. C'est mathématiquement propre mais déconnecté de l'objectif réel qui est le ROI. Le log_loss cherche des probabilités bien calibrées pour tous les chevaux - alors que le ROI ne dépend que du meilleur cheval par course. C'est pour ça que Haie s'effondre : Optuna trouve des hyperparamètres qui calibrent mieux les probabilités sur 171 courses, mais ces params ne correspondent pas aux chevaux qui gagnent réellement.
# La solution est de remplacer l'objectif par une métrique ROI-proxy calculée sur l'OOF :
# Problème 2 - n_jobs=-1 dans study.optimize est dangereux
# Avec n_jobs=-1, Optuna lance les trials en parallèle - mais chaque trial fait lui-même n_jobs=-1 dans XGBoost/LightGBM.
# Donc CPU × CPU threads qui se battent pour les mêmes cores. En pratique ça ralentit plutôt qu'accélérer.
# Changer en n_jobs=1 dans study.optimize et laisser les modèles individuels utiliser tous les cores
# ------------------------------------------------

def objective(trial, X, y, race_ids, odds):
    y_vals = y.values if hasattr(y, "values") else y
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y_vals))
    models = build_base_models(trial)
    # X here is X_train_df (DataFrame)
    oof, _ = generate_oof(models, X, y, folds)

    # Stacker OOF
    stacker = LogisticRegression(max_iter=2000)
    # losses = []
    # for train_idx, val_idx in folds:
    #     X_train_stack, X_val_stack = oof[train_idx], oof[val_idx]
    #     y_train_stack, y_val_stack = y_vals[train_idx], y_vals[val_idx]
    #     stacker.fit(X_train_stack, y_train_stack)
    #     preds = stacker.predict_proba(X_val_stack)[:, 1]
    #     loss = log_loss(y_val_stack, preds)
    #     losses.append(loss)
    #     trial.report(loss, step=len(losses))
    #     if trial.should_prune():
    #         raise optuna.TrialPruned()
    # return np.mean(losses)
    oof_preds = np.zeros(len(X))
    for train_idx, val_idx in folds:
        stacker.fit(oof[train_idx], y_vals[train_idx])
        oof_preds[val_idx] = stacker.predict_proba(oof[val_idx])[:, 1]

    # ROI OOF — même logique que _calculate_metrics
    df_oof = pd.DataFrame({
        'proba': oof_preds,
        'is_winner': y_vals,
        'race_id': race_ids.values,
        'effective_odds': odds.values
    })
    total_return, total_bets = 0, 0
    for _, race_group in df_oof.groupby('race_id'):
        best = race_group.loc[race_group['proba'].idxmax()]
        if best['is_winner'] == 1:
            total_return += best['effective_odds']
        total_bets += 1

    roi = (total_return - total_bets) / total_bets if total_bets > 0 else -1
    return -roi  # minimize → maximize ROI

# ------------------------------------------------
# HyperStack trainer
# ------------------------------------------------

class HyperStackTrainer:
    def __init__(self, n_trials=30, model_dir: str = "data"):
        self.logger = logging.getLogger("ML.HyperStackTrainer")
        self.model_dir = Path(model_dir)
        self.loader = DataLoader()
        self.db = DatabaseManager()
        self.n_trials = n_trials

        self.categorical_features = CATEGORICAL_FEATURES
        # [
        #     'racetrack_code', 'discipline', 'track_type', 'sex',
        #     'shoeing_status', 'jockey_name', 'trainer_name', 'terrain_label', 'meeting_type',
        # ]

        self.numerical_features = NUMERICAL_FEATURES
        #     [
        #     'horse_age_at_race', 'distance_m', 'declared_runners_count',
        #     'career_winnings', 'relative_winnings', 'winnings_rank_in_race',
        #     'odds_rank_in_race', 'reference_odds',
        #     'is_debutant', 'race_month', 'hist_avg_speed', 'hist_earnings',
        #     'speed_form_ratio', 'pct_clean_on_discipline', 'pct_races_on_discipline',
        #     'duo_total_races', 'duo_win_rate', 'duo_avg_rank',
        #     'duo_best_rank', 'duo_confidence',
        # ]

    def _ensure_table_schema(self):
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cur:
                cols_to_add = {
                    "algorithm": "TEXT DEFAULT 'hyperstack_context'",
                    "roi": "DOUBLE PRECISION DEFAULT 0",
                    "win_rate": "DOUBLE PRECISION DEFAULT 0",
                    "avg_odds": "DOUBLE PRECISION DEFAULT 0"
                }
                for col, dtype in cols_to_add.items():
                    cur.execute(f"""
                        DO $$ 
                        BEGIN 
                            IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                           WHERE table_name='ml_model_metrics' AND column_name='{col}') THEN
                                ALTER TABLE ml_model_metrics ADD COLUMN {col} {dtype};
                            END IF;
                        END $$;
                    """)
                cur.execute("""
                    DO $$ 
                    BEGIN 
                        IF EXISTS (SELECT 1 FROM information_schema.table_constraints 
                                   WHERE table_name='ml_model_metrics' AND constraint_type='PRIMARY KEY') THEN
                            ALTER TABLE ml_model_metrics DROP CONSTRAINT IF EXISTS ml_model_metrics_pkey;
                            ALTER TABLE ml_model_metrics ADD PRIMARY KEY (model_name, algorithm, segment_type, segment_value, test_month);
                        END IF;
                    EXCEPTION WHEN OTHERS THEN 
                    END $$;
                """)
                conn.commit()
        except Exception as e:
            self.logger.error(f"Migration error: {e}")
        finally:
            self.db.release_connection(conn)

    def train(self, test_days=90, val_days=60):
        self.logger.info("--- STARTING CONTEXTUAL HYPERSTACK TOURNAMENT ---")
        self._ensure_table_schema()
        try:
            raw_df = self.loader.get_training_data()
            self.logger.info(f"Data Loaded: {raw_df.shape} rows")
        except Exception as e:
            self.logger.error(f"CRITICAL: Database connection failed. {e}")
            return
        if raw_df.empty: return

        # Targets specific disciplines plus global
        targets = ["global"]
        if 'discipline' in raw_df.columns:
            disciplines = [d for d in raw_df['discipline'].unique() if pd.notna(d)]
            targets.extend([str(d).lower() for d in disciplines])

        for target in targets:
            self.logger.info(f"--- AutoML for Target: {target.upper()} ---")
            target_df = raw_df.copy() if target == "global" else raw_df[
                raw_df['discipline'].str.lower() == target].copy()
            if len(target_df) < 1500:
                self.logger.warning(f"Skipping {target}: insufficient data ({len(target_df)} rows)")

                continue
            self._train_target(target_df, target, test_days, val_days)

    def _train_target(self, data: pd.DataFrame, target_name: str, test_days: int, val_days: int):
        engineer = PmuFeatureEngineer()
        context_encoder = RaceContextEncoder(group_col="race_id")
        
        # # Initial transformation
        # df = engineer.fit_transform(data)
        # """
        # Utilisation des prédictions TabNet comme features dans HS. Pas comme stacking naïf, mais comme signal calibré :
        # la proba TabNet sur Monté est déjà à 63% ROI. On passe proba_tabnet comme feature numérique dans le ColumnTransformer de HS,
        # le stacker peut apprendre à pondérer les deux architectures selon la discipline.
        # Potentiellement +10-15% ROI sur Monté sans rien changer d'autre.
        # """
        # tabnet_model_path = self.model_dir / f"model_{target_name}_tabnet.pkl"
        # if tabnet_model_path.exists():
        #     try:
        #         tabnet_pipeline = safe_load(tabnet_model_path)
        #         df['proba_tabnet'] = tabnet_pipeline.predict_proba(df)[:, 1]
        #         if 'proba_tabnet' not in self.numerical_features:
        #             self.numerical_features = self.numerical_features + ['proba_tabnet']
        #     except Exception as e:
        #         self.logger.warning(f"TabNet model not loadable for {target_name}: {e}")
        #
        # df = context_encoder.transform(df)
        # TabNet bridge — chargé avant tout
        tabnet_path = self.model_dir / f"model_{target_name}_tabnet.pkl"
        tabnet_bridge = TabNetBridge(
            tabnet_path=str(tabnet_path) if tabnet_path.exists() else None
        )

        df = engineer.fit_transform(data)
        df = context_encoder.transform(df)
        df = tabnet_bridge.fit_transform(df)  # ajoute proba_tabnet si dispo

        train_df, val_df, test_df = self._temporal_split(df, test_days, val_days)

        # Contextual Features added to the selection
        # features = self.numerical_features + self.categorical_features + [
        #     'odds_log', 'market_sentiment', 'reference_odds_rel_race',
        #     'reference_odds_rank_race', 'hist_avg_speed_z_race',
        #     'career_winnings_rank_race',
        #     'avg_speed_last_3',       # nouveau
        #     'days_since_last_race',   # nouveau
        #     'hist_pct_clean_runs',    # nouveau
        #     'proba_tabnet',  # inclus si TabNet disponible
        # ]
        features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + CONTEXTUAL_FEATURES + EXTRA_FEATURES

        features = [f for f in features if f in df.columns] # filtre ce qui n'existe pas encore

        X_train, y_train = train_df[features], train_df['is_winner']
        cat_cols = [c for c in self.categorical_features if c in features]
        num_cols = [f for f in features if f not in cat_cols]

        preprocessor = ColumnTransformer([
            ('cat', CatBoostEncoder(cols=cat_cols), cat_cols),
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols)
        ])

        X_train_enc = preprocessor.fit_transform(X_train, y_train)
        feature_names = preprocessor.get_feature_names_out()
        X_train_df = pd.DataFrame(X_train_enc, columns=feature_names)

        # Optimize
        study_name = f"hs_ctx_{target_name}_3003_02"
        study = optuna.create_study(direction="minimize", storage=DB_URL, study_name=study_name, load_if_exists=True)
        # Problème 1 — Optuna optimise le mauvais critère
        # study.optimize(lambda trial: objective(trial, X_train_df, y_train), n_trials=self.n_trials, n_jobs=-1)
        train_effective_odds = train_df['live_odds'].fillna(
            train_df['reference_odds']
        ).fillna(1.0).clip(lower=1.05)

        # Réindexer pour correspondre à X_train_df
        train_effective_odds = train_effective_odds.reset_index(drop=True)
        train_race_ids = train_df['race_id'].reset_index(drop=True)

        study.optimize(
            lambda trial: objective(
                trial, X_train_df, y_train,
                train_race_ids, train_effective_odds
            ),
            n_trials=self.n_trials,
            n_jobs=1  # trials séquentiels, modèles parallèles n_jobs=-1
        )
        # Re-build best
        best_trial = optuna.trial.FixedTrial(study.best_params)
        base_models = build_base_models(best_trial)
        y_vals = y_train.values
        folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train_df, y_vals))
        oof, fold_models = generate_oof(base_models, X_train_df, y_train, folds)

        stacker = LogisticRegression(max_iter=2000)
        stacker.fit(oof, y_vals)

        final_model = HyperStackModel(fold_models, stacker, feature_names=feature_names)
        # full_pipeline = Pipeline([
        #     ('engineer', engineer),
        #     ('context', context_encoder),
        #     ('preprocessor', preprocessor),
        #     ('model', final_model)
        # ])
        # Pipeline avec TabNetBridge intégré
        full_pipeline = Pipeline([
            ('engineer', engineer),
            ('context', context_encoder),
            ('tabnet_bridge', tabnet_bridge),  # ← s'exécute automatiquement à l'inférence
            ('preprocessor', preprocessor),
            ('model', final_model)
        ])
        
        test_df_eval = test_df.copy()
        test_df_eval['proba'] = full_pipeline.predict_proba(test_df)[:, 1]
        metrics = self._calculate_metrics(test_df_eval)
        
        if metrics:
            self.logger.info(f"Target {target_name} Contextual HS Metrics: {metrics}")
            save_path = self.model_dir / f"model_{target_name}.pkl"
            joblib.dump(full_pipeline, save_path)

            # Rolling Year Evaluation: Indispensable pour capter la saisonnalité (meeting d'hiver vs été)
            max_date = df['program_date'].max()
            one_year_ago = max_date - pd.Timedelta(days=365)
            rolling_eval_df = df[df['program_date'] > one_year_ago].copy()

            self._generate_and_save_perf(full_pipeline, rolling_eval_df, target_name, "hyperstack_context")

    def _generate_and_save_perf(self, pipeline, test_df, target_name, algo_name):
        df = test_df.copy()
        try:
            df['proba'] = pipeline.predict_proba(test_df)[:, 1]
            df['month'] = df['program_date'].dt.month
            df['effective_odds'] = df['live_odds'].fillna(df['reference_odds']).fillna(1.0).clip(lower=1.05)
        except Exception: return

        perf_list = []
        segments = [
            ('discipline_overall', 'discipline', 0),
            ('discipline_month', 'discipline', 'month'),
            ('track_month', 'racetrack_code', 'month')
        ]

        for seg_type, col, month_col in segments:
            groupby_cols = [col]
            if month_col != 0: groupby_cols.append(month_col)

            for keys, group in df.groupby(groupby_cols):
                if len(group) < 20: continue
                val = keys if isinstance(keys, str) else keys[0]
                month = 0 if month_col == 0 else (keys if isinstance(keys, int) else keys[1])
                metrics = self._calculate_metrics(group)
                if metrics:
                    metrics.update({'segment_type': seg_type, 'segment_value': val, 'month': month})
                    perf_list.append(metrics)

        if perf_list:
            self._save_metrics_to_db_v2(target_name, algo_name, pd.DataFrame(perf_list))

    def _save_metrics_to_db_v2(self, model_name, algo_name, df):
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cur:
                for _, row in df.iterrows():
                    cur.execute("""
                        INSERT INTO ml_model_metrics (model_name, algorithm, segment_type, segment_value, test_month, num_races, logloss, auc, roi, win_rate, avg_odds)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (model_name, algorithm, segment_type, segment_value, test_month) 
                        DO UPDATE SET num_races=EXCLUDED.num_races, logloss=EXCLUDED.logloss, auc=EXCLUDED.auc, 
                        roi=EXCLUDED.roi, win_rate=EXCLUDED.win_rate, avg_odds=EXCLUDED.avg_odds, updated_at=NOW()
                    """, (model_name, algo_name, row['segment_type'], str(row['segment_value']), int(row['month']), int(row['count']), row['logloss'], row['auc'], row['roi'], row['win_rate'], row['avg_odds']))
                conn.commit()
        except Exception: pass
        finally: self.db.release_connection(conn)

    def _temporal_split(self, df, test_days, val_days):
        max_date = df['program_date'].max()
        test_cutoff = max_date - pd.Timedelta(days=test_days)
        val_cutoff = test_cutoff - pd.Timedelta(days=val_days)
        return df[df['program_date'] <= val_cutoff], \
               df[(df['program_date'] > val_cutoff) & (df['program_date'] <= test_cutoff)], \
               df[df['program_date'] > test_cutoff]

    def _calculate_metrics(self, group):
        if 'proba' not in group.columns or group['is_winner'].nunique() < 2: return None
        total_return, total_bets, win_count = 0, 0, 0
        df = group.copy()
        df['effective_odds'] = df['live_odds'].fillna(df['reference_odds']).fillna(1.0).clip(lower=1.05)
        for _, race_group in df.groupby('race_id'):
            best_horse = race_group.loc[race_group['proba'].idxmax()]
            if best_horse['is_winner'] == 1:
                total_return += best_horse['effective_odds']
                win_count += 1
            total_bets += 1
        return {
            'logloss': log_loss(group['is_winner'], group['proba']),
            'auc': roc_auc_score(group['is_winner'], group['proba']),
            'roi': ((total_return - total_bets) / total_bets * 100) if total_bets > 0 else 0,
            'win_rate': (total_return / total_bets if total_bets > 0 else 0),
            'avg_odds': df['effective_odds'].mean(),
            'count': total_bets
        }

if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[3]
    data_dir = project_root / "data"
    trainer = HyperStackTrainer(n_trials=3, model_dir=str(data_dir))
    trainer.train()
