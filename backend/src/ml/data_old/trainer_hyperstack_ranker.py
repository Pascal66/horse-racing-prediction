import logging
import sys
from pathlib import Path

import numpy as np
import joblib
import optuna
import pandas as pd
from category_encoders import CatBoostEncoder
from scipy.special import softmax
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

from xgboost import XGBClassifier, XGBRanker
from lightgbm import LGBMRanker
from catboost import CatBoostClassifier

from src.ml.features import PmuFeatureEngineer
from src.ml.loader import DataLoader


# ------------------------------------------------
# Utilities
# ------------------------------------------------

def group_sizes(groups):
    _, counts = np.unique(groups, return_counts=True)
    return counts.tolist()


def softmax_by_group(scores, groups):
    probs = np.zeros_like(scores)
    unique = np.unique(groups)
    for g in unique:
        idx = groups == g
        probs[idx] = softmax(scores[idx])
    return probs


# ------------------------------------------------
# Build models
# ------------------------------------------------

def build_models(trial):
    models = {
        "xgb_ranker": XGBRanker(
            n_estimators=trial.suggest_int("xgb_ranker_estimators", 200, 1200),
            max_depth=trial.suggest_int("xgb_ranker_depth", 3, 8),
            learning_rate=trial.suggest_float("xgb_ranker_lr", 0.01, 0.15),
            subsample=trial.suggest_float("xgb_ranker_sub", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("xgb_ranker_col", 0.6, 1.0),
            tree_method="hist",
            random_state=42
        ),
        "lgbm_ranker": LGBMRanker(
            n_estimators=trial.suggest_int("lgbm_ranker_estimators", 200, 1200),
            learning_rate=trial.suggest_float("lgbm_ranker_lr", 0.01, 0.15),
            num_leaves=trial.suggest_int("lgbm_ranker_leaves", 16, 128),
            random_state=42,
            verbosity=-1
        ),
        "xgb_classifier": XGBClassifier(
            n_estimators=trial.suggest_int("xgb_estimators", 200, 1200),
            max_depth=trial.suggest_int("xgb_depth", 3, 8),
            learning_rate=trial.suggest_float("xgb_lr", 0.01, 0.15),
            tree_method="hist",
            eval_metric="logloss",
            random_state=42
        ),
        "cat_classifier": CatBoostClassifier(
            iterations=trial.suggest_int("cat_iter", 200, 1000),
            depth=trial.suggest_int("cat_depth", 4, 8),
            learning_rate=trial.suggest_float("cat_lr", 0.01, 0.15),
            verbose=False,
            random_seed=42
        )
    }
    return models


# ------------------------------------------------
# OOF generation
# ------------------------------------------------

def generate_oof(models, X, y, groups, folds):
    n_models = len(models)
    oof = np.zeros((len(X), n_models))
    trained_models = []

    X_is_df = isinstance(X, pd.DataFrame)
    y_vals = y.values if hasattr(y, "values") else y

    for m_id, (name, model) in enumerate(models.items()):
        preds = np.zeros(len(X))
        fold_models = []
        for train_idx, val_idx in folds:
            X_train = X.iloc[train_idx] if X_is_df else X[train_idx]
            X_val = X.iloc[val_idx] if X_is_df else X[val_idx]
            y_train, y_val = y_vals[train_idx], y_vals[val_idx]
            g_train, g_val = groups[train_idx], groups[val_idx]

            m = clone(model)
            if "ranker" in name:
                m.fit(X_train, y_train, group=group_sizes(g_train))
                scores = m.predict(X_val)
                preds[val_idx] = softmax_by_group(scores, g_val)
            else:
                m.fit(X_train, y_train)
                preds[val_idx] = m.predict_proba(X_val)[:, 1]
            fold_models.append(m)

        oof[:, m_id] = preds
        trained_models.append((name, fold_models))

    return oof, trained_models


# ------------------------------------------------
# Optuna objective
# ------------------------------------------------

def objective(trial, X, y, groups):
    y_vals = y.values if hasattr(y, "values") else y
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(X, y_vals, groups))

    models = build_models(trial)
    oof, _ = generate_oof(models, X, y_vals, groups, folds)

    stacker = LogisticRegression(max_iter=2000)
    # stacker_type = trial.suggest_categorical("stacker", ["logistic", "xgb"])
    #
    # if stacker_type == "logistic":
    #     stacker = LogisticRegression(max_iter=2000)
    # else:
    #     stacker = XGBClassifier(
    #         n_estimators=400,
    #         max_depth=3,
    #         learning_rate=0.05,
    #         tree_method="hist",
    #         eval_metric="logloss",
    #         random_state=42
    #     )

    losses = []
    for train_idx, val_idx in folds:
        X_train_stack, X_val_stack = oof[train_idx], oof[val_idx]
        y_train_stack, y_val_stack = y_vals[train_idx], y_vals[val_idx]

        stacker.fit(X_train_stack, y_train_stack)
        preds = stacker.predict_proba(X_val_stack)[:, 1]
        loss = log_loss(y_val_stack, preds)
        losses.append(loss)

        trial.report(loss, step=len(losses))
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(losses)


# Set up logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


# ------------------------------------------------
# Trainer
# ------------------------------------------------

class HyperStackRankerTrainer:
    def __init__(self, n_trials=50, model_dir: str = "data"):
        self.logger = logging.getLogger("ML.HyperStackRankerTrainer")
        self.model_dir = Path(model_dir)
        self.loader = DataLoader()

        self.n_trials = n_trials
        self.base_models = None
        self.fold_models = None
        self.stacker = LogisticRegression(max_iter=2000) #None
        self.categorical_features = [
            'racetrack_code', 'discipline', 'track_type', 'sex',
            'shoeing_status', 'jockey_name', 'trainer_name', 'terrain_label'
        ]

        self.numerical_features = [
            'horse_age_at_race', 'distance_m', 'declared_runners_count',
            'career_winnings', 'relative_winnings', 'winnings_rank_in_race', 
            'odds_rank_in_race', 'reference_odds',
            'is_debutant', 'race_month', 'hist_avg_speed', 'hist_earnings'
        ]

    def fit(self, X, y, groups):
        # Study and fit using DataFrames for LGBM
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner()
        )

        y_vals = y.values if hasattr(y, "values") else y
        g_vals = groups.values if hasattr(groups, "values") else groups

        study.optimize(
            lambda trial: objective(trial, X, y_vals, g_vals),
            n_trials=self.n_trials
        )

        self.logger.info(f"Best params: {study.best_params}")
        best_trial = optuna.trial.FixedTrial(study.best_params)
        self.base_models = build_models(best_trial)

        # y_vals = y.values if hasattr(y, "values") else y
        # folds = list(StratifiedKFold(
        #     n_splits=5,
        #     shuffle=True,
        #     random_state=42
        # ).split(X, y_vals))

        gkf = GroupKFold(n_splits=5)
        folds = list(gkf.split(X, y_vals, g_vals))

        oof, self.fold_models = generate_oof(self.base_models, X, y_vals, g_vals, folds)

        # if study.best_params["stacker"] == "logistic":
        self.stacker = LogisticRegression(max_iter=2000)
        # else:
        #     self.stacker = XGBClassifier(
        #         n_estimators=400,
        #         max_depth=3,
        #         learning_rate=0.05,
        #         tree_method="hist",
        #         eval_metric="logloss",
        #         random_state=42
        #     )
        #
        self.stacker.fit(oof, y_vals)

    def predict_proba(self, X, groups):
        g_vals = groups.values if hasattr(groups, "values") else groups
        base_preds = []
        for name, fold_models in self.fold_models:
            preds = []
            for m in fold_models:
                if "ranker" in name:
                    scores = m.predict(X)
                    preds.append(softmax_by_group(scores, g_vals))
                else:
                    preds.append(m.predict_proba(X)[:, 1])
            preds = np.mean(preds, axis=0)
            base_preds.append(preds)

        base_preds = np.vstack(base_preds).T
        return self.stacker.predict_proba(base_preds)

    def train(self, test_days=90, val_days=60):
        self.logger.info("--- STARTING HYPERSTACK RANKER TOURNAMENT ---")
        
        try:
            raw_df = self.loader.get_training_data()
            self.logger.info(f"Data Loaded: {raw_df.shape} rows")
        except Exception as e:
            self.logger.error(f"CRITICAL: Database connection failed. {e}")
            return

        if raw_df.empty: return

        engineer = PmuFeatureEngineer()
        df = engineer.fit_transform(raw_df)
        df = self._add_features(df)

        train_df, val_df, test_df = self._temporal_split(df, test_days, val_days)

        features = self.numerical_features + self.categorical_features + [
            'odds_log', 'earnings_log', 'speed_rel', 'odds_rank_pct'
        ]

        X_train, y_train = train_df[features], train_df['is_winner']
        g_train = train_df['race_id'].values
        X_test, y_test = test_df[features], test_df['is_winner']
        g_test = test_df['race_id'].values

        preprocessor = ColumnTransformer([
            ('cat', CatBoostEncoder(cols=self.categorical_features), self.categorical_features),
            ('num', SimpleImputer(strategy='median'), [f for f in features if f not in self.categorical_features])
        ])

        X_train_enc = preprocessor.fit_transform(X_train, y_train)
        feature_names = preprocessor.get_feature_names_out()
        
        # Keep DataFrames for feature names
        X_train_df = pd.DataFrame(X_train_enc, columns=feature_names)
        X_test_df = pd.DataFrame(preprocessor.transform(X_test), columns=feature_names)

        self.fit(X_train_df, y_train, g_train)
        
        # Evaluation
        probs = self.predict_proba(X_test_df, g_test)[:, 1]
        eval_df = test_df.copy()
        eval_df['proba'] = probs
        
        metrics = self._calculate_metrics(eval_df)
        if metrics:
            self.logger.info(f"HyperStack Ranker Metrics: {metrics}")

    def _add_features(self, df):
        df['odds_log'] = np.log1p(df['reference_odds'].fillna(20))
        df['earnings_log'] = np.log1p(df['career_winnings'].fillna(0))
        df['speed_rel'] = df['hist_avg_speed'] / df.groupby('race_id')['hist_avg_speed'].transform('mean')
        df['odds_rank_pct'] = df.groupby('race_id')['reference_odds'].rank(pct=True)
        return df

    def _temporal_split(self, df, test_days, val_days):
        max_date = df['program_date'].max()
        test_cutoff = max_date - pd.Timedelta(days=test_days)
        val_cutoff = test_cutoff - pd.Timedelta(days=val_days)

        train = df[df['program_date'] <= val_cutoff]
        val = df[(df['program_date'] > val_cutoff) & (df['program_date'] <= test_cutoff)]
        test = df[df['program_date'] > test_cutoff]

        return train, val, test

    def _calculate_metrics(self, group):
        y_true = group['is_winner']
        y_pred = group['proba']
        if y_true.nunique() < 2: return None

        total_return = 0
        total_bets = 0
        
        df = group.copy()
        df['effective_odds'] = df['live_odds'].fillna(df['reference_odds']).fillna(1.0).clip(lower=1.05)

        for race_id in df['race_id'].unique():
            race_group = df[df['race_id'] == race_id]
            best_horse = race_group.loc[race_group['proba'].idxmax()]
            if best_horse['is_winner'] == 1:
                total_return += best_horse['effective_odds']
            total_bets += 1

        return {
            'logloss': log_loss(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_pred),
            'roi': ((total_return - total_bets) / total_bets * 100) if total_bets > 0 else 0,
            'win_rate': (total_return / total_bets if total_bets > 0 else 0),
            'count': total_bets
        }

if __name__ == "__main__":
    trainer = HyperStackRankerTrainer(n_trials=20)
    trainer.train()
