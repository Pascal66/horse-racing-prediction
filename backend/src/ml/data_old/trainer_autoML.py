import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import joblib
from category_encoders import CatBoostEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.core.config import DB_URL
from src.ml.features import PmuFeatureEngineer
from src.ml.loader import DataLoader

def build_model(trial, algo):
    if algo == "xgb":
        return XGBClassifier(
            n_estimators=trial.suggest_int("xgb_n_estimators", 200, 2000),
            max_depth=trial.suggest_int("xgb_max_depth", 3, 8),
            learning_rate=trial.suggest_float("xgb_lr", 0.01, 0.2),
            subsample=trial.suggest_float("xgb_subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("xgb_colsample", 0.6, 1.0),
            min_child_weight=trial.suggest_int("xgb_min_child_weight", 1, 10),
            eval_metric="logloss",
            gamma=trial.suggest_float("xgb_gamma", 0, 5),
            reg_alpha=trial.suggest_float("xgb_alpha", 0, 1),
            reg_lambda=trial.suggest_float("xgb_lambda", 0, 2),
            tree_method="hist",
            n_jobs=-1,
            random_state=42
        )

    if algo == "lgbm":
        return LGBMClassifier(
            n_estimators=trial.suggest_int("lgbm_n_estimators", 200, 2000),
            max_depth=trial.suggest_int("lgbm_max_depth", 3, 10),
            learning_rate=trial.suggest_float("lgbm_lr", 0.01, 0.2),
            num_leaves=trial.suggest_int("lgbm_leaves", 16, 128),
            min_child_weight=trial.suggest_int("lgbm_min_child_weight", 1, 10),
            min_child_samples=trial.suggest_int("lgbm_child_samples", 5, 30),
            reg_alpha=trial.suggest_float("lgbm_alpha", 0, 1),
            reg_lambda=trial.suggest_float("lgbm_lambda", 0, 2),
            n_jobs=-1,
            random_state=42,
            verbosity=-1
        )

    if algo == "cat":
        return CatBoostClassifier(
            iterations=trial.suggest_int("cat_iter", 200, 1500),
            depth=trial.suggest_int("cat_depth", 4, 8),
            learning_rate=trial.suggest_float("cat_lr", 0.01, 0.2),
            l2_leaf_reg=trial.suggest_float("cat_l2", 1, 10),
            border_count=trial.suggest_int("cat_border", 32, 255),
            verbose=False,
            random_seed=42
        )
    return None


# --------------------------------------------------
# Objective Optuna
# --------------------------------------------------

def objective(trial, X, y, algo):
    model = build_model(trial, algo)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    losses = []

    X_is_df = isinstance(X, pd.DataFrame)
    y_vals = y.values if hasattr(y, "values") else y

    for train_idx, val_idx in cv.split(X, y_vals):
        X_train = X.iloc[train_idx] if X_is_df else X[train_idx]
        X_val = X.iloc[val_idx] if X_is_df else X[val_idx]
        y_train, y_val = y_vals[train_idx], y_vals[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        loss = log_loss(y_val, preds)
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

# --------------------------------------------------
# AutoML Trainer
# --------------------------------------------------

class AutoMLTrainer:
    def __init__(self, n_trials_per_algo=20, model_dir: str = "data"):
        self.logger = logging.getLogger("ML.AutoMLTrainer")
        self.model_dir = Path(model_dir)
        self.loader = DataLoader()

        self.n_trials = n_trials_per_algo
        self.base_models = []
        self.meta_model = LogisticRegression()
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

    def fit(self, X, y, target_name: str):
        self.base_models = []
        all_algo_preds = []

        for algo in ["xgb", "lgbm", "cat"]:
            self.logger.info(f"Optimizing {algo.upper()} for {target_name}...")
            study_name = f"automl_{target_name}_{algo}"
            
            study = optuna.create_study(
                direction="minimize",
                pruner=optuna.pruners.MedianPruner(),
                load_if_exists=True,
                study_name=study_name,
                storage=DB_URL
            )

            study.optimize(
                lambda trial: objective(trial, X, y, algo),
                n_trials=self.n_trials,
                n_jobs=-1
            )

            self.logger.info(f"Best {algo} params: {study.best_params}")
            
            # Re-train the best model for this algo
            best_model = build_model(optuna.trial.FixedTrial(study.best_params), algo)
            best_model.fit(X, y)
            self.base_models.append(best_model)
            
            # Collect predictions for meta-model
            all_algo_preds.append(best_model.predict_proba(X)[:, 1])

        # Train meta-model (Logistic Regression) on the best of each algorithm
        meta_X = np.vstack(all_algo_preds).T
        self.meta_model.fit(meta_X, y)

    def predict_proba(self, X):
        all_preds = []
        for m in self.base_models:
            all_preds.append(m.predict_proba(X)[:, 1])
        
        meta_X = np.vstack(all_preds).T
        return self.meta_model.predict_proba(meta_X)

    def train(self, test_days=90, val_days=60):
        self.logger.info("--- STARTING AUTOML TOURNAMENT (Multi-Study) ---")
        
        try:
            raw_df = self.loader.get_training_data()
            self.logger.info(f"Data Loaded: {raw_df.shape} rows")
        except Exception as e:
            self.logger.error(f"CRITICAL: Database connection failed. {e}")
            return

        if raw_df.empty: return

        targets = ["global"]
        if 'discipline' in raw_df.columns:
            disciplines = raw_df['discipline'].unique()
            targets.extend([str(d).lower() for d in disciplines if pd.notna(d) and str(d).strip() != ""])

        for target in targets:
            self.logger.info(f"--- AutoML Tournament for Target: {target.upper()} ---")
            target_df = raw_df.copy() if target == "global" else raw_df[raw_df['discipline'].str.lower() == target].copy()
            if len(target_df) < 1000: continue
            
            self._train_target(target_df, target, test_days, val_days)

    def _train_target(self, data: pd.DataFrame, target_name: str, test_days: int, val_days: int):
        engineer = PmuFeatureEngineer()
        df = engineer.fit_transform(data)
        df = self._add_features(df)

        train_df, val_df, test_df = self._temporal_split(df, test_days, val_days)

        features = self.numerical_features + self.categorical_features + [
            'odds_log', 'earnings_log', 'speed_rel', 'odds_rank_pct'
        ]

        X_train, y_train = train_df[features], train_df['is_winner']
        X_test, y_test = test_df[features], test_df['is_winner']

        preprocessor = ColumnTransformer([
            ('cat', CatBoostEncoder(cols=self.categorical_features), self.categorical_features),
            ('num', SimpleImputer(strategy='median'), [f for f in features if f not in self.categorical_features])
        ])

        X_train_enc = preprocessor.fit_transform(X_train, y_train)
        feature_names = preprocessor.get_feature_names_out()
        
        X_train_df = pd.DataFrame(X_train_enc, columns=feature_names)
        X_test_df = pd.DataFrame(preprocessor.transform(X_test), columns=feature_names)

        self.fit(X_train_df, y_train, target_name)
        
        # Evaluation
        probs = self.predict_proba(X_test_df)[:, 1]
        eval_df = test_df.copy()
        eval_df['proba'] = probs
        
        metrics = self._calculate_metrics(eval_df)
        if metrics:
            self.logger.info(f"Target {target_name} FINAL Metrics (Ensemble): {metrics}")

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
    trainer = AutoMLTrainer(n_trials_per_algo=20)
    trainer.train()
