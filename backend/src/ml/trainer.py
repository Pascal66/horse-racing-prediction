import time

import joblib
import logging
import sys
import numpy as np
import optuna
import pandas as pd
from category_encoders import CatBoostEncoder
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier, DMatrix, train, XGBRanker
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score, mean_absolute_error, mean_squared_error
from pathlib import Path
from src.core.config import DB_URL
import shap

# Ensure python finds the source modules
sys.path.append(str(Path(__file__).parent.parent))

from src.ml.loader import DataLoader
from src.ml.features import PmuFeatureEngineer

# Set up logging to stdout for Docker
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


class XGBoostTrainer:
    def __init__(self, model_dir: str = "data") -> None:

        self.logger = logging.getLogger("ML.Trainer")
        self.model_dir = Path(model_dir)
        self.loader = DataLoader()

        self.categorical_features = [
            'racetrack_code', 'discipline', 'track_type', 'sex',
            'shoeing_status', 'jockey_name', 'trainer_name', 'terrain_label'
        ]

        self.numerical_features = [
            'horse_age_at_race', 'distance_m', 'declared_runners_count',
            'career_winnings', 'relative_winnings', 'winnings_per_race',
            'winnings_rank_in_race', 'odds_rank_in_race', 'reference_odds',
            'is_debutant', 'race_month', 'hist_avg_speed', 'hist_earnings'
        ]

    def _add_features(self, df):
        """
        Feature engineering critique
        """
        df['odds_log'] = np.log1p(df['reference_odds'])
        df['earnings_log'] = np.log1p(df['career_winnings'])

        df['speed_rel'] = df['hist_avg_speed'] / df.groupby('race_id')['hist_avg_speed'].transform('mean')
        df['odds_rank_pct'] = df.groupby('race_id')['reference_odds'].rank(pct=True)

        return df

    def _temporal_split(self, df, test_days, val_days):
        """
        rolling window training
        """
        max_date = df['program_date'].max()
        test_cutoff = max_date - pd.Timedelta(days=test_days)
        val_cutoff = test_cutoff - pd.Timedelta(days=val_days)

        train = df[df['program_date'] <= val_cutoff]
        val = df[(df['program_date'] > val_cutoff) & (df['program_date'] <= test_cutoff)]
        test = df[df['program_date'] > test_cutoff]

        return train, val, test

    def train(self, test_days: int = 90, val_days: int = 60) -> None:
        self.logger.info("--- STARTING MULTI-MODEL TRAINING PIPELINE ---")

        # 1. Loading
        try:
            raw_df = self.loader.get_training_data()
            self.logger.info(f"Data Loaded: {raw_df.shape} rows")
        except Exception as e:
            self.logger.error(f"CRITICAL: Database connection failed. {e}")
            return

        if raw_df.empty:
            self.logger.error("No data returned by the loader.")
            return

        # 2. Global Training (Fallback Model)
        self.logger.info("--- Training GLOBAL (Fallback) Model ---")
        self._train_and_save(raw_df, "global", test_days, val_days)

        # 3. Specialty Training (per Discipline)
        if 'discipline' in raw_df.columns:
            disciplines = raw_df['discipline'].unique()
            for discipline in disciplines:
                if pd.isna(discipline) or str(discipline).strip() == "":
                    continue

                self.logger.info(f"--- Training Specialty Model: {discipline} ---")
                discipline_df = raw_df[raw_df['discipline'] == discipline].copy()

                if len(discipline_df) < 500:  # Minimum threshold for specialty training
                    self.logger.warning(f"Insufficient data for {discipline} ({len(discipline_df)} rows). Skipping.")
                    continue

                self._train_and_save(discipline_df, str(discipline).lower(), test_days, val_days)

    def _train_and_save(self, data: pd.DataFrame, model_name: str, test_days: int, val_days: int) -> None:
        # 2. Feature Engineering
        engineer = PmuFeatureEngineer()
        full_df = engineer.fit_transform(data)

        full_df = self._add_features(full_df)

        # 3. Temporal Split (fragile si drift)
        max_date = full_df['program_date'].max()
        test_cutoff = max_date - pd.Timedelta(days=test_days)
        val_cutoff = test_cutoff - pd.Timedelta(days=val_days)

        train_df = full_df[full_df['program_date'] <= val_cutoff]
        val_df = full_df[(full_df['program_date'] > val_cutoff) & (full_df['program_date'] <= test_cutoff)]
        test_df = full_df[full_df['program_date'] > test_cutoff]

        if len(train_df) < 100 or len(val_df) < 20:
            self.logger.warning(f"Not enough data for {model_name} after split. Skipping.")
            return

        self.logger.info(f"[{model_name}] Split -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

        # Select Features
        features = self.numerical_features + self.categorical_features + [
            'odds_log', 'earnings_log', 'speed_rel', 'odds_rank_pct'
        ]
        X_train, y_train = train_df[features], train_df['is_winner']
        X_val, y_val = val_df[features], val_df['is_winner']
        X_test, y_test = test_df[features], test_df['is_winner']

        # 4. Preprocessing
        # Sous-optimal pour XGBoost.
        # Introduit un ordre artificiel (jockey A < jockey B)
        # XGBoost interprète ça comme une relation numérique → faux signal
        # preprocessor = ColumnTransformer(
        #     transformers=[
        #         ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
        #          self.categorical_features),
        #         ('num', 'passthrough', self.numerical_features)
        #     ]
        # )
        # 4.1 CatBoostEncoder (moins de leakage)
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', CatBoostEncoder(cols=self.categorical_features), self.categorical_features),
                ('num', 'passthrough', self.numerical_features)
            ]
        )

        X_train_enc = preprocessor.fit_transform(X_train, y_train)
        X_val_enc = preprocessor.transform(X_val)
        X_test_enc = preprocessor.transform(X_test)

        # GPU version with device='cuda'

        dtrain_gpu = DMatrix(X_train_enc, label=y_train)
        dvalid_gpu = DMatrix(X_val_enc, label=y_val)
        dtest_gpu = DMatrix(X_test_enc, label=y_test)

        # 5. Base XGBoost paramètres trop “génériques”: réduire l’overfitting (clé en PMU)
        base_xgb = XGBClassifier(
            n_estimators=3000, #1000,
            max_depth=4, #6,
            learning_rate=0.03, #0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            # tree_method='hist',
            early_stopping_rounds=50,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            # device="cuda"

        )
        def objective(trial):

            params = {
                "n_estimators": trial.suggest_int("n_estimators", 1000, 4000),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 2),
                "tree_method": "hist",
                # "device":"cuda"
            }

            model = XGBClassifier(**params)

            model.fit(X_train_enc, y_train,
                      eval_set=[(X_val_enc, y_val)],
                      # early_stopping_rounds=50,
                      verbose=False)

            preds = model.predict_proba(X_val_enc)[:, 1]
            return log_loss(y_val, preds)

        study = optuna.create_study(direction="minimize", load_if_exists=True, study_name=f"optuna_study_{model_name}", storage=DB_URL)
        study.optimize(objective, n_trials=10, n_jobs=-1, show_progress_bar=False)
        # study.trials_dataframe()
        # print(study.best_params)

        base_xgb.fit(
            X_train_enc,
            y_train,
            eval_set=[(X_val_enc, y_val)],
            verbose=False  # Keep it clean
        )

        # 6. Calibration (deprecated)
        # calibrated_model = CalibratedClassifierCV(
        #     estimator=base_xgb,
        #     method='isotonic',
        #     cv='prefit'
        # )
        #   -  calib_clf = CalibratedClassifierCV(clf, cv="prefit")
        #   +  calib_clf = CalibratedClassifierCV(FrozenEstimator(clf), cv=2)
        # it is not advised to use isotonic calibration with too few calibration samples (<<1000) since it tends to overfit.
        # calibrated_model = CalibratedClassifierCV(FrozenEstimator(base_xgb), method='isotonic', cv=2)
        # 6.1 sigmoid > isotonic dans 90% des cas réels
        calibrated_model = CalibratedClassifierCV(FrozenEstimator(base_xgb), method='sigmoid')
        calibrated_model.fit(X_val_enc, y_val)

        # 7. Evaluation
        probs_m1 = calibrated_model.predict_proba(X_test_enc)[:, 1]
        loss = log_loss(y_test, probs_m1)
        try:
            auc = roc_auc_score(y_test, probs_m1)
        except ValueError:
            auc = 0.5

        self.logger.info(f"[{model_name}_m1] Metrics -> LogLoss: {loss:.4f} | AUC: {auc:.4f}")
        # ======================
        # MODEL 2 - RANKER
        # ======================
        ranker = XGBRanker(
            objective='rank:pairwise',
            n_estimators=2000,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            random_state=42
        )

        group_train = train_df.groupby('race_id').size().to_list()
        group_val = val_df.groupby('race_id').size().to_list()

        ranker.fit(X_train_enc, y_train,
                   group=group_train,
                   eval_set=[(X_val_enc, y_val)],
                   eval_group=[group_val],
                   verbose=False)

        # p2_raw = ranker.predict(X_test_enc)
        # test_df2 = test_df.copy()
        # test_df2['rank_score'] = p2_raw
        # probs_m2 = test_df2.groupby('race_id')['rank_score'].transform(lambda x: x / x.sum()).values
        # test_def2['proba'] = probs_m2


        # 8. Save Pipeline
        full_inference_pipeline = Pipeline([
            ('engineer', engineer),
            ('preprocessor', preprocessor),
            # ('model', {calibrated_model, ranker})
            ('model', calibrated_model)
        ])

        # Ensure directory exists
        self.model_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.model_dir / f"model_{model_name}.pkl"

        joblib.dump(full_inference_pipeline, save_path)
        self.logger.info(f"SUCCESS: {model_name.upper()} Model saved to {save_path}")


if __name__ == "__main__":
    import sys

    # For local execution testing, default to project data folder
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[3]  # backend/src/ml/trainer.py -> backend/src/ -> backend/ -> root/
    data_dir = project_root / "data"

    trainer = XGBoostTrainer(str(data_dir))
    trainer.train()