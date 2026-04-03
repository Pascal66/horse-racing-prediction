import joblib
import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from xgboost import XGBClassifier, XGBRanker
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from category_encoders import CatBoostEncoder

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from src.ml.loader import DataLoader
from src.ml.features import PmuFeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


class MultiModelTrainer:
    def __init__(self, model_dir: str = "data"):
        self.logger = logging.getLogger("ML.MultiTrainer")
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

    def train(self, test_days=60, val_days=30):
        self.logger.info("=== MULTI MODEL TRAINING START ===")

        raw_df = self.loader.get_training_data()
        if raw_df.empty:
            self.logger.error("No data")
            return

        engineer = PmuFeatureEngineer()
        df = engineer.fit_transform(raw_df)

        df = self._add_features(df)

        train_df, val_df, test_df = self._temporal_split(df, test_days, val_days)

        features = self.categorical_features + self.numerical_features + [
            'odds_log', 'earnings_log', 'speed_rel', 'odds_rank_pct'
        ]

        X_train, y_train = train_df[features], train_df['is_winner']
        X_val, y_val = val_df[features], val_df['is_winner']
        X_test, y_test = test_df[features], test_df['is_winner']

        preprocessor = ColumnTransformer([
            ('cat', CatBoostEncoder(cols=self.categorical_features), self.categorical_features),
            ('num', 'passthrough', [f for f in features if f not in self.categorical_features])
        ])

        X_train_enc = preprocessor.fit_transform(X_train, y_train)
        X_val_enc = preprocessor.transform(X_val)
        X_test_enc = preprocessor.transform(X_test)

        # ======================
        # MODEL 1 - CLASSIFIER
        # ======================
        clf = XGBClassifier(
            n_estimators=3000,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method='hist',
            eval_metric='logloss',
            n_jobs=-1
        )

        clf.fit(X_train_enc, y_train,
                eval_set=[(X_val_enc, y_val)],
                # early_stopping_rounds=100,
                verbose=False)

        calibrated = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
        calibrated.fit(X_val_enc, y_val)

        p1 = calibrated.predict_proba(X_test_enc)[:, 1]

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
            tree_method='hist'
        )

        group_train = train_df.groupby('race_id').size().to_list()
        group_val = val_df.groupby('race_id').size().to_list()

        ranker.fit(X_train_enc, y_train,
                   group=group_train,
                   eval_set=[(X_val_enc, y_val)],
                   eval_group=[group_val],
                   verbose=False)

        p2_raw = ranker.predict(X_test_enc)
        test_df['rank_score'] = p2_raw
        p2 = test_df.groupby('race_id')['rank_score'].transform(lambda x: x / x.sum()).values

        # ======================
        # MODEL 3 - GPU STYLE (CPU fallback)
        # ======================
        gpu_model = XGBClassifier(
            n_estimators=2000,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            tree_method='hist',
            eval_metric='logloss'
        )

        gpu_model.fit(X_train_enc, y_train,
                      eval_set=[(X_val_enc, y_val)],
                      # early_stopping_rounds=50,
                      verbose=False)

        p3 = gpu_model.predict_proba(X_test_enc)[:, 1]

        # ======================
        # STACKING
        # ======================
        stack_X = np.vstack([p1, p2, p3]).T

        meta = XGBClassifier(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.05
        )

        meta.fit(stack_X, y_test)

        final_preds = meta.predict_proba(stack_X)[:, 1]

        loss = log_loss(y_test, final_preds)
        self.logger.info(f"FINAL LOGLOSS: {loss:.4f}")

        # ======================
        # SAVE PIPELINE
        # ======================
        pipeline = {
            'engineer': engineer,
            'preprocessor': preprocessor,
            'models': {
                'clf': calibrated,
                'ranker': ranker,
                'gpu': gpu_model,
                'meta': meta
            }
        }

        self.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, self.model_dir / "multimodel.pkl")

        self.logger.info("MODEL SAVED")

    def _add_features(self, df):
        df['odds_log'] = np.log1p(df['reference_odds'])
        df['earnings_log'] = np.log1p(df['career_winnings'])

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


if __name__ == "__main__":
    trainer = MultiModelTrainer()
    trainer.train()
