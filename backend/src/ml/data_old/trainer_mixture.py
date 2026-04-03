import logging
import sys
from pathlib import Path

import numpy as np
import joblib
from category_encoders import CatBoostEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import log_loss, roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.ml.features import PmuFeatureEngineer
from src.ml.loader import DataLoader
import pandas as pd

# Set up logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class MixtureTrainer:

    def __init__(self, n_experts=3, model_dir: str = "data"):
        self.logger = logging.getLogger("ML.MixtureTrainer")
        self.model_dir = Path(model_dir)
        self.loader = DataLoader()

        self.n_experts = n_experts

        self.cluster = KMeans(n_clusters=n_experts, random_state=42, n_init=10)

        self.experts = []

        self.gate = LogisticRegression(max_iter=1000)
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

    def fit(self, X, y):
        self.logger.info(f"Fitting Mixture of {self.n_experts} Experts...")
        
        # Ensure X and y are numpy arrays
        X_np = X.values if hasattr(X, "values") else np.array(X)
        y_np = y.values if hasattr(y, "values") else np.array(y)

        clusters = self.cluster.fit_predict(X_np)

        expert_idx = []
        for k in range(self.n_experts):
            idx = clusters == k
            X_k = X_np[idx]
            y_k = y_np[idx]

            if len(X_k) < 100 or len(np.unique(y_k)) < 2:
                self.logger.warning(f"Expert {k} has too few samples ({len(X_k)}). Skipping.")
                continue

            model = XGBClassifier(
                n_estimators=800,
                max_depth=4,
                learning_rate=0.05,
                eval_metric='logloss',
                random_state=42
            )

            model.fit(X_k, y_k)
            self.experts.append(model)
            expert_idx.append(k)

        # train gating network
        expert_preds = []
        for m in self.experts:
            expert_preds.append(m.predict_proba(X_np)[:, 1])

        expert_preds = np.vstack(expert_preds).T
        self.gate.fit(expert_preds, y_np)

    def predict_proba(self, X):
        X_np = X.values if hasattr(X, "values") else np.array(X)
        expert_preds = []
        for m in self.experts:
            expert_preds.append(m.predict_proba(X_np)[:, 1])

        expert_preds = np.vstack(expert_preds).T
        return self.gate.predict_proba(expert_preds)

    def train(self, test_days=90, val_days=60):
        self.logger.info("--- STARTING MIXTURE TOURNAMENT ---")
        
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
        X_test, y_test = test_df[features], test_df['is_winner']

        preprocessor = ColumnTransformer([
            ('cat', CatBoostEncoder(cols=self.categorical_features), self.categorical_features),
            ('num', SimpleImputer(strategy='median'), [f for f in features if f not in self.categorical_features])
        ])

        X_train_enc = preprocessor.fit_transform(X_train, y_train)
        X_test_enc = preprocessor.transform(X_test)

        self.fit(X_train_enc, y_train)
        
        # Evaluation
        probs = self.predict_proba(X_test_enc)[:, 1]
        eval_df = test_df.copy()
        eval_df['proba'] = probs
        
        metrics = self._calculate_metrics(eval_df)
        if metrics:
            self.logger.info(f"Mixture Metrics: {metrics}")

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
    trainer = MixtureTrainer(n_experts=5)
    trainer.train()
