import logging
import sys
from pathlib import Path

import numpy as np
import joblib
import pandas as pd
from category_encoders import CatBoostEncoder

from pytorch_tabnet.tab_model import TabNetClassifier

from sklearnex import patch_sklearn
patch_sklearn(verbose=False)  # avant tous les imports sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline

from src.core.database import DatabaseManager
from src.ml.features import PmuFeatureEngineer, RaceContextEncoder
from src.ml.loader import DataLoader

# Set up logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

from src.ml.tabnet_utils import TabNetEnsembleWrapper
from src.ml.feature_config import (
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    CONTEXTUAL_FEATURES, EXTRA_FEATURES
)
import os
os.environ["SKLEARNEX_VERBOSE"] = "WARNING"

# ------------------------------------------------
# Race Contextualization
# ------------------------------------------------
#
# class _RaceContextEncoder(BaseEstimator, TransformerMixin):
#     def __init__(self, group_col="race_id", feature_cols=None):
#         self.group_col = group_col
#         self.feature_cols = feature_cols
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         df = X.copy()
#         if self.feature_cols is None:
#             self.feature_cols = [
#                 'reference_odds', 'career_winnings', 'horse_age_at_race',
#                 'hist_avg_speed', 'hist_earnings'
#             ]
#
#         cols_to_use = [c for c in self.feature_cols if c in df.columns]
#         if self.group_col not in df.columns:
#             return df
#
#         grouped = df.groupby(self.group_col)
#         for col in cols_to_use:
#             mean_val = grouped[col].transform("mean")
#             std_val = grouped[col].transform("std")
#             df[f"{col}_rel_race"] = df[col] - mean_val
#             df[f"{col}_z_race"] = (df[col] - mean_val) / (std_val + 1e-6)
#             df[f"{col}_rank_race"] = grouped[col].rank(pct=True)
#
#         if 'reference_odds' in df.columns:
#             df['implied_prob'] = 1.0 / df['reference_odds'].replace(0, np.nan).fillna(100).clip(lower=1.01)
#             race_total_prob = grouped['implied_prob'].transform("sum")
#             df['market_sentiment'] = df['implied_prob'] / (race_total_prob + 1e-6)
#
#         return df

# ------------------------------------------------
# TabNet Factory
# ------------------------------------------------

def build_tabnet_model():
    return TabNetClassifier(
        n_d=24, n_a=24, n_steps=4,
        gamma=1.3,
        lambda_sparse=1e-4,
        optimizer_params=dict(lr=2e-2),
        mask_type='entmax',
        seed=42,
        verbose=0,
        device_name="auto"
    )

# ------------------------------------------------
# HyperStack trainer
# ------------------------------------------------

class TabNetTrainer:
    def __init__(self, model_dir: str = "data"):
        self.logger = logging.getLogger("ML.TabnetTrainer")
        self.model_dir = Path(model_dir)
        self.loader = DataLoader()
        self.db = DatabaseManager()

        self.categorical_features = CATEGORICAL_FEATURES
        #     [
        #     'racetrack_code', 'discipline', 'track_type', 'sex',
        #     'shoeing_status', 'jockey_name', 'trainer_name', 'terrain_label', 'meeting_type',
        # ]

        self.numerical_features = NUMERICAL_FEATURES
        # [
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
                cols = {
                    "algorithm": "TEXT DEFAULT 'tabnet_only'",
                    "roi": "DOUBLE PRECISION DEFAULT 0",
                    "win_rate": "DOUBLE PRECISION DEFAULT 0",
                    "avg_odds": "DOUBLE PRECISION DEFAULT 0"
                }
                for col, dtype in cols.items():
                    cur.execute(f"ALTER TABLE ml_model_metrics ADD COLUMN IF NOT EXISTS {col} {dtype};")
                conn.commit()
        except Exception as e: self.logger.error(f"Migration error: {e}")
        finally: self.db.release_connection(conn)

    def train(self, test_days=90, val_days=60):
        self.logger.info("--- STARTING TABNET TOURNAMENT ---")
        self._ensure_table_schema()
        
        try:
            raw_df = self.loader.get_training_data()
            self.logger.info(f"Data Loaded: {raw_df.shape} rows")
        except Exception as e:
            self.logger.error(f"CRITICAL: Database connection failed. {e}")
            return

        if raw_df.empty: return

        # Targets specific disciplines only
        targets = ["global"]
        if 'discipline' in raw_df.columns:
            disciplines = [d for d in raw_df['discipline'].unique() if pd.notna(d)]
            targets.extend([str(d).lower() for d in disciplines])

        for target in targets:
            self.logger.info(f"--- Training Target: {target.upper()} ---")
            if target == "global":
                target_df = raw_df.copy()
            else:
                target_df = raw_df[raw_df['discipline'].str.lower() == target].copy()
            
            if len(target_df) < 1500:
                self.logger.warning(f"Skipping {target}: insufficient data ({len(target_df)} rows)")
                continue

            self._train_target(target_df, target, test_days, val_days)

    def _train_target(self, data: pd.DataFrame, target_name: str, test_days: int, val_days: int):
        engineer = PmuFeatureEngineer()
        context_encoder = RaceContextEncoder(group_col="race_id")
        
        df = engineer.fit_transform(data)
        df = context_encoder.transform(df)

        train_df, val_df, test_df = self._temporal_split(df, test_days, val_days)

        # Contextual Features added to the selection
        # features = self.numerical_features + self.categorical_features + [
        #     'odds_log', 'market_sentiment', 'reference_odds_rel_race',
        #     'reference_odds_rank_race', 'hist_avg_speed_z_race',
        #     'career_winnings_rank_race',
        #     'avg_speed_last_3',       # nouveau
        #     'days_since_last_race',   # nouveau
        #     'hist_pct_clean_runs',    # nouveau
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

        # Cross-validation and ensemble of TabNet folds
        y_vals = y_train.values
        folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train_df, y_vals))
        
        fold_models = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            self.logger.info(f"Training Fold {fold_idx + 1}/5 for {target_name}...")
            X_tr, X_val = X_train_df.iloc[train_idx], X_train_df.iloc[val_idx]
            y_tr, y_val = y_vals[train_idx], y_vals[val_idx]
            
            model = build_tabnet_model()
            model.fit(
                X_train=X_tr.values, y_train=y_tr,
                eval_set=[(X_val.values, y_val)],
                eval_name=['val'],
                eval_metric=['auc', 'logloss'],
                max_epochs=100, patience=15,
                batch_size=1024, virtual_batch_size=128,
                num_workers=0,
                drop_last=False
            )
            fold_models.append(model)

        self._log_feature_importances(fold_models, feature_names)

        # Use the wrapper defined at the top
        final_model = TabNetEnsembleWrapper(models=fold_models, feature_names=feature_names)

        full_pipeline = Pipeline([
            ('engineer', engineer),
            ('context', context_encoder),
            ('preprocessor', preprocessor),
            ('model', final_model)
        ])

        test_df_eval = test_df.copy()
        test_df_eval['proba'] = full_pipeline.predict_proba(test_df)[:, 1]
        metrics = self._calculate_metrics(test_df_eval)
        
        if metrics:
            self.logger.info(f"Target {target_name} Metrics: {metrics}")
            save_path = self.model_dir / f"model_{target_name}_tabnet.pkl"
            joblib.dump(full_pipeline, save_path)
            
            # Rolling Year Evaluation: On prend les 12 derniers mois de 'df' pour le dashboard
            max_date = df['program_date'].max()
            one_year_ago = max_date - pd.Timedelta(days=365)
            rolling_eval_df = df[df['program_date'] > one_year_ago].copy()
            
            self._generate_and_save_perf(full_pipeline, rolling_eval_df, target_name, "tabnet_only")

    def _log_feature_importances(self, fold_models, feature_names):
        importances = np.mean([m.feature_importances_ for m in fold_models], axis=0)
        fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
        print(f"\n--- TABNET FEATURE IMPORTANCES ({len(feature_names)} features) ---")
        print(fi_df.to_string(index=False))
        print("-" * 50 + "\n")

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
    trainer = TabNetTrainer(model_dir=str(data_dir))
    trainer.train()
