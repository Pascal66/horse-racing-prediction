import time
import joblib
import logging
import sys
import numpy as np
import optuna
import pandas as pd
from typing import Dict, Any, List, Union
from category_encoders import CatBoostEncoder
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier, XGBRanker
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score
from pathlib import Path
from src.core.config import DB_URL
from src.core.database import DatabaseManager

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
        self.db = DatabaseManager()

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

    def _ensure_table_schema(self):
        """Migre le schéma de la table si nécessaire."""
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cur:
                # Création initiale si absent
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ml_model_metrics (
                        model_name TEXT,
                        segment_type TEXT,
                        segment_value TEXT,
                        test_month INTEGER,
                        num_races INTEGER,
                        logloss DOUBLE PRECISION,
                        auc DOUBLE PRECISION,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (model_name, segment_type, segment_value, test_month)
                    )
                """)
                # Ajout des nouvelles colonnes si absentes
                cols_to_add = {
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
                conn.commit()
        except Exception as e:
            self.logger.error(f"Migration error: {e}")
        finally:
            self.db.release_connection(conn)

    def _add_features(self, df):
        df['odds_log'] = np.log1p(df['reference_odds'].fillna(20))
        df['earnings_log'] = np.log1p(df['career_winnings'].fillna(0))
        df['speed_rel'] = df['hist_avg_speed'] / df.groupby('race_id')['hist_avg_speed'].transform('mean')
        df['odds_rank_pct'] = df.groupby('race_id')['reference_odds'].rank(pct=True)
        return df

    def train(self, test_days: int = 90, val_days: int = 60) -> None:
        self.logger.info("--- STARTING MULTI-MODEL TRAINING PIPELINE ---")
        self._ensure_table_schema()
        
        try:
            raw_df = self.loader.get_training_data()
            self.logger.info(f"Data Loaded: {raw_df.shape} rows")
        except Exception as e:
            self.logger.error(f"CRITICAL: Database connection failed. {e}")
            return

        if raw_df.empty: return

        # 1. GLOBAL Model
        self.logger.info("--- Training GLOBAL (Fallback) Model ---")
        self._train_and_save(raw_df, "global", test_days, val_days)

        # 2. Specialty Models
        if 'discipline' in raw_df.columns:
            disciplines = raw_df['discipline'].unique()
            for discipline in disciplines:
                if pd.isna(discipline) or str(discipline).strip() == "": continue
                self.logger.info(f"--- Training Specialty Model: {discipline} ---")
                discipline_df = raw_df[raw_df['discipline'] == discipline].copy()
                if len(discipline_df) > 1000:
                    self._train_and_save(discipline_df, str(discipline).lower(), test_days, val_days)

    def _save_metrics_to_db(self, model_name: str, metrics_df: pd.DataFrame):
        """Enregistre les performances segmentées pour l'UI / Back-Analysis"""
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cur:
                for _, row in metrics_df.iterrows():
                    cur.execute("""
                        INSERT INTO ml_model_metrics (model_name, segment_type, segment_value, test_month, num_races, logloss, auc, roi, win_rate, avg_odds)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (model_name, segment_type, segment_value, test_month) 
                        DO UPDATE SET num_races=EXCLUDED.num_races, logloss=EXCLUDED.logloss, auc=EXCLUDED.auc, 
                        roi=EXCLUDED.roi, win_rate=EXCLUDED.win_rate, avg_odds=EXCLUDED.avg_odds, updated_at=NOW()
                    """, (
                        model_name, row['segment_type'], str(row['segment_value']), int(row['month']), 
                        int(row['count']), float(row['logloss']), float(row['auc']), 
                        float(row['roi']), float(row['win_rate']), float(row['avg_odds'])
                    ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")
        finally:
            self.db.release_connection(conn)

    def _calculate_segment_metrics(self, group: pd.DataFrame, segment_type: str, segment_value: str, month: int) -> Dict[str, Any]:
        """Calculates all relevant metrics for a given segment group."""
        if len(group) < 20: 
            return None

        y_true_seg = group['is_winner']
        y_pred_seg = group['proba']

        seg_logloss = log_loss(y_true_seg, y_pred_seg)
        seg_auc = 0.5
        if y_true_seg.nunique() > 1:
            seg_auc = roc_auc_score(y_true_seg, y_pred_seg)

        # ROI & Win Rate calculation
        total_return = 0
        total_bets = 0
        num_predicted_winners_correct = 0
        sum_odds_predicted_winners = 0

        for race_id in group['race_id'].unique():
            race_group = group[group['race_id'] == race_id]
            if not race_group.empty:
                predicted_winner = race_group.loc[race_group['proba'].idxmax()]
                if predicted_winner['is_winner'] == 1:
                    total_return += predicted_winner['effective_odds']
                    num_predicted_winners_correct += 1
                sum_odds_predicted_winners += predicted_winner['effective_odds']
                total_bets += 1

        seg_roi = ((total_return - total_bets) / total_bets * 100) if total_bets > 0 else 0
        seg_win_rate = (num_predicted_winners_correct / total_bets * 100) if total_bets > 0 else 0
        seg_avg_odds = (sum_odds_predicted_winners / total_bets) if total_bets > 0 else 0

        return {
            'segment_type': segment_type,
            'segment_value': segment_value,
            'month': month,
            'count': total_bets,
            'logloss': seg_logloss,
            'auc': seg_auc,
            'roi': seg_roi,
            'win_rate': seg_win_rate,
            'avg_odds': seg_avg_odds
        }

    def _train_and_save(self, data: pd.DataFrame, model_name: str, test_days: int, val_days: int) -> None:
        engineer = PmuFeatureEngineer()
        full_df = engineer.fit_transform(data)
        full_df = self._add_features(full_df)

        max_date = full_df['program_date'].max()
        test_cutoff = max_date - pd.Timedelta(days=test_days)
        val_cutoff = test_cutoff - pd.Timedelta(days=val_days)

        train_df = full_df[full_df['program_date'] <= val_cutoff]
        val_df = full_df[(full_df['program_date'] > val_cutoff) & (full_df['program_date'] <= test_cutoff)]
        test_df = full_df[full_df['program_date'] > test_cutoff]

        if len(train_df) < 500: return

        features = self.numerical_features + self.categorical_features + ['odds_log', 'earnings_log', 'speed_rel', 'odds_rank_pct']
        X_train, y_train = train_df[features], train_df['is_winner']
        X_val, y_val = val_df[features], val_df['is_winner']
        X_test, y_test = test_df[features], test_df['is_winner']

        preprocessor = ColumnTransformer(transformers=[
            ('cat', CatBoostEncoder(cols=self.categorical_features), self.categorical_features),
            ('num', 'passthrough', self.numerical_features)
        ])

        X_train_enc = preprocessor.fit_transform(X_train, y_train)
        X_val_enc = preprocessor.transform(X_val)
        X_test_enc = preprocessor.transform(X_test)

        # 1. Base Model
        base_params = {
            'n_estimators': 2500, 'max_depth': 4, 'learning_rate': 0.03,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'tree_method': 'auto', 
            'early_stopping_rounds': 50, 'random_state': 42, 'eval_metric': 'logloss'
        }
        base_xgb = XGBClassifier(**base_params)
        base_xgb.fit(X_train_enc, y_train, eval_set=[(X_val_enc, y_val)], verbose=False)
        
        # 2. Optuna Optimization
        def objective(trial):
            p = base_params.copy()
            p.update({
                "n_estimators": trial.suggest_int("n_estimators", 1000, 4000),
                "max_depth": trial.suggest_int("max_depth", 3, 5),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 2),
            })
            m = XGBClassifier(**p)
            m.fit(X_train_enc, y_train, eval_set=[(X_val_enc, y_val)], verbose=False)
            return log_loss(y_val, m.predict_proba(X_val_enc)[:, 1])

        study = optuna.create_study(direction="minimize", study_name=f"optuna_{model_name}", storage=DB_URL, load_if_exists=True)
        study.optimize(objective, n_trials=10)
        
        base_val_loss = log_loss(y_val, base_xgb.predict_proba(X_val_enc)[:, 1])
        if (base_val_loss - study.best_value) / base_val_loss > 0.005:
            final_xgb = XGBClassifier(**{**base_params, **study.best_params})
            final_xgb.fit(X_train_enc, y_train, eval_set=[(X_val_enc, y_val)], verbose=False)
        else:
            final_xgb = base_xgb

        # 3. Calibration
        calibrated_model = CalibratedClassifierCV(FrozenEstimator(final_xgb), method='sigmoid')
        calibrated_model.fit(X_val_enc, y_val)

        # 4. Back-Analysis Metrics Generation
        test_df = test_df.copy()
        test_df['proba'] = calibrated_model.predict_proba(X_test_enc)[:, 1]
        test_df['month'] = test_df['program_date'].dt.month
        test_df['effective_odds'] = test_df['live_odds'].apply(lambda x: x if x and x > 1.1 else None)
        test_df['effective_odds'] = test_df['effective_odds'].fillna(test_df['reference_odds']).fillna(1.0).clip(lower=1.05)

        perf_list = []
        for (track, month), group in test_df.groupby(['racetrack_code', 'month']):
            metrics = self._calculate_segment_metrics(group, 'track_month', track, month)
            if metrics: perf_list.append(metrics)
        
        for (discipline, month), group in test_df.groupby(['discipline', 'month']):
            metrics = self._calculate_segment_metrics(group, 'discipline_month', discipline, month)
            if metrics: perf_list.append(metrics)

        for discipline, group in test_df.groupby('discipline'):
            metrics = self._calculate_segment_metrics(group, 'discipline_overall', discipline, 0)
            if metrics: perf_list.append(metrics)
        
        if perf_list:
            self._save_metrics_to_db(model_name, pd.DataFrame(perf_list))

        # 5. Save Pipeline
        full_pipeline = Pipeline([('engineer', engineer), ('preprocessor', preprocessor), ('model', calibrated_model)])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(full_pipeline, self.model_dir / f"model_{model_name}.pkl")
        self.logger.info(f"[{model_name}] SUCCESS: Training complete and metrics saved.")

if __name__ == "__main__":
    trainer = XGBoostTrainer()
    trainer.train()
