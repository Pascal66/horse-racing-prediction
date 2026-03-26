import time
import joblib
import logging
import sys
import numpy as np
import optuna
import pandas as pd
from typing import Dict, Any, List, Union, Optional
from category_encoders import CatBoostEncoder
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier, XGBRanker
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
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

class MultiModelTrainer:
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
            'career_winnings', 'relative_winnings', 'winnings_rank_in_race',
            'odds_rank_in_race', 'reference_odds',
            'is_debutant', 'race_month', 'hist_avg_speed', 'hist_earnings'
        ]

    def _ensure_table_schema(self):
        """Migre le schéma de la table pour inclure l'algorithme."""
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cur:
                # Table principale des metrics
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ml_model_metrics (
                        model_name TEXT,
                        algorithm TEXT DEFAULT 'xgboost',
                        segment_type TEXT,
                        segment_value TEXT,
                        test_month INTEGER,
                        num_races INTEGER,
                        logloss DOUBLE PRECISION,
                        auc DOUBLE PRECISION,
                        roi DOUBLE PRECISION,
                        win_rate DOUBLE PRECISION,
                        avg_odds DOUBLE PRECISION,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (model_name, algorithm, segment_type, segment_value, test_month)
                    )
                """)
                # Migration de la PK si nécessaire (si l'ancienne n'avait pas 'algorithm')
                cur.execute("""
                    DO $$ 
                    BEGIN 
                        IF EXISTS (SELECT 1 FROM information_schema.table_constraints 
                                   WHERE constraint_name='ml_model_metrics_pkey' AND table_name='ml_model_metrics') THEN
                            ALTER TABLE ml_model_metrics DROP CONSTRAINT ml_model_metrics_pkey;
                            ALTER TABLE ml_model_metrics ADD PRIMARY KEY (model_name, algorithm, segment_type, segment_value, test_month);
                        END IF;
                    EXCEPTION WHEN OTHERS THEN 
                        -- PK already correct or table empty
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
        self.logger.info("--- STARTING MULTI-MODEL TOURNAMENT ---")
        self._ensure_table_schema()
        
        try:
            raw_df = self.loader.get_training_data()
            self.logger.info(f"Data Loaded: {raw_df.shape} rows")
        except Exception as e:
            self.logger.error(f"CRITICAL: Database connection failed. {e}")
            return

        if raw_df.empty: return

        # Entraîner sur Global et les spécialités
        targets = ["global"]
        if 'discipline' in raw_df.columns:
            disciplines = raw_df['discipline'].unique()
            targets.extend([str(d).lower() for d in disciplines if pd.notna(d) and str(d).strip() != ""])

        for target in targets:
            self.logger.info(f"--- Tournament for Target: {target.upper()} ---")
            target_df = raw_df.copy() if target == "global" else raw_df[raw_df['discipline'].str.lower() == target].copy()
            
            if len(target_df) < 1000: continue
            
            # Entraînement des 4 modèles
            self._train_tournament(target_df, target, test_days, val_days)

    def _train_tournament(self, data: pd.DataFrame, target_name: str, test_days: int, val_days: int):
        engineer = PmuFeatureEngineer()
        full_df = engineer.fit_transform(data)
        full_df = self._add_features(full_df)

        max_date = full_df['program_date'].max()
        test_cutoff = max_date - pd.Timedelta(days=test_days)
        val_cutoff = test_cutoff - pd.Timedelta(days=val_days)

        train_df = full_df[full_df['program_date'] <= val_cutoff]
        val_df = full_df[(full_df['program_date'] > val_cutoff) & (full_df['program_date'] <= test_cutoff)]
        test_df = full_df[full_df['program_date'] > test_cutoff]

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

        # Liste des architectures à tester
        architectures = {
            "xgboost": XGBClassifier(n_estimators=2000, max_depth=4, learning_rate=0.03, tree_method='auto', random_state=42),
            "lightgbm": LGBMClassifier(n_estimators=2000, max_depth=4, learning_rate=0.03, boosting_type='dart', random_state=42),
            "catboost": CatBoostClassifier(iterations=2000, depth=4, learning_rate=0.03, verbose=False, random_seed=42),
            "ranker": XGBRanker(objective='rank:pairwise', n_estimators=2000, max_depth=4, learning_rate=0.05, random_state=42)
        }

        best_algo = None
        best_val_loss = float('inf')

        for name, model in architectures.items():
            try:
                self.logger.info(f"Training Algorithm: {name}...")
                if name == "ranker":
                    # Special handling for Ranker
                    group_train = train_df.groupby('race_id').size().to_list()
                    group_val = val_df.groupby('race_id').size().to_list()
                    model.fit(X_train_enc, y_train, group=group_train, eval_set=[(X_val_enc, y_val)], eval_group=[group_val], verbose=False)
                else:
                    model.fit(X_train_enc, y_train, eval_set=[(X_val_enc, y_val)], verbose=False)

                # Optuna (Only for the best baseline or all? Let's do a quick run for each)
                # For this tournament, we skip full Optuna to save time, using solid defaults.
                
                # Calibration (Mandatory for probability comparison)
                calibrated = CalibratedClassifierCV(FrozenEstimator(model), method='sigmoid')
                calibrated.fit(X_val_enc, y_val)

                # Eval
                val_probs = calibrated.predict_proba(X_val_enc)[:, 1]
                v_loss = log_loss(y_val, val_probs)
                self.logger.info(f"Algorithm {name} Val Loss: {v_loss:.4f}")

                # Save performance
                self._generate_and_save_perf(calibrated, X_test_enc, test_df, target_name, name)

                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    best_algo = (name, calibrated)
            except Exception as e:
                self.logger.error(f"Error training {name}: {e}")

        # Save the winner of the tournament as the official target model
        if best_algo:
            name, final_model = best_algo
            self.logger.info(f"Tournament WINNER for {target_name}: {name.upper()}")
            full_pipeline = Pipeline([('engineer', engineer), ('preprocessor', preprocessor), ('model', final_model)])
            save_path = self.model_dir / f"model_{target_name}.pkl"
            joblib.dump(full_pipeline, save_path)

    def _generate_and_save_perf(self, model, X_test_enc, test_df, target_name, algo_name):
        """Calcule et sauvegarde les performances détaillées par algorithme."""
        df = test_df.copy()
        df['proba'] = model.predict_proba(X_test_enc)[:, 1]
        df['month'] = df['program_date'].dt.month
        df['effective_odds'] = df['live_odds'].fillna(df['reference_odds']).fillna(1.0).clip(lower=1.05)

        perf_list = []
        # Segments: Overall, Month, Track
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

    def _calculate_metrics(self, group):
        y_true = group['is_winner']
        y_pred = group['proba']
        if y_true.nunique() < 2: return None

        # ROI & Win Rate (Simulated 1 unit on top predicted)
        total_return = 0
        total_bets = 0
        for race_id in group['race_id'].unique():
            race_group = group[group['race_id'] == race_id]
            best_horse = race_group.loc[race_group['proba'].idxmax()]
            if best_horse['is_winner'] == 1:
                total_return += best_horse['effective_odds']
            total_bets += 1

        return {
            'logloss': log_loss(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_pred),
            'roi': ((total_return - total_bets) / total_bets * 100),
            'win_rate': (group[group['is_winner']==1]['proba'].idxmax() == group['is_winner'].idxmax()) if False else 0, # simplified
            'win_rate': (total_return > 0), # placeholder
            'count': total_bets,
            'avg_odds': group['effective_odds'].mean(),
            'win_rate': (total_return / total_bets if total_bets > 0 else 0) # Real win rate of the model favorite
        }

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
        except Exception as e: self.logger.error(f"DB Save error: {e}")
        finally: self.db.release_connection(conn)

if __name__ == "__main__":
    trainer = MultiModelTrainer()
    trainer.train()
