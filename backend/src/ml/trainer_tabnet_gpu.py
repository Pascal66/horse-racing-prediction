import logging
import sys
from pathlib import Path

from sklearnex import patch_sklearn
patch_sklearn(verbose=False)

import numpy as np
import joblib
import pandas as pd
from category_encoders import CatBoostEncoder

from pytorch_tabnet.tab_model import TabNetClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from src.core.database import DatabaseManager
from src.ml.features import PmuFeatureEngineer, RaceContextEncoder
from src.ml.loader import DataLoader

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
os.environ["SCIPY_ARRAY_API"] = "1"

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

class TabNetTrainer:
    def __init__(self, model_dir: str = "data"):
        self.logger = logging.getLogger("ML.TabnetTrainer")
        self.model_dir = Path(model_dir)
        self.loader = DataLoader()
        self.db = DatabaseManager()

    def _ensure_table_schema(self):
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("ALTER TABLE ml_model_metrics ADD COLUMN IF NOT EXISTS algorithm TEXT DEFAULT 'tabnet_only';")
                cur.execute("ALTER TABLE ml_model_metrics ADD COLUMN IF NOT EXISTS roi DOUBLE PRECISION DEFAULT 0;")
                cur.execute("ALTER TABLE ml_model_metrics ADD COLUMN IF NOT EXISTS win_rate DOUBLE PRECISION DEFAULT 0;")
                cur.execute("ALTER TABLE ml_model_metrics ADD COLUMN IF NOT EXISTS avg_odds DOUBLE PRECISION DEFAULT 0;")
                conn.commit()
        except Exception as e: self.logger.error(f"Migration error: {e}")
        finally: self.db.release_connection(conn)

    def train(self):
        self.logger.info("--- STARTING TABNET WALK-FORWARD TOURNAMENT ---")
        self._ensure_table_schema()
        
        try:
            raw_df = self.loader.get_training_data()
            self.logger.info(f"Data Loaded: {raw_df.shape} rows")
        except Exception as e:
            self.logger.error(f"CRITICAL: Database connection failed. {e}")
            return

        if raw_df.empty: return

        targets = ["global"]
        if 'discipline' in raw_df.columns:
            disciplines = [d for d in raw_df['discipline'].unique() if pd.notna(d)]
            targets.extend([str(d).lower() for d in disciplines])

        for target in targets:
            self.logger.info(f"--- Training Target: {target.upper()} ---")
            target_df = raw_df.copy() if target == "global" else raw_df[
                raw_df['discipline'].str.lower() == target].copy()
            
            if len(target_df) < 1500:
                self.logger.warning(f"Skipping {target}: insufficient data")
                continue

            self._train_target(target_df, target)

    def _train_target(self, data: pd.DataFrame, target_name: str):
        engineer = PmuFeatureEngineer()
        context_encoder = RaceContextEncoder(group_col="race_id")
        
        df = engineer.fit_transform(data)
        df = context_encoder.transform(df)

        # 1. Split Walk-Forward pour stats et validation
        folds, final_train_df, final_test_df = self._walk_forward_split(df)
        
        if not folds:
            self.logger.warning(f"Insufficient history for WF on {target_name}")
            return

        # --- PHASE 1: Backtesting (Stats sur les années passées) ---
        # On utilise un mode "Fast" (1 seul fit) pour ne pas exploser le temps de calcul
        fold_rois = []
        for fold_idx, (train_df, test_df) in enumerate(folds):
            fold_year = test_df['program_date'].dt.year.iloc[0]
            self.logger.info(f"[{target_name}] WF Fold {fold_idx + 1} (Test year {fold_year})")
            
            metrics = self._train_and_eval(train_df, test_df, engineer, context_encoder, target_name, mode="fast")
            if metrics:
                fold_rois.append(metrics['roi'])
                self.logger.info(f"  Fold {fold_year} Result -> ROI: {metrics['roi']:+.1f}% | AUC: {metrics['auc']:.4f}")

        # --- PHASE 2: Production (Modèle final) ---
        # On utilise le mode "Full" (Ensemble 3-folds) sur TOUTES les données (Train + Test)
        # Mais seulement si le backtest n'est pas catastrophique (ROI moyen > -10% par ex)
        avg_wf_roi = np.mean(fold_rois) if fold_rois else -100
        
        if avg_wf_roi > -10:
            self.logger.info(f"[{target_name}] Backtest OK (Avg ROI: {avg_wf_roi:+.1f}%). Training PRODUCTION model...")
            
            # Pour la prod, on fusionne Train et Test pour ne pas perdre l'année la plus récente
            prod_df = pd.concat([final_train_df, final_test_df])
            
            # On utilise le dernier test_df comme set de validation final pour les métriques de rapport
            self._train_and_eval(prod_df, final_test_df, engineer, context_encoder, target_name, mode="full", save=True)
        else:
            self.logger.warning(f"[{target_name}] Backtest FAILED (Avg ROI: {avg_wf_roi:+.1f}%). Model not saved.")

    def _train_and_eval(self, train_df, test_df, engineer, context_encoder, target_name, mode="fast", save=False):
        features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + CONTEXTUAL_FEATURES + EXTRA_FEATURES
        features = [f for f in features if f in train_df.columns]

        X_train, y_train = train_df[features], train_df['is_winner']
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in features]
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
        y_vals = y_train.values

        if mode == "full":
            # Mode Production: On fait un vrai ensemble 3-folds
            folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(X_train_df, y_vals))
            fold_models = []
            for f_idx, (tr_idx, val_idx) in enumerate(folds):
                self.logger.info(f"    Training Fold {f_idx+1}/3...")
                model = build_tabnet_model()
                model.fit(
                    X_train=X_train_df.iloc[tr_idx].values, y_train=y_vals[tr_idx],
                    eval_set=[(X_train_df.iloc[val_idx].values, y_vals[val_idx])],
                    eval_metric=['auc'], max_epochs=50, patience=10, batch_size=1024, virtual_batch_size=128
                )
                fold_models.append(model)
            final_model = TabNetEnsembleWrapper(models=fold_models, feature_names=feature_names)
        else:
            # Mode Backtest: 1 seul split pour gagner du temps
            X_tr, X_val, y_tr, y_val = train_test_split(X_train_df, y_vals, test_size=0.2, stratify=y_vals, random_state=42)
            model = build_tabnet_model()
            model.fit(
                X_train=X_tr.values, y_train=y_tr,
                eval_set=[(X_val.values, y_val)],
                eval_metric=['auc'], max_epochs=50, patience=10, batch_size=1024, virtual_batch_size=128
            )
            final_model = TabNetEnsembleWrapper(models=[model], feature_names=feature_names)

        full_pipeline = Pipeline([
            ('engineer', engineer),
            ('context', context_encoder),
            ('preprocessor', preprocessor),
            ('model', final_model)
        ])

        # Évaluation sur le test_df fourni
        test_df_eval = test_df.copy()
        test_df_eval['proba'] = full_pipeline.predict_proba(test_df)[:, 1]
        metrics = self._calculate_metrics(test_df_eval)

        if save and metrics:
            save_path = self.model_dir / f"model_{target_name}_tabnet.pkl"
            joblib.dump(full_pipeline, save_path)
            self._generate_and_save_perf(full_pipeline, test_df, target_name, "tabnet_only")
        
        return metrics

    def _walk_forward_split(self, df: pd.DataFrame):
        df = df.copy()
        df['_year'] = df['program_date'].dt.year
        years = sorted(df['_year'].unique())
        current_year = pd.Timestamp.now().year
        complete_years = [y for y in years if y < current_year]

        if len(complete_years) < 3: return [], None, None

        folds = []
        # On crée des folds croissants : [2021], [2021, 2022], etc.
        for i in range(2, len(complete_years)):
            train_df = df[df['_year'].isin(complete_years[:i])].drop(columns=['_year'])
            test_df = df[df['_year'] == complete_years[i]].drop(columns=['_year'])
            if len(train_df) > 500:
                folds.append((train_df, test_df))

        # Les sets pour le split final (avant fusion prod)
        final_train = df[df['_year'].isin(complete_years[:-1])].drop(columns=['_year'])
        final_test = df[df['_year'] == complete_years[-1]].drop(columns=['_year'])

        return folds, final_train, final_test

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
            'win_rate': (win_count / total_bets * 100 if total_bets > 0 else 0),
            'avg_odds': df['effective_odds'].mean(),
            'count': total_bets
        }

    def _generate_and_save_perf(self, pipeline, test_df, target_name, algo_name):
        df = test_df.copy()
        try:
            df['proba'] = pipeline.predict_proba(test_df)[:, 1]
            df['month'] = df['program_date'].dt.month
            df['effective_odds'] = df['live_odds'].fillna(df['reference_odds']).fillna(1.0).clip(lower=1.05)
        except Exception: return

        perf_list = []
        segments = [('discipline_overall', 'discipline', 0), ('discipline_month', 'discipline', 'month')]
        for seg_type, col, month_col in segments:
            groupby_cols = [col] if month_col == 0 else [col, month_col]
            for keys, group in df.groupby(groupby_cols):
                if len(group) < 20: continue
                m = self._calculate_metrics(group)
                if m:
                    m.update({'segment_type': seg_type, 'segment_value': str(keys if isinstance(keys, str) else keys[0]), 'month': 0 if month_col == 0 else keys[1]})
                    perf_list.append(m)
        if perf_list: self._save_metrics_to_db_v2(target_name, algo_name, pd.DataFrame(perf_list))

    def _save_metrics_to_db_v2(self, model_name, algo_name, df):
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cur:
                for _, row in df.iterrows():
                    cur.execute("""
                        INSERT INTO ml_model_metrics (model_name, algorithm, segment_type, segment_value, test_month, num_races, logloss, auc, roi, win_rate, avg_odds)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (model_name, algorithm, segment_type, segment_value, test_month) 
                        DO UPDATE SET num_races=EXCLUDED.num_races, roi=EXCLUDED.roi, updated_at=NOW()
                    """, (model_name, algo_name, row['segment_type'], row['segment_value'], int(row['month']), int(row['count']), row['logloss'], row['auc'], row['roi'], row['win_rate'], row['avg_odds']))
                conn.commit()
        except Exception: pass
        finally: self.db.release_connection(conn)

if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[3]
    data_dir = project_root / "data"
    trainer = TabNetTrainer(model_dir=str(data_dir))
    trainer.train()
