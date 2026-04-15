import logging
import sys
from pathlib import Path

from sklearnex import patch_sklearn

from src.ml.ltr_bridge import LTRBridge

patch_sklearn(verbose=False)

import numpy as np
import joblib
import optuna
import pandas as pd
from category_encoders import CatBoostEncoder


from src.ml.tabnet_bridge import TabNetBridge

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
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
from src.ml.models import HyperStackModel
from src.ml.feature_config import (
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    CONTEXTUAL_FEATURES, EXTRA_FEATURES
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

import os
os.environ["SKLEARNEX_VERBOSE"] = "WARNING"
os.environ["SCIPY_ARRAY_API"] = "1"

def build_base_models(trial):
    models = {}
    models["xgb"] = XGBClassifier(
        n_estimators=trial.suggest_int("xgb_n_estimators", 300, 1000),
        max_depth=trial.suggest_int("xgb_max_depth", 3, 6),
        learning_rate=trial.suggest_float("xgb_lr", 0.01, 0.1, log=True),
        subsample=trial.suggest_float("xgb_sub", 0.7, 0.9),
        colsample_bytree=trial.suggest_float("xgb_col", 0.7, 0.9),
        tree_method="hist", n_jobs=-1, random_state=42
    )
    models["lgbm"] = LGBMClassifier(
        n_estimators=trial.suggest_int("lgbm_n_estimators", 300, 1000),
        max_depth=trial.suggest_int("lgbm_max_depth", 3, 6),
        learning_rate=trial.suggest_float("lgbm_lr", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("lgbm_leaves", 20, 60),
        n_jobs=-1, random_state=42, verbosity=-1
    )
    models["cat"] = CatBoostClassifier(
        iterations=trial.suggest_int("cat_iter", 300, 800),
        depth=trial.suggest_int("cat_depth", 4, 7),
        learning_rate=trial.suggest_float("cat_lr", 0.01, 0.1, log=True),
        verbose=False, random_seed=42
    )
    return models

def generate_oof(models, X, y, folds):
    n_models = len(models)
    oof = np.zeros((len(X), n_models))
    trained_models = []
    y_vals = y.values if hasattr(y, "values") else y

    for m_id, (name, model) in enumerate(models.items()):
        preds = np.zeros(len(X))
        fold_models = []
        for train_idx, val_idx in folds:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            m = clone(model)
            m.fit(X_train, y_vals[train_idx])
            preds[val_idx] = m.predict_proba(X_val)[:, 1]
            fold_models.append(m)
        oof[:, m_id] = preds
        trained_models.append((name, fold_models))
    return oof, trained_models

def objective(trial, X, y, race_ids, odds):
    y_vals = y.values if hasattr(y, "values") else y
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y_vals))
    models = build_base_models(trial)
    oof, _ = generate_oof(models, X, y, folds)

    stacker = LogisticRegression(max_iter=2000)
    oof_preds = np.zeros(len(X))
    for train_idx, val_idx in folds:
        stacker.fit(oof[train_idx], y_vals[train_idx])
        oof_preds[val_idx] = stacker.predict_proba(oof[val_idx])[:, 1]

    df_oof = pd.DataFrame({'proba': oof_preds, 'is_winner': y_vals, 'race_id': race_ids.values, 'effective_odds': odds.values})
    total_return, total_bets = 0, 0
    for _, group in df_oof.groupby('race_id'):
        best = group.loc[group['proba'].idxmax()]
        if best['is_winner'] == 1: total_return += best['effective_odds']
        total_bets += 1
    custom_roi = (total_return - total_bets) / total_bets if total_bets > 0 else -1
    return -custom_roi

class HyperStackTrainer:
    def __init__(self, n_trials=30, model_dir: str = "data"):
        self.logger = logging.getLogger("ML.HyperStackTrainer")
        self.model_dir = Path(model_dir)
        self.loader = DataLoader()
        self.db = DatabaseManager()
        self.n_trials = n_trials

    def _ensure_table_schema(self):
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("ALTER TABLE ml_model_metrics ADD COLUMN IF NOT EXISTS algorithm TEXT DEFAULT 'hyperstack_context';")
                cur.execute("ALTER TABLE ml_model_metrics ADD COLUMN IF NOT EXISTS roi DOUBLE PRECISION DEFAULT 0;")
                conn.commit()
        except Exception: pass
        finally: self.db.release_connection(conn)

    def train(self):
        self.logger.info("--- STARTING HYPERSTACK WALK-FORWARD TOURNAMENT ---")
        self._ensure_table_schema()
        raw_df = self.loader.get_training_data()
        if raw_df.empty: return

        targets = ["global"]
        if 'discipline' in raw_df.columns:
            targets.extend([str(d).lower() for d in raw_df['discipline'].unique() if pd.notna(d)])

        for target in targets:
            self.logger.info(f"--- AutoML for Target: {target.upper()} ---")
            target_df = raw_df.copy() if target == "global" else raw_df[raw_df['discipline'].str.lower() == target].copy()
            if len(target_df) < 1500: continue
            self._train_target(target_df, target)

    def _train_target(self, data, target_name):
        engineer = PmuFeatureEngineer()
        context_encoder = RaceContextEncoder(group_col="race_id")
        tabnet_bridge = TabNetBridge(tabnet_path=str(self.model_dir / f"model_{target_name}_tabnet.pkl"))

        df = engineer.fit_transform(data)
        df = context_encoder.transform(df)
        df = tabnet_bridge.fit_transform(df)

        folds, final_train_df, final_test_df = self._walk_forward_split(df)
        if not folds: return

        # Phase 1: Backtesting rapide
        fold_rois = []
        all_fold_metrics = []
        for f_idx, (tr_df, ts_df) in enumerate(folds):
            fold_year = ts_df['program_date'].dt.year.iloc[0]
            self.logger.info(f"[{target_name}] WF Fold {f_idx+1}/{len(folds)}")
            m = self._train_and_eval(tr_df, ts_df, engineer, context_encoder, tabnet_bridge, target_name, use_optuna=False)
            if m: fold_rois.append(m['roi'])

        # Phase 2: Production sur Full Data
        avg_roi = np.mean(fold_rois) if fold_rois else -100
        if avg_roi > -10:
            self.logger.info(f"[{target_name}] Backtest OK ({avg_roi:+.1f}%). Optimizing Production Model...")
            prod_df = pd.concat([final_train_df, final_test_df])
            metrics = self._train_and_eval(prod_df, final_test_df, engineer, context_encoder, tabnet_bridge, target_name, use_optuna=True, save=True)
            if metrics:
                metrics['test_year'] = fold_year
                all_fold_metrics.append(metrics)
                self.logger.info(
                    f"  Fold {fold_year} → AUC={metrics['auc']:.4f} "
                    f"ROI={metrics['roi']:+.1f}% count={metrics['count']}"
                )
        # Résumé walk-forward
        if all_fold_metrics:
            avg_roi = np.mean([m['roi'] for m in all_fold_metrics])
            avg_auc = np.mean([m['auc'] for m in all_fold_metrics])
            std_roi = np.std([m['roi'] for m in all_fold_metrics])
            self.logger.info(
                f"[{target_name}] Walk-forward: AUC={avg_auc:.4f} "
                f"ROI={avg_roi:+.1f}% ±{std_roi:.1f}% "
                f"({len(all_fold_metrics)} folds)"
            )

        # --- Modèle final sur toutes données sauf dernière année ---
        self.logger.info(f"[{target_name}] Entraînement modèle final...")
        final_metrics = self._train_and_eval(
            final_train_df, final_test_df, engineer, context_encoder,
            tabnet_bridge, target_name, save=True
        )
        if final_metrics:
            self.logger.info(
                f"[{target_name}] Final → AUC={final_metrics['auc']:.4f} "
                f"ROI={final_metrics['roi']:+.1f}% count={final_metrics['count']}"
            )

    def _train_and_eval(self, train_df, test_df, engineer, context_encoder, tabnet_bridge, target_name, use_optuna=False, save=False):
        features = [f for f in NUMERICAL_FEATURES + CATEGORICAL_FEATURES + CONTEXTUAL_FEATURES + EXTRA_FEATURES if f in train_df.columns]
        X_train, y_train = train_df[features], train_df['is_winner']
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in features]
        num_cols = [f for f in features if f not in cat_cols]

        preprocessor = ColumnTransformer([
            ('cat', CatBoostEncoder(cols=cat_cols), cat_cols),
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols)
        ])
        X_train_enc = preprocessor.fit_transform(X_train, y_train)
        feature_names = preprocessor.get_feature_names_out()
        X_train_df = pd.DataFrame(X_train_enc, columns=feature_names)

        if use_optuna:
            study = optuna.create_study(direction="minimize", storage=DB_URL, study_name=f"hs_{target_name}_prod", load_if_exists=True)
            odds = train_df['live_odds'].fillna(train_df['reference_odds']).fillna(1.0).clip(lower=1.05).reset_index(drop=True)
            r_ids = train_df['race_id'].reset_index(drop=True)
            study.optimize(lambda t: objective(t, X_train_df, y_train, r_ids, odds), n_trials=self.n_trials, n_jobs=1)
            best_params = study.best_params
        else:
            best_params = {'xgb_n_estimators': 500, 'xgb_max_depth': 4, 'xgb_lr': 0.05, 'xgb_sub': 0.8, 'xgb_col': 0.8,
                           'lgbm_n_estimators': 500, 'lgbm_max_depth': 4, 'lgbm_lr': 0.05, 'lgbm_leaves': 31,
                           'cat_iter': 400, 'cat_depth': 5, 'cat_lr': 0.05}

        best_trial = optuna.trial.FixedTrial(best_params)
        base_models = build_base_models(best_trial)
        cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train_df, y_train))
        oof, fold_models = generate_oof(base_models, X_train_df, y_train, cv)
        stacker = LogisticRegression(max_iter=2000).fit(oof, y_train)

        full_pipeline = Pipeline([
            ('engineer', engineer), ('context', context_encoder), ('tabnet_bridge', tabnet_bridge), # ('ltr_bridge', ltr_bridge),
            ('preprocessor', preprocessor), ('model', HyperStackModel(fold_models, stacker, feature_names=feature_names))
        ])

        test_eval = test_df.copy()
        test_eval['proba'] = full_pipeline.predict_proba(test_df)[:, 1]
        metrics = self._calculate_metrics(test_eval)

        if save and metrics:
            joblib.dump(full_pipeline, self.model_dir / f"model_{target_name}.pkl")
            self._generate_and_save_perf(full_pipeline, test_df, target_name, "hyperstack_context")
        return metrics

    def _walk_forward_split(self, df: pd.DataFrame):
        df = df.copy()
        df['_year'] = df['program_date'].dt.year
        complete_years = [y for y in sorted(df['_year'].unique()) if y < pd.Timestamp.now().year]
        if len(complete_years) < 3: return [], None, None
        folds = [(df[df['_year'].isin(complete_years[:i])].drop(columns=['_year']), df[df['_year'] == complete_years[i]].drop(columns=['_year'])) for i in range(2, len(complete_years))]
        return folds, df[df['_year'].isin(complete_years[:-1])].drop(columns=['_year']), df[df['_year'] == complete_years[-1]].drop(columns=['_year'])

    def _calculate_metrics(self, group):
        if 'proba' not in group.columns or group['is_winner'].nunique() < 2: return None
        df = group.copy()
        df['effective_odds'] = df['live_odds'].fillna(df['reference_odds']).fillna(1.0).clip(lower=1.05)
        total_return, total_bets = 0, 0
        for _, race in df.groupby('race_id'):
            best = race.loc[race['proba'].idxmax()]
            if best['is_winner'] == 1: total_return += best['effective_odds']
            total_bets += 1
        return {'logloss': log_loss(group['is_winner'], group['proba']), 'auc': roc_auc_score(group['is_winner'], group['proba']),
                'roi': ((total_return - total_bets) / total_bets * 100) if total_bets > 0 else 0,
                'win_rate': (total_return / total_bets if total_bets > 0 else 0), 'count': total_bets, 'avg_odds': df['effective_odds'].mean()}

    def _generate_and_save_perf(self, pipeline, test_df, target_name, algo_name):
        df = test_df.copy()
        df['proba'] = pipeline.predict_proba(test_df)[:, 1]
        df['month'] = df['program_date'].dt.month
        df['effective_odds'] = df['live_odds'].fillna(df['reference_odds']).fillna(1.0).clip(lower=1.05)
        segments = [
            ('discipline_overall', 'discipline', 0),
            ('discipline_month', 'discipline', 'month'),
            ('track_month', 'meeting_code', 'month')
        ]

        perf_list = []
        for seg_type, col, month_col in segments:
            for keys, group in df.groupby([col] if month_col == 0 else [col, month_col]):
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
                    cur.execute("""INSERT INTO ml_model_metrics (model_name, algorithm, segment_type, segment_value, test_month, num_races, logloss, auc, roi, win_rate, avg_odds)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (model_name, algorithm, segment_type, segment_value, test_month) 
                        DO UPDATE SET roi=EXCLUDED.roi, updated_at=NOW()""", (model_name, algo_name, row['segment_type'], row['segment_value'], int(row['month']), int(row['count']), row['logloss'], row['auc'], row['roi'], row['win_rate'], row['avg_odds']))
                conn.commit()
        finally: self.db.release_connection(conn)

if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[3]
    data_dir = project_root / "data"
    trainer = HyperStackTrainer(n_trials=3, model_dir=str(data_dir))
    trainer.train()
