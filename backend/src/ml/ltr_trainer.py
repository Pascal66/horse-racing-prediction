# src/ml/ltr_trainer.py
import logging
import sys
from pathlib import Path

import numpy as np
import joblib
import pandas as pd
from sklearnex import patch_sklearn
patch_sklearn(verbose=False)

from category_encoders import CatBoostEncoder
from lightgbm import LGBMRanker
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss


from src.ml.features import PmuFeatureEngineer, RaceContextEncoder, build_ltr_target
from src.ml.loader import DataLoader
from src.ml.safe_loader import safe_load
from src.ml.feature_config import (
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    CONTEXTUAL_FEATURES, FEATURE_DEFAULTS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

import os
os.environ["SKLEARNEX_VERBOSE"] = "WARNING"
os.environ["SCIPY_ARRAY_API"] = "1"

# ------------------------------------------------
# Wrapper sklearn pour LGBMRanker
# ------------------------------------------------

class LTRRankerWrapper(BaseEstimator, RegressorMixin):
    """
    Wraps LGBMRanker pour l'intégrer dans un sklearn Pipeline.
    Stocke les race_ids pour reconstruire les groupes à l'inférence.
    À l'inférence, retourne des scores continus (pas de groupes nécessaires).
    """

    def __init__(self, model=None, feature_names=None):
        self.model = model
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        """Retourne les scores bruts LTR (valeur continue)."""
        if self.model is None:
            raise ValueError("Model not fitted.")
        X_vals = X.values if hasattr(X, 'values') else X
        return self.model.predict(X_vals)

    def predict_proba(self, X):
        """
        Convertit les scores LTR en pseudo-probabilités par course.
        Nécessite race_id dans X pour normaliser par course.
        Si pas de race_id, normalise globalement (fallback inférence).
        """
        scores = self.predict(X)

        # Softmax par course si race_id disponible
        if hasattr(X, 'columns') and 'race_id' in X.columns:
            probas = np.zeros(len(scores))
            for _, idx in X.groupby('race_id').groups.items():
                s = scores[idx]
                # Softmax — transforme scores en distribution de probabilités
                e = np.exp(s - s.max())
                probas[idx] = e / e.sum()
        else:
            # Fallback : softmax global (inférence sans race_id)
            e = np.exp(scores - scores.max())
            probas = e / e.sum()

        return np.vstack([1 - probas, probas]).T

    def __sklearn_is_fitted__(self):
        return self.model is not None


# ------------------------------------------------
# Fonction de build du ranker avec hyperparamètres
# ------------------------------------------------

def build_ltr_model(n_estimators=500, max_depth=6, learning_rate=0.05,
                    num_leaves=31, min_child_samples=20,
                    reg_alpha=0.1, reg_lambda=1.0):
    return LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        ndcg_eval_at=[1, 3, 5],  # NDCG@1 = trouver le gagnant
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        n_jobs=-1,
        random_state=42,
        verbosity=-1
    )


# ------------------------------------------------
# Trainer LTR
# ------------------------------------------------

class LTRTrainer:
    def __init__(self, model_dir: str = "data"):
        self.logger = logging.getLogger("ML.LTRTrainer")
        self.model_dir = Path(model_dir)
        self.loader = DataLoader()

    def train(self):
        self.logger.info("--- STARTING LTR WALK-FORWARD TOURNAMENT ---")
        raw_df = self.loader.get_training_data()
        if raw_df.empty: return

        targets = ["global"]
        targets.extend([
            str(d).lower() for d in raw_df['discipline'].unique() if pd.notna(d)
        ])

        for target in targets:
            self.logger.info(f"--- LTR Target: {target.upper()} ---")
            target_df = (raw_df.copy() if target == "global"
                         else raw_df[raw_df['discipline'].str.lower() == target].copy())
            if len(target_df) < 1500: continue
            self._train_target(target_df, target)

    def _train_target(self, data: pd.DataFrame, target_name: str):
        engineer = PmuFeatureEngineer()
        context_encoder = RaceContextEncoder(group_col="race_id")

        df = engineer.fit_transform(data)
        df = context_encoder.transform(df)

        # Target LTR
        df['ltr_score'] = build_ltr_target(df)

        folds, final_train_df, final_test_df = self._walk_forward_split(df)
        if not folds: return

        # Walk-forward backtest
        fold_rois = []
        for fold_idx, (train_df, test_df) in enumerate(folds):
            fold_year = test_df['program_date'].dt.year.iloc[0]
            self.logger.info(f"[{target_name}] Fold {fold_idx + 1} — test {fold_year}")

            m = self._train_and_eval(
                train_df, test_df, engineer, context_encoder,
                target_name, save=False
            )
            if m:
                fold_rois.append(m['roi'])
                self.logger.info(
                    f"  {fold_year} → AUC={m['auc']:.4f} "
                    f"ROI={m['roi']:+.1f}% count={m['count']}"
                )

        avg_roi = np.mean(fold_rois) if fold_rois else -100
        std_roi = np.std(fold_rois) if fold_rois else 0
        self.logger.info(
            f"[{target_name}] Walk-forward: ROI={avg_roi:+.1f}% "
            f"±{std_roi:.1f}% ({len(fold_rois)} folds)"
        )

        # Modèle final
        if avg_roi > -15:
            self.logger.info(f"[{target_name}] Training final LTR model...")
            self._train_and_eval(
                final_train_df, final_test_df, engineer, context_encoder,
                target_name, save=True
            )

    def _train_and_eval(self, train_df, test_df, engineer, context_encoder,
                        target_name, save=False):
        features = [
            f for f in NUMERICAL_FEATURES + CATEGORICAL_FEATURES + CONTEXTUAL_FEATURES
            if f in train_df.columns
                       and f != 'proba_tabnet'  # pas de stacking TabNet ici
        ]

        # Trier par race_id — obligatoire pour LGBMRanker
        train_df = train_df.sort_values('race_id').reset_index(drop=True)
        test_df = test_df.sort_values('race_id').reset_index(drop=True)

        X_train = train_df[features]
        y_train = train_df['ltr_score']
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in features]
        num_cols = [f for f in features if f not in cat_cols]

        # Groupes pour LGBMRanker : nb de chevaux par course, dans l'ordre
        train_groups = train_df.groupby('race_id').size().values
        test_groups = test_df.groupby('race_id').size().values

        preprocessor = ColumnTransformer([
            ('cat', CatBoostEncoder(cols=cat_cols), cat_cols),
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols)
        ])

        X_train_enc = preprocessor.fit_transform(X_train, y_train)
        feature_names = preprocessor.get_feature_names_out()

        # Entraînement LGBMRanker
        ranker = build_ltr_model()
        ranker.fit(
            X_train_enc, y_train.values,
            group=train_groups,
            eval_set=[(X_train_enc, y_train.values)],
            eval_group=[train_groups],
        )

        wrapper = LTRRankerWrapper(model=ranker, feature_names=feature_names)

        # Pipeline complet — le wrapper gère l'inférence sans groupes
        full_pipeline = Pipeline([
            ('engineer', engineer),
            ('context', context_encoder),
            ('preprocessor', preprocessor),
            ('model', wrapper)
        ])

        # Évaluation : on convertit les scores LTR en probas par course
        test_enc = preprocessor.transform(test_df[features])
        scores = ranker.predict(test_enc)

        test_eval = test_df.copy()
        test_eval['raw_score'] = scores

        # Normalisation par course → pseudo-probabilité
        test_eval['proba'] = 0.0
        for race_id, group in test_eval.groupby('race_id'):
            s = group['raw_score'].values
            e = np.exp(s - s.max())
            test_eval.loc[group.index, 'proba'] = e / e.sum()

        metrics = self._calculate_metrics(test_eval)

        if save and metrics:
            save_path = self.model_dir / f"model_{target_name}_ltr.pkl"
            joblib.dump(full_pipeline, save_path)
            self.logger.info(
                f"[{target_name}] LTR saved → "
                f"AUC={metrics['auc']:.4f} ROI={metrics['roi']:+.1f}%"
            )

        return metrics

    def _walk_forward_split(self, df: pd.DataFrame):
        df = df.copy()
        df['_year'] = df['program_date'].dt.year
        complete_years = [
            y for y in sorted(df['_year'].unique())
            if y < pd.Timestamp.now().year
        ]
        if len(complete_years) < 3: return [], None, None

        folds = [
            (
                df[df['_year'].isin(complete_years[:i])].drop(columns=['_year']),
                df[df['_year'] == complete_years[i]].drop(columns=['_year'])
            )
            for i in range(2, len(complete_years))
        ]
        return (
            folds,
            df[df['_year'].isin(complete_years[:-1])].drop(columns=['_year']),
            df[df['_year'] == complete_years[-1]].drop(columns=['_year'])
        )

    def _calculate_metrics(self, group):
        if 'proba' not in group.columns or group['is_winner'].nunique() < 2:
            return None
        df = group.copy()
        df['effective_odds'] = (
            df['live_odds'].fillna(df['reference_odds']).fillna(1.0).clip(lower=1.05)
        )
        total_return, total_bets = 0, 0
        for _, race in df.groupby('race_id'):
            best = race.loc[race['proba'].idxmax()]
            if best['is_winner'] == 1:
                total_return += best['effective_odds']
            total_bets += 1

        if total_bets == 0: return None
        return {
            'logloss': log_loss(group['is_winner'], group['proba']),
            'auc': roc_auc_score(group['is_winner'], group['proba']),
            'roi': (total_return - total_bets) / total_bets * 100,
            'win_rate': total_return / total_bets,
            'avg_odds': df['effective_odds'].mean(),
            'count': total_bets
        }


if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[3]
    data_dir = project_root / "data"
    trainer = LTRTrainer(model_dir=str(data_dir))
    trainer.train()
