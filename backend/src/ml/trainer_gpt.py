# src/ml/trainer_gpt.py
import logging
import sys
import os
from pathlib import Path

import numpy as np
import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

from sklearnex import patch_sklearn
patch_sklearn(verbose=False)

from src.core.database import DatabaseManager
from category_encoders import CatBoostEncoder
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss

from src.ml.features import PmuFeatureEngineer, RaceContextEncoder, build_ltr_target
from src.ml.loader import DataLoader
from src.ml.feature_config import (
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    CONTEXTUAL_FEATURES
)

# Configuration du device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    import intel_npu_acceleration_library
    device = torch.device("npu")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

os.environ["SKLEARNEX_VERBOSE"] = "WARNING"
os.environ["SCIPY_ARRAY_API"] = "1"

# ------------------------------------------------
# Architecture PMU Transformer
# ------------------------------------------------

class PMUTransformer(nn.Module):
    def __init__(self, n_features, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.n_embd = n_embd
        self.block_size = block_size
        
        # Entrée : Projection des features
        self.feature_proj = nn.Linear(n_features, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        
        # Corps : Blocs Transformer
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd, 
                nhead=n_head, 
                dim_feedforward=4*n_embd,
                activation='relu', 
                batch_first=True,
                norm_first=True
            ) for _ in range(n_layer)
        ])
        
        # Tête : Calcul de la force (logit) de chaque cheval
        self.head = nn.Linear(n_embd, 1)

    def forward(self, x):
        # x shape: (num_horses, n_features)
        num_horses = x.size(0)
        
        # 1. Projection et Position
        x = self.feature_proj(x) # (num_horses, n_embd)
        # On sature à block_size si besoin, mais normalement num_horses < block_size
        pos_indices = torch.arange(min(num_horses, self.block_size), device=x.device)
        if num_horses > self.block_size:
             # Fallback pour les courses géantes (rare)
             x = x[:self.block_size]

        x = x + self.pos_emb(pos_indices)
        
        # 2. Transformer Blocks
        x = x.unsqueeze(0) 
        for block in self.blocks:
            x = block(x)
        
        # 3. Logits
        logits = self.head(x.squeeze(0)) # (num_horses, 1)
        return logits.flatten()

# ------------------------------------------------
# Wrapper sklearn pour PMUTransformer
# ------------------------------------------------

class GPTModelWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, n_features=None, n_embd=64, n_head=4, n_layer=3,
                 block_size=30, lr=1e-3, epochs=10, device=device, feature_names=None):
        self.n_features = n_features
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.model = None
        self.feature_names = list(feature_names) if feature_names is not None else None

    @property
    def feature_importances_(self):
        # Pour un transformer, l'importance est moins directe, on peut retourner les poids de la projection
        if self.model is not None:
            return torch.abs(self.model.feature_proj.weight).sum(dim=0).detach().cpu().numpy()
        return None

    def fit(self, X, y, groups=None):
        if groups is None:
            raise ValueError("GPTModelWrapper requires groups (race_id) for training.")

        # Mémorisation des noms de features si dispo
        if self.feature_names is None and hasattr(X, 'columns'):
            self.feature_names = list(X.columns)

        # Conversion en numpy si nécessaire
        X_val = X.values if hasattr(X, 'values') else X
        y_val = y.values if hasattr(y, 'values') else y

        self.n_features = X_val.shape[1]
        self.model = PMUTransformer(
            self.n_features, self.n_embd, self.n_head, self.n_layer, self.block_size
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        # Préparation des données par groupe (course)
        group_indices = []
        start = 0
        for g_size in groups:
            group_indices.append(slice(start, start + g_size))
            start += g_size

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            random_indices = np.random.permutation(len(group_indices))

            for idx in random_indices:
                slc = group_indices[idx]
                X_race = torch.tensor(X_val[slc], dtype=torch.float32).to(self.device)
                y_race = torch.tensor(y_val[slc], dtype=torch.float32).to(self.device)

                if len(X_race) == 0: continue

                # Forward
                logits = self.model(X_race)

                # Loss : On utilise une approche softmax sur la course pour le gagnant
                # On identifie l'index du gagnant (celui qui a le score LTR le plus élevé)
                winner_idx = torch.argmax(y_race)
                loss = F.cross_entropy(logits.unsqueeze(0), winner_idx.unsqueeze(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not fitted.")

        self.model.eval()
        X_val = X.values if hasattr(X, 'values') else X

        X_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = []
            for i in range(X_tensor.size(0)):
                # Simule une course d'un seul cheval pour satisfaire l'API point-wise
                out = self.model(X_tensor[i].unsqueeze(0))
                outputs.append(out.item())
            return np.array(outputs)

    def predict_proba(self, X):
        scores = self.predict(X)

        # Softmax par course si race_id disponible
        if hasattr(X, 'columns') and 'race_id' in X.columns:
            probas = np.zeros(len(scores))
            for _, idx in X.groupby('race_id').groups.items():
                s = scores[idx]
                e = np.exp(s - s.max())
                probas[idx] = e / e.sum()
        else:
            # Fallback : softmax global
            e = np.exp(scores - scores.max())
            probas = e / e.sum()

        return np.vstack([1 - probas, probas]).T

    def __sklearn_is_fitted__(self):
        return self.model is not None

# ------------------------------------------------
# GPT Trainer (Architecture calquée sur LTRTrainer)
# ------------------------------------------------

class GPTTrainer:
    def __init__(self, model_dir: str = "data"):
        self.logger = logging.getLogger("ML.GPTTrainer")
        self.model_dir = Path(model_dir)
        self.loader = DataLoader()
        self.db = DatabaseManager()

    def train(self):
        self.logger.info("--- STARTING GPT (TRANSFORMER) WALK-FORWARD TOURNAMENT ---")
        raw_df = self.loader.get_training_data()
        if raw_df.empty: return

        targets = ["global"]
        targets.extend([
            str(d).lower() for d in raw_df['discipline'].unique() if pd.notna(d)
        ])

        for target in targets:
            self.logger.info(f"--- GPT Target: {target.upper()} ---")
            target_df = (raw_df.copy() if target == "global"
                         else raw_df[raw_df['discipline'].str.lower() == target].copy())
            if len(target_df) < 1500: continue
            self._train_target(target_df, target)

    def _train_target(self, data: pd.DataFrame, target_name: str):
        engineer = PmuFeatureEngineer()
        context_encoder = RaceContextEncoder(group_col="race_id")

        df = engineer.fit_transform(data)
        df = context_encoder.transform(df)
        df['ltr_score'] = build_ltr_target(df)

        folds, final_train_df, final_test_df = self._walk_forward_split(df)
        if not folds: return

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
                self.logger.info(f"  {fold_year} → AUC={m['auc']:.4f} ROI={m['roi']:+.1f}%")

        avg_roi = np.mean(fold_rois) if fold_rois else -100
        if avg_roi > -15:
            self.logger.info(f"[{target_name}] Training final GPT model...")
            self._train_and_eval(
                final_train_df, final_test_df, engineer, context_encoder,
                target_name, save=True
            )

    def _train_and_eval(self, train_df, test_df, engineer, context_encoder,
                        target_name, save=False):
        features = [
            f for f in NUMERICAL_FEATURES + CATEGORICAL_FEATURES + CONTEXTUAL_FEATURES
            if f in train_df.columns
               and f != 'proba_tabnet'
        ]

        train_df = train_df.sort_values(['program_date', 'race_id']).reset_index(drop=True)
        test_df = test_df.sort_values(['program_date', 'race_id']).reset_index(drop=True)

        X_train = train_df[features]
        y_train = train_df['ltr_score']

        train_groups = train_df.groupby('race_id').size().values

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

        # Initialisation et entraînement du wrapper GPT
        wrapper = GPTModelWrapper(epochs=15, feature_names=feature_names)
        wrapper.fit(X_train_enc, y_train, groups=train_groups)

        full_pipeline = Pipeline([
            ('engineer', engineer),
            ('context', context_encoder),
            ('preprocessor', preprocessor),
            ('model', wrapper)
        ])

        # Évaluation
        test_eval = test_df.copy()
        test_enc = preprocessor.transform(test_df[features])
        scores = wrapper.predict(test_enc)
        test_eval['raw_score'] = scores

        test_eval['proba'] = 0.0
        for race_id, group in test_eval.groupby('race_id'):
            s = group['raw_score'].values
            e = np.exp(s - s.max())
            test_eval.loc[group.index, 'proba'] = e / e.sum()

        metrics = self._calculate_metrics(test_eval)

        if save and metrics:
            save_path = self.model_dir / f"model_{target_name}_gpt.pkl"
            joblib.dump(full_pipeline, save_path)
            self._generate_and_save_perf(full_pipeline, test_df, target_name, "gpt_transformer")
            self.logger.info(f"[{target_name}] GPT saved → ROI={metrics['roi']:+.1f}%")

        return metrics

    def _walk_forward_split(self, df: pd.DataFrame):
        df = df.copy()
        df['_year'] = df['program_date'].dt.year
        complete_years = [y for y in sorted(df['_year'].unique()) if y < pd.Timestamp.now().year]
        if len(complete_years) < 3: return [], None, None

        folds = [
            (df[df['_year'].isin(complete_years[:i])].drop(columns=['_year']),
             df[df['_year'] == complete_years[i]].drop(columns=['_year']))
            for i in range(2, len(complete_years))
        ]
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

        if total_bets == 0: return None
        return {
            'logloss': log_loss(group['is_winner'], group['proba']),
            'auc': roc_auc_score(group['is_winner'], group['proba']),
            'roi': (total_return - total_bets) / total_bets * 100,
            'win_rate': total_return / total_bets,
            'avg_odds': df['effective_odds'].mean(),
            'count': total_bets
        }

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
                    m.update({
                        'segment_type': seg_type,
                        'segment_value': str(keys if isinstance(keys, str) else keys[0]),
                        'month': 0 if month_col == 0 else keys[1]
                    })
                    perf_list.append(m)

        if perf_list:
            self._save_metrics_to_db_v2(target_name, algo_name, pd.DataFrame(perf_list))

    def _save_metrics_to_db_v2(self, model_name, algo_name, df):
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cur:
                for _, row in df.iterrows():
                    cur.execute("""
                        INSERT INTO ml_model_metrics
                        (model_name, algorithm, segment_type, segment_value, test_month, num_races, logloss, auc, roi, win_rate, avg_odds)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (model_name, algorithm, segment_type, segment_value, test_month)
                        DO UPDATE SET roi=EXCLUDED.roi, updated_at=NOW()
                    """, (model_name, algo_name, row['segment_type'], row['segment_value'], int(row['month']), int(row['count']), row['logloss'], row['auc'], row['roi'], row['win_rate'], row['avg_odds']))
                conn.commit()
        finally:
            self.db.release_connection(conn)

if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[3]
    data_dir = project_root / "data"
    trainer = GPTTrainer(model_dir=str(data_dir))
    trainer.train()
