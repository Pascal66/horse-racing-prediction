import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

# Configuration du device (portable)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    import intel_npu_acceleration_library

    device = torch.device("npu")
except ImportError:
    pass


class PMUTransformer(nn.Module):
    def __init__(self, n_features, n_embd, n_head, n_layer, block_size, dropout=0.1):
        super().__init__()
        self.n_embd = n_embd
        self.block_size = block_size

        self.feature_proj = nn.Sequential(
            nn.Linear(n_features, n_embd),
            nn.LayerNorm(n_embd),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd, nhead=n_head,
                dim_feedforward=4 * n_embd,
                dropout=dropout,
                activation='gelu',
                batch_first=True, norm_first=True
            ) for _ in range(n_layer)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, 1)
        )

    def forward(self, x):
        x = self.feature_proj(x)
        x = x.unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        logits = self.head(x.squeeze(0))
        return logits.flatten()


class GPTModelWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, n_features=None, n_embd=64, n_head=4, n_layer=3,
                 block_size=30, lr=1e-3, epochs=50, device=device, feature_names=None):
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
        if self.model is not None:
            return torch.abs(self.model.feature_proj[0].weight).sum(dim=0).detach().cpu().numpy()
        return None

    def fit(self, X, y, groups=None):
        if groups is None:
            raise ValueError("GPTModelWrapper requires groups (race_id) for training.")

        if self.feature_names is None and hasattr(X, 'columns'):
            self.feature_names = list(X.columns)

        X_val = X.values if hasattr(X, 'values') else X
        y_val = y.values if hasattr(y, 'values') else y

        self.n_features = X_val.shape[1]
        self.model = PMUTransformer(
            self.n_features, self.n_embd, self.n_head, self.n_layer, self.block_size
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        group_indices = []
        start = 0
        for g_size in groups:
            group_indices.append(slice(start, start + g_size))
            start += g_size

        self.model.train()
        for epoch in range(self.epochs):
            random_indices = np.random.permutation(len(group_indices))
            for idx in random_indices:
                slc = group_indices[idx]
                X_race = torch.tensor(X_val[slc], dtype=torch.float32).to(self.device)
                y_race = torch.tensor(y_val[slc], dtype=torch.float32).to(self.device)
                if len(X_race) == 0: continue

                logits = self.model(X_race)
                winner_idx = torch.argmax(y_race)
                loss = F.cross_entropy(logits.unsqueeze(0), winner_idx.unsqueeze(0))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
        return self

    def predict(self, X, groups=None):
        if self.model is None: raise ValueError("Model not fitted.")
        self.model.eval()
        X_val = X.values if hasattr(X, 'values') else X
        X_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = []
            if groups is not None:
                start = 0
                for g_size in groups:
                    X_race = X_tensor[start:start + g_size]
                    if len(X_race) > 0:
                        logits = self.model(X_race)
                        outputs.extend(logits.cpu().numpy())
                    start += g_size
            else:
                for i in range(X_tensor.size(0)):
                    out = self.model(X_tensor[i].unsqueeze(0))
                    outputs.append(out.item())
            return np.array(outputs)

    def predict_proba(self, X, groups=None):
        scores = self.predict(X, groups=groups)
        probas = np.zeros(len(scores))
        if groups is not None:
            start = 0
            for g_size in groups:
                slc = slice(start, start + g_size)
                s = scores[slc]
                if len(s) > 0:
                    e = np.exp(s - s.max())
                    probas[slc] = e / e.sum()
                start += g_size
        else:
            e = np.exp(scores - scores.max())
            probas = e / e.sum()
        return np.vstack([1 - probas, probas]).T

    def __sklearn_is_fitted__(self):
        return self.model is not None