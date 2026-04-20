"""
PMU Transformer Vectorized Version
Optimisé pour le calcul matriciel sur GPU/NPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Configuration du device (Support CUDA/NPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    import intel_npu_acceleration_library
    device = torch.device("npu")
except ImportError:
    print("No NPU support available.")
    pass

print(f"Utilisation du device: {device}")

# 1. Simulation Dataset PMU (Vectorisé)
def get_pmu_batch_torch(n_features, device):
    num_horses = random.randint(8, 18)
    # Features normalisées (Tensor)
    race_features = torch.randn(num_horses, n_features).to(device)
    winner_idx = torch.tensor([random.randint(0, num_horses - 1)]).to(device)
    return race_features, winner_idx

# 2. Architecture PMU Transformer
class PMUTransformer(nn.Module):
    def __init__(self, n_features, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.n_embd = n_embd
        self.block_size = block_size
        
        # Entrée : Projection des features LTR/TabNet
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
                norm_first=True # Proche de RMSNorm
            ) for _ in range(n_layer)
        ])
        
        # Tête : Calcul de la force (logit) de chaque cheval
        self.head = nn.Linear(n_embd, 1)

    def forward(self, x):
        # x shape: (num_horses, n_features)
        num_horses = x.size(0)
        
        # 1. Projection et Position
        x = self.feature_proj(x) # (num_horses, n_embd)
        positions = torch.arange(num_horses, device=x.device)
        x = x + self.pos_emb(positions)
        
        # 2. Transformer Blocks (Attention sur toute la course)
        # On ajoute une dimension batch de 1 pour satisfaire l'API
        x = x.unsqueeze(0) 
        for block in self.blocks:
            x = block(x)
        
        # 3. Logits (Force relative)
        logits = self.head(x.squeeze(0)) # (num_horses, 1)
        return logits.flatten()

# 3. Hyperparamètres
n_features = 45
n_embd = 64 # Augmenté car on est sur GPU
n_head = 4
n_layer = 3
block_size = 22 # Max partants PMU

model = PMUTransformer(n_features, n_embd, n_head, n_layer, block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print(f"Modèle initialisé avec {sum(p.numel() for p in model.parameters())} paramètres.")

# 4. Boucle d'entraînement (Vectorisée)
num_steps = 1000
model.train()

for step in range(num_steps):
    race_features, winner_idx = get_pmu_batch_torch(n_features, device)
    
    # Forward
    logits = model(race_features)
    
    # Perte : CrossEntropy sur la course (LogLoss)
    # F.cross_entropy attend (input_logits, target_index)
    loss = F.cross_entropy(logits.unsqueeze(0), winner_idx)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(f"Step {step:4d} | LogLoss: {loss.item():.4f}")

# 5. Inférence Vectorisée
@torch.no_grad()
def predict_probs_vectorized(model, race_features):
    model.eval()
    logits = model(race_features)
    probs = F.softmax(logits, dim=0)
    return probs

print("\n--- Test Inférence Vectorisée ---")
test_race, _ = get_pmu_batch_torch(n_features, device)
probs = predict_probs_vectorized(model, test_race)

for i, p in enumerate(probs):
    print(f"Cheval {i+1:2d} | Probabilité: {p.item():.4f}")

print(f"\nSomme des probabilités: {probs.sum().item():.2f} (Vérification Softmax)")
