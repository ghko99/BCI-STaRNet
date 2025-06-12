import torch
import torch.nn as nn

class BilinearMapping(nn.Module):
    """ X ∈ ℝ^{d×d}  →  W X Wᵀ ,   W ∈ Stiefel(out_dim, in_dim) """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        W = torch.empty(out_dim, in_dim)
        nn.init.orthogonal_(W)                 # semi-orthogonal
        self.W = nn.Parameter(W)

    def forward(self, X):
        return self.W @ X @ self.W.t()

    def orthogonalize(self):                   # 주기적 re-orth
        with torch.no_grad():
            q, _ = torch.linalg.qr(self.W.data.t())
            self.W.copy_(q[:, :self.W.size(0)].t())

class LogEig(nn.Module):
    def forward(self, X):
        lam, vec = torch.linalg.eigh(X)
        lam = torch.clamp(lam, min=1e-6).log()
        return vec @ torch.diag_embed(lam) @ vec.transpose(-1, -2)

class RiemannianFC(nn.Sequential):
    def __init__(self, mat_dim: int, num_classes: int):
        super().__init__(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(mat_dim * mat_dim, num_classes)
        )