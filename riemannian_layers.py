# ===== riemannian_layers.py =====
import torch, torch.nn as nn

class BilinearMapping(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        W = torch.empty(out_dim, in_dim)
        nn.init.orthogonal_(W)
        self.W = nn.Parameter(W)

    def forward(self, X):  # (B,d,d)
        return self.W @ X @ self.W.t()

    def orthogonalize(self):
        with torch.no_grad():
            q,_ = torch.linalg.qr(self.W.data.t())
            self.W.copy_(q[:,:self.W.size(0)].t())

class LogEig(nn.Module):
    def forward(self, X):
        dtype = X.dtype
        lam, vec = torch.linalg.eigh(X.float())
        lam = torch.clamp(lam,1e-6).log()
        return (vec @ torch.diag_embed(lam) @ vec.transpose(-1,-2)).to(dtype)

class RiemannianFC(nn.Sequential):
    def __init__(self, mat_dim, num_classes):
        super().__init__(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(mat_dim*mat_dim, num_classes)
        )
