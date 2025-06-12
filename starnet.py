import torch, torch.nn as nn
from spatial_conv   import MultiScaleSpatialBlock
from temporal_conv  import MultiScaleTemporalBlock
from riemannian_layers import BilinearMapping, LogEig, RiemannianFC

def _cov(x, eps: float = 1e-4):
    x = x - x.mean(dim=-1, keepdim=True)
    c = x @ x.transpose(-1, -2) / (x.size(-1) - 1)
    I = torch.eye(c.size(-1), device=c.device)
    return c + eps * I

class STaRNet(nn.Module):
    def __init__(self, channels: int = 22, num_classes: int = 4,
                 bilinear_dim: int = 16):
        super().__init__()
        self.spatial  = MultiScaleSpatialBlock(channels)
        self.temporal = MultiScaleTemporalBlock(32)
        feat_dim      = self.temporal.out_dim
        self.bilinear = BilinearMapping(feat_dim, bilinear_dim)
        self.logeig   = LogEig()
        self.classify = RiemannianFC(bilinear_dim, num_classes)

    # ─────────────────────────────────────────────────────────────
    def forward(self, x):                 # x : (B,C,T)
        x = x.unsqueeze(1)                # (B,1,C,T)
        s = self.spatial(x)               # (B,32,1,T)
        t = self.temporal(s)              # (B,1024,L)
        C = _cov(t)                       # (B,1024,1024)
        Y = self.bilinear(C)              # (B,16,16)
        Z = self.logeig(Y)                # (B,16,16)
        return self.classify(Z)

    def orth(self):                       # helper
        self.bilinear.orthogonalize()