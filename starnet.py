# ===== starnet.py =====
import torch, torch.nn as nn
from spatial_conv      import MultiScaleSpatialBlock
from temporal_conv     import MultiScaleTemporalBlock
from riemannian_layers import BilinearMapping, LogEig, RiemannianFC

@torch.jit.script
def _cov(x: torch.Tensor, eps: float = 1e-4):
    x = x - x.mean(dim=-1, keepdim=True)
    c = x @ x.transpose(-1,-2) / (x.size(-1)-1)
    return c + eps*torch.eye(c.size(-1), device=x.device, dtype=x.dtype)

class STaRNet(nn.Module):
    def __init__(self, channels=22, num_classes=4):
        super().__init__()
        self.spatial   = MultiScaleSpatialBlock(channels)
        self.temporal  = MultiScaleTemporalBlock(32)
        self.ln        = nn.LayerNorm(1024)          # channel LN
        self.bilinear  = BilinearMapping(1024, 32)
        self.logeig    = LogEig()
        self.cls       = RiemannianFC(32, num_classes)

    # ------- single window -------
    def _forward_once(self, x):            # x:(B,22,L)
        x  = self.spatial(x.unsqueeze(1))  # (B,32,1,L)
        t  = self.temporal(x)              # (B,1024,L)
        t  = self.ln(t.transpose(1,2)).transpose(1,2)
        z  = self.logeig(self.bilinear(_cov(t)))
        return self.cls(z)

    # ------- train / eval -------
    def forward(self, x):
        if x.dim() == 3:                   # (B,22,L) – train
            return self._forward_once(x)

        # eval: (B,K,22,L) or (B,22,L,K)
        if x.dim() != 4: raise RuntimeError("Input must be 3- or 4-D")
        if x.shape[1] == 22:               # (B,22,L,K) → (B,K,22,L)
            x = x.permute(0,3,1,2).contiguous()
        B,K,C,L = x.shape
        CHUNK   = 512
        outs = []
        flat = x.view(B*K,C,L)
        for st in range(0, B*K, CHUNK):
            outs.append(self._forward_once(flat[st:st+CHUNK]))
        out = torch.cat(outs).view(B,K,-1).softmax(-1).mean(1)
        return out

    def orth(self): self.bilinear.orthogonalize()
