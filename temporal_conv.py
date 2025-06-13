# ===== temporal_conv.py =====
import torch.nn as nn
import torch

class _TempBlock(nn.Sequential):
    def __init__(self, out_ch: int, k_w: int):
        pad = k_w // 2
        super().__init__(
            nn.utils.weight_norm(
                nn.Conv2d(1, out_ch, (1, k_w), padding=(0, pad), bias=False)
            ),
            nn.BatchNorm2d(out_ch),
            nn.ELU(),
        )

class MultiScaleTemporalBlock(nn.Module):
    """
    Branch kernels: 25 / 50 / 75 samples.
    Output dim = (8+8+16)*in_planes = 1024.
    """
    def __init__(self, in_planes: int, dropout: float = 0.25):
        super().__init__()
        self.br25 = _TempBlock(8, 25)
        self.br50 = _TempBlock(8, 50)
        self.br75 = _TempBlock(16, 75)
        self.do   = nn.Dropout1d(dropout)          # ← Dropout2d → Dropout1d
        self.out_dim = (8 + 8 + 16) * in_planes    # 1024

    def forward(self, x):            # x: (B,32,1,T)
        bs, c, _, T = x.shape
        x = x.view(bs, 1, c, T)
        y25, y50, y75 = self.br25(x), self.br50(x), self.br75(x)
        L = min(y25.size(-1), y50.size(-1), y75.size(-1))
        y_cat = torch.cat([y25[..., :L], y50[..., :L], y75[..., :L]], dim=1)
        return self.do(y_cat.reshape(bs, -1, L))   # (B,1024,L)
