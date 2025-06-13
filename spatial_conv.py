# ===== spatial_conv.py =====
import torch.nn as nn, torch

class _SpatialBlock(nn.Sequential):
    def __init__(self, out_ch: int, k_h: int):
        super().__init__(
            nn.Conv2d(1, out_ch, (k_h,1), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU()
        )

class MultiScaleSpatialBlock(nn.Module):
    def __init__(self, channels: int, k: int = 2):
        super().__init__()
        self.br1 = _SpatialBlock( 8*k,  8)
        self.br2 = _SpatialBlock( 8*k, 16)
        self.br3 = _SpatialBlock(16*k, channels)

        total = (8*k)*(channels-8+1) + (8*k)*(channels-16+1) + 16*k
        self.fuse = nn.Sequential(
            nn.Conv2d(total, 32, (1,1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU()
        )

    def forward(self, x):               # (B,1,C,T)
        bs, _, C, T = x.shape
        feats = []
        for br in (self.br1, self.br2, self.br3):
            f = br(x).reshape(bs, -1, 1, T)
            feats.append(f)
        return self.fuse(torch.cat(feats, dim=1)) # (B,32,1,T)
