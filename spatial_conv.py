import torch
import torch.nn as nn

class _SpatialBlock(nn.Sequential):
    """(ΔC × 1) 커널로 electrode 축만 따라 이동"""
    def __init__(self, out_ch: int, k_h: int):
        super().__init__(
            nn.Conv2d(1, out_ch, (k_h, 1), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU()
        )

class MultiScaleSpatialBlock(nn.Module):
    """
    ┌ branch-1 : 8 × 1 (local, 8 electrodes)
    ├ branch-2 : 16 × 1 (mid-range)
    └ branch-3 : C  × 1 (global)
        → h-축을 일렬로 펼쳐 concat → 1×1 fuse (32ch)
    """
    def __init__(self, channels: int, k: int = 2):
        super().__init__()
        self.br1 = _SpatialBlock( 8 * k,  8)
        self.br2 = _SpatialBlock( 8 * k, 16)
        self.br3 = _SpatialBlock(16 * k, channels)

        total_planes = (8 * k) * (channels - 8  + 1) + \
                       (8 * k) * (channels - 16 + 1) + 16 * k
        self.fuse = nn.Sequential(
            nn.Conv2d(total_planes, 32, (1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU()
        )

    def forward(self, x):                       # x : (B,1,C,T)
        bs, _, C, T = x.shape
        feats = []
        for br in (self.br1, self.br2, self.br3):
            f = br(x)                           # (B,*,H,T)
            f = f.reshape(bs, -1, 1, T)         # H 축을 channel 로 펼침
            feats.append(f)
        x_cat = torch.cat(feats, dim=1)         # (B,total,1,T)
        return self.fuse(x_cat)                 # (B,32,1,T)