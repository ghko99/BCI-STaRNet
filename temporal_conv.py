import torch
import torch.nn as nn

class _TempBlock(nn.Sequential):
    def __init__(self, out_ch: int, k_w: int):
        pad = k_w // 2                          # 홀·짝 모두 동일 출력 길이
        super().__init__(
            nn.Conv2d(1, out_ch, (1, k_w), padding=(0, pad), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU()
        )

class MultiScaleTemporalBlock(nn.Module):
    """
    ┌ branch-25 ── 8ch
    ├ branch-50 ── 8ch
    └ branch-100 ─ 16ch
        (논문 Fig-3 & Ablation)
    출력: (B, (8+8+16)*32 = 1024, L)
    """
    def __init__(self, in_planes: int, dropout: float = 0.25):
        super().__init__()
        self.br25 = _TempBlock( 8,  25)
        self.br50 = _TempBlock( 8,  50)
        self.br100= _TempBlock(16, 100)
        self.do = nn.Dropout2d(dropout)
        self.out_dim = (8 + 8 + 16) * in_planes   # 1024

    def forward(self, x):                        # x : (B,32,1,T)
        bs, c, _, T = x.shape
        x = x.view(bs, 1, c, T)                 # 시간축에만 kernel
        y25, y50, y100 = self.br25(x), self.br50(x), self.br100(x)
        L = min(y25.size(-1), y50.size(-1), y100.size(-1))
        feats = [y[..., :L].reshape(bs, -1, L) for y in (y25, y50, y100)]
        return self.do(torch.cat(feats, dim=1))  # (B,1024,L)