import os, numpy as np, scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

_LABEL_MAP = {769:0, 770:1, 771:2, 772:3, 1:0, 2:1, 3:2, 4:3}

# ───────────────────────── MAT 파서 ─────────────────────────
def _load_mat(path: str):
    """Return X:(N,22,1750)  y:(N,)  ─ 예외 발생 시 FileNotFoundError"""
    mat  = sio.loadmat(path)               # 에러 시 상위에서 잡힘
    runs = mat["data"][0]                  # 6 runs
    segs, labels = [], []
    NWIN = 7 * 250                         # 1750 samples (4 s 구간 쓰시려면 4*250)
    for run in runs:
        r  = run[0, 0]
        X  = r["X"]                        # (samples, 25)
        beg_idx = r["trial"].squeeze()
        y      = r["y"].squeeze()
        for b, lbl in zip(beg_idx, y):
            frag = X[b : b + NWIN, :22]
            if frag.shape[0] == NWIN:      # artefact-free, 길이 OK
                segs.append(frag.T.astype(np.float32))
                labels.append(_LABEL_MAP[int(lbl)])
    return np.stack(segs), np.array(labels, np.int64)

# ───────────────────────── Dataset/Loader ─────────────────────────
class BCIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)       # CPU tensor
        self.y = torch.from_numpy(y)
    def __len__(self):  return len(self.X)
    def __getitem__(self, idx):            # (22,1750) → (22,1750)
        return self.X[idx], self.y[idx]

def get_loader(sub:int, batch:int, train:bool, root:str, device):
    fname   = f"A{sub:02d}{'T' if train else 'E'}.mat"
    X, y    = _load_mat(os.path.join(root, fname))
    loader  = DataLoader(
        BCIDataset(X, y),
        batch_size=batch,
        shuffle=train,
        pin_memory=(str(device) == "cuda"),
        num_workers=0,
    )
    return loader