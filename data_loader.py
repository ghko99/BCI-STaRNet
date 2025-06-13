# ===== data_loader.py =====
import os, numpy as np, scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

_LABEL_MAP = {769:0, 770:1, 771:2, 772:3, 1:0, 2:1, 3:2, 4:3}

# ──────────────────────────── MAT reader ────────────────────────────
def _load_mat(path: str):
    """Return X:(N,22,1750)  y:(N,) after artefact removal."""
    mat  = sio.loadmat(path)               # may raise FileNotFoundError
    runs = mat["data"][0]                 # 6 runs
    segs, labels = [], []
    WIN = 7 * 250                         # 1750 samples  (4 s 구간만 쓰셔도 OK)

    for run in runs:
        r = run[0, 0]
        X_full  = r["X"]                  # (samples,25)
        starts  = r["trial"].squeeze()
        y_vec   = r["y"].squeeze()

        for s_idx, lbl in zip(starts, y_vec):
            frag = X_full[s_idx : s_idx + WIN, :22]
            if frag.shape[0] == WIN:      # keep only full-length, artefact-free
                segs.append(frag.T.astype(np.float32))
                labels.append(_LABEL_MAP[int(lbl)])

    return np.stack(segs), np.array(labels, np.int64)

# ─────────────────────────────── windows ───────────────────────────────
def make_windows(X, y, fs=250, win=2.0, stride=0.5):
    """2-s windows, 0.5-s stride – used **only** for training augmentation."""
    L   = int(win * fs)
    stp = int(stride * fs)
    segs, labs = [], []
    for xi, yi in zip(X, y):
        for s in range(0, xi.shape[-1] - L + 1, stp):
            segs.append(xi[:, s:s+L])
            labs.append(yi)
    return np.stack(segs), np.array(labs, np.int64)

# ───────────────────────── Dataset / Loader ─────────────────────────
class BCIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)      # (N,C,T)
        self.y = torch.from_numpy(y)
    def __len__(self):  return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_loader(sub:int, batch:int, train:bool, root:str, device):
    fname  = f"A{sub:02d}{'T' if train else 'E'}.mat"
    X, y   = _load_mat(os.path.join(root, fname))
    if train:
        X, y = make_windows(X, y)         # augmentation
    return DataLoader(
        BCIDataset(X, y),
        batch_size=batch,
        shuffle=train,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )
