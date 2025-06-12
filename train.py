import torch, torch.nn as nn, torch.optim as optim
from starnet import STaRNet
from data_loader import get_loader        # ← 이름 오류 해결

# ────────────────────────────────────────────
def train_one_subject(sub, data_root, epochs=500, batch_size=16, device="cpu"):
    tr_loader = get_loader(sub, batch_size, True,  data_root, device)
    va_loader = get_loader(sub, batch_size, False, data_root, device)

    net  = STaRNet().to(device)
    opt  = optim.Adam(net.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    best = 0.0

    for ep in range(1, epochs+1):
        # --- Train ---
        net.train(); corr = tot = 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = net(x)
            loss = crit(out, y)
            loss.backward(); opt.step()
            corr += (out.argmax(1) == y).sum().item();  tot += y.size(0)
        tr_acc = corr / tot

        # --- Val ---
        net.eval(); corr = tot = 0
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(device), y.to(device)
                out  = net(x)
                corr += (out.argmax(1) == y).sum().item();  tot += y.size(0)
        va_acc = corr / tot
        best   = max(best, va_acc)

        if ep % 10 == 0 or ep == 1:
            print(f"Ep{ep:3d}  tr_acc {tr_acc:0.3f}  va_acc {va_acc:0.3f}")
    print(f"Subject {sub}: best val acc = {best:0.3f}")