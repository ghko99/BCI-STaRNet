# ===== train.py =====
import os, json, datetime
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler           # GradScaler 그대로
from torch import amp                           # 새 API
from starnet     import STaRNet
from data_loader import get_loader
from sklearn.metrics import cohen_kappa_score

# ────────────────────────────────────────────────────────────────
def train_one_subject(sub, data_root, epochs=500, batch_size=16, device="cpu"):
    tr_loader = get_loader(sub, batch_size, True,  data_root, device)
    va_loader = get_loader(sub, batch_size, False, data_root, device)

    net  = STaRNet().to(device)
    opt  = optim.Adam(net.parameters(), lr=1e-3)
    sch  = optim.lr_scheduler.StepLR(opt, 150, 0.5)
    ce   = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device.type == "cuda"))

    va_curve, best_acc, best_kap = [], 0., 0.

    for ep in range(1, epochs + 1):
        # ─── train ────────────────────────────────────────────
        net.train()
        for x_cpu, y_cpu in tr_loader:
            x = x_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with amp.autocast(device_type=device.type,
                              enabled=(device.type == "cuda")):
                out  = net(x)
                loss = ce(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        net.orth(); sch.step()

        # ─── validation ───────────────────────────────────────
        net.eval();  preds, gts = [], []
        with torch.no_grad(), amp.autocast(device_type=device.type,
                                           enabled=(device.type == "cuda")):
            for x_cpu, y_cpu in va_loader:
                x = x_cpu.to(device, non_blocking=True)
                y = y_cpu.to(device, non_blocking=True)
                o = net(x).argmax(1)
                preds.extend(o.cpu()); gts.extend(y.cpu())

        acc = (torch.tensor(preds) == torch.tensor(gts)).float().mean().item()
        kap = cohen_kappa_score(gts, preds)
        va_curve.append(acc)
        if acc > best_acc: best_acc, best_kap = acc, kap

        if ep % 10 == 0 or ep == 1:
            print(f"Ep {ep:3d}  val_acc {acc:0.3f}  κ {kap:0.3f}")

    # ─── save val-curve ────────────────────────────────────────
    os.makedirs("figs", exist_ok=True)
    plt.figure(figsize=(6,3))
    plt.plot(range(1, epochs + 1), va_curve)
    plt.xlabel("Epoch"); plt.ylabel("Val Acc"); plt.title(f"Subject {sub}")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"figs/val_curve_sub{sub:02d}.png"); plt.close()

    # 개별 로그 JSON
    os.makedirs("results", exist_ok=True)
    json.dump({"acc_curve": va_curve,
               "best_acc" : best_acc,
               "best_kap" : best_kap,
               "datetime" : str(datetime.datetime.now())},
              open(f"results/log_sub{sub:02d}.json","w"), indent=2)
    return best_acc, best_kap
