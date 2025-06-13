# ===== main.py =====
import argparse, os, json, datetime, torch, textwrap
from train import train_one_subject

# 논문 기준치 (수정 필요시 여기에서!)
PAPER_ACC = {1:0.8854,2:0.6910,3:0.9410,4:0.8229,5:0.7812,6:0.7083,7:0.9340,8:0.8854,9:0.8470}
PAPER_KAP = {1:0.777,2:0.690,3:0.941,4:0.823,5:0.781,6:0.708,7:0.934,8:0.886,9:0.847}

ap = argparse.ArgumentParser()
ap.add_argument("--subjects", nargs="+", type=int, default=sorted(PAPER_ACC))
ap.add_argument("--epochs",   type=int, default=500)
ap.add_argument("--batch",    type=int, default=16)
ap.add_argument("--data",     type=str,  default="./data/")
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results", exist_ok=True)

records = []

for sid in args.subjects:
    print(f"\n=== Subject {sid} ===")
    acc, kap = train_one_subject(
        sub=sid, data_root=args.data,
        epochs=args.epochs, batch_size=args.batch, device=device
    )
    records.append((sid, acc, kap))

# ── summary table ──────────────────────────────────────────
print("\n" + "-"*66)
print(f"{'SUB':>3} | {'ACC':>7} | {'κappa':>7} | {'ΔACC':>7} | {'Δκ':>7}")
print("-"*66)

rows, d_accs, d_kaps, accs, kaps = [], [], [], [], []
for sid, acc, kap in records:
    da = acc - PAPER_ACC.get(sid, float('nan'))
    dk = kap - PAPER_KAP.get(sid, float('nan'))
    accs.append(acc); kaps.append(kap); d_accs.append(da); d_kaps.append(dk)
    row = f"{sid:>3d} | {acc:7.3f} | {kap:7.3f} | {da:+7.3f} | {dk:+7.3f}"
    print(row); rows.append(row)

print("-"*66)
avg_acc = sum(accs)/len(accs); avg_kap = sum(kaps)/len(kaps)
avg_da  = sum(d_accs)/len(d_accs); avg_dk = sum(d_kaps)/len(d_kaps)
final = f"{'AVG':>3} | {avg_acc:7.3f} | {avg_kap:7.3f} | {avg_da:+7.3f} | {avg_dk:+7.3f}"
print(final); print("-"*66)

# ── save summary files ────────────────────────────────────
summary_txt = "\n".join(["-"*66,
                         f"{'SUB':>3} | {'ACC':>7} | {'κappa':>7} | {'ΔACC':>7} | {'Δκ':>7}",
                         "-"*66,
                         *rows,
                         "-"*66,
                         final,
                         "-"*66])

os.makedirs("results", exist_ok=True)
with open("results/summary.txt", "w", encoding="utf-8") as f:   # ★ UTF-8 지정
    f.write(summary_txt)

json.dump({
    "datetime" : str(datetime.datetime.now()),
    "records"  : [{"sub":s,"acc":a,"kap":k,
                   "paper_acc":PAPER_ACC.get(s),"paper_kap":PAPER_KAP.get(s)}
                  for s,a,k in records],
    "avg_acc"  : avg_acc,
    "avg_kap"  : avg_kap},
    open("results/result_summary.json","w"), indent=2)

print("Saved figure(s) → figs/  |  summary → results/")
