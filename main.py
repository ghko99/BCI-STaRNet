import argparse, torch
from train import train_one_subject

parser = argparse.ArgumentParser()
parser.add_argument("--subjects", nargs="+", type=int, default=[1])
parser.add_argument("--epochs",  type=int, default=500)
parser.add_argument("--batch",   type=int, default=16)
parser.add_argument("--data",    type=str, default="./data/")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for sid in args.subjects:
    print(f"\n=== Subject {sid} ===")
    train_one_subject(
        sub        = sid,
        data_root  = args.data,
        epochs     = args.epochs,
        batch_size = args.batch,
        device     = device,
    )