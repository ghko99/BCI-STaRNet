# Dataset Setup Notes

STaRNet expects BCI Competition IV 2a `.mat` files under `./data/`.

## Expected layout

```text
data/
  A01T.mat
  A01E.mat
  A02T.mat
  A02E.mat
  ...
  A09T.mat
  A09E.mat
```

## Before training

- Confirm all subject train/evaluation files are present.
- Keep the raw dataset outside Git; the files are large and licensed separately.
- Use the same dataset split when comparing runs with the paper or with previous local results.

## Quick check

Run a short single-subject job before a full experiment:

```bash
python main.py --subjects 1 --epochs 5 --batch 8 --data ./data/
```
