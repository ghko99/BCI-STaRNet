# Reproducibility Notes

Use these notes when comparing STaRNet runs across machines or experiment revisions.

## Record with each run

- Git commit SHA.
- Python and PyTorch versions.
- CUDA version and GPU model.
- Subject IDs, epoch count, batch size, and dataset path.
- Random seed if one is configured locally.

## Recommended flow

```bash
pip install -r requirements.txt
python main.py --subjects 1 --epochs 50 --batch 8 --data ./data/
python main.py --subjects 1 2 3 4 5 6 7 8 9 --epochs 500 --batch 16 --data ./data/
```

## Outputs

Archive generated `results/summary.txt`, `results/result_summary.json`, and validation plots together with the run configuration. Generated files should stay out of Git.
