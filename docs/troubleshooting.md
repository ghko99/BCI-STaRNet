# Troubleshooting Notes

Use these checks when STaRNet runs fail or produce unexpected metrics.

## Data Issues

- Confirm the subject id exists in the local BCI IV 2a dataset path.
- Check that train and evaluation sessions are not swapped.
- Verify preprocessing caches were regenerated after changing channel or window settings.

## Training Issues

- If loss is unstable, lower the learning rate and confirm batch shapes.
- If CUDA memory fails, reduce batch size or disable expensive logging.
- If results vary widely, record seed, subject id, and deterministic settings before comparing runs.

## Result Issues

For suspicious metrics, rerun a small smoke test and inspect per-subject outputs before promoting aggregate tables.
