# Training Checklist

Use this checklist before launching STaRNet experiments.

## Environment

- Confirm the Python environment and CUDA version.
- Record GPU model, driver version, and available memory.
- Confirm all package versions used for PyTorch training.
- Save the Git commit and branch with each run.

## Data

- Verify the BCI IV 2a files are present in the expected layout.
- Check subject ids and session ids before training.
- Keep raw EEG files outside normal Git history.
- Record any preprocessing or channel selection changes.

## Run Record

Save the command line, random seed, model configuration, checkpoint path, and final metric output together. This keeps subject-level comparisons reproducible.
