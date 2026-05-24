# Artifact Policy

STaRNet training can create large intermediate files. Keep repository history focused on source code, configuration, and curated documentation.

## Keep Out Of Git

- Raw BCI dataset files.
- Generated preprocessing caches.
- Model checkpoints and tensorboard logs.
- Per-run prediction dumps.
- Temporary notebooks or local analysis exports.

## Keep With Reports

For a reported experiment, preserve the training command, dataset revision, checkpoint identifier, subject split, and metric table in the same external run folder.

## Documentation

When adding result notes, include subject ids, preprocessing settings, random seed, and metric calculation script so the result can be traced later.
