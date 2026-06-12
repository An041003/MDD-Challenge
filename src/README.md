# Source Modules

This folder contains reusable pieces of the final C-MED pipeline.

The Kaggle notebook remains self-contained so it can run after upload without
extra packaging steps. These modules mirror the core logic in a cleaner project
layout for code review, reuse, and report writing.

## Layout

- `mdd_cmed/config.py`: experiment and model defaults.
- `mdd_cmed/paths.py`: Kaggle path contract and audio path resolution.
- `mdd_cmed/phonemes.py`: phoneme tokenization and vocabulary utilities.
- `mdd_cmed/alignment.py`: Levenshtein alignment and V1 KEEP/SUB labels.
- `mdd_cmed/model.py`: C-MED V1 model with detection, operation, replacement, and utterance heads.
- `mdd_cmed/losses.py`: focal loss, token CE, and combined C-MED loss.
- `mdd_cmed/metrics.py`: internal MDD metrics and token-supervised diagnostics.
- `mdd_cmed/submission.py`: `results.csv` validation and `predict.zip` packaging.
