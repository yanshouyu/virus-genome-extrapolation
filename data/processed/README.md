# Processed datasets

Use [clean_v2](clean_v2/README.md) for continued work. It provides separate train, validation and test files, a strand-aware grouping policy, label checks and a source/hash manifest.

The original `train.csv`, `test.csv`, `train.fasta` and `test.fasta` in this directory are local course artifacts. They remain unchanged for provenance. Their per-class MMseqs2 split contains same-strand genomic overlaps across train/test and three label-inconsistent source rows; do not use them for new model selection or final evaluation.

To reproduce the current export from the local interim inputs, run from the repository root:

```bash
PYTHONPATH=src python scripts/grouped_train_test_split.py
PYTHONPATH=src python scripts/audit_dataset.py
```

The clean export is tracked by Git. Large raw data, interim clustering artifacts and the historical course exports remain ignored. See [the repair report](../../documents/repair-2026-10-13/README.md).
