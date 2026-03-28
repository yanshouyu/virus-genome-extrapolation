# Processed Datasets

This directory contains the processed training and testing datasets for the virus genome extrapolation project.

## Files

- `train.fasta`: Combined FASTA file containing training sequences from both positive and negative TSS classes. Sequences are grouped by MMseqs2 clusters to prevent within-cluster leakage during training.
- `train.csv`: Same sequences as in `train.fasta`, for easy loading by `datasets.load_dataset()`.
- `test.fasta`: Combined FASTA file containing testing sequences from both positive and negative TSS classes, similarly grouped.
- `test.csv`: Same sequences as in `test.fasta`, for easy loading by `datasets.load_dataset()`.

## Generation

These files were created using the `grouped_train_test_split.py` script, which performs a grouped train-test split (20% test size) on the interim TSS sequence data (`data/interim/tss_seq_range.fasta` and `data/interim/tss_neg_seq_range.fasta`) using cluster information from MMseqs2 clustering results.

The split ensures that sequences from the same cluster (based on sequence similarity) are not split across train and test sets, maintaining data integrity for model evaluation.