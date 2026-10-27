# Clean AcMNPV dataset, version 2

This is the strand-aware replacement for the course split. The original files one directory above remain unchanged.

| Split | Positive | Negative | Total | Groups |
|---|---:|---:|---:|---:|
| Train | 122 | 111 | 233 | 191 |
| Validation | 40 | 37 | 77 | 64 |
| Test | 41 | 37 | 78 | 64 |
| Total | 203 | 185 | 388 | 319 |

All sequences are 300 bases from `NC_001623.1`, oriented by strand. CSV columns are `sequence`, `name`, `label`, `group`; `name` encodes a zero-based TSS or negative identifier, a zero-based half-open genomic interval, strand and label. Group IDs are metadata and must not be model inputs.

Same-strand overlapping windows, positives associated with the same ORF, same-strand exact/reverse-complement sequence duplicates and same-strand members of known clusters stay in one split. Grouping is transitive and applied jointly across labels. Opposite-strand overlaps are allowed by the project owner's policy. The audit found 97 such allowed interval pairs across the three splits.

Three source rows were excluded: two positive windows that contain neither their named TSS nor any other recorded same-strand TSS, and one negative window containing a recorded same-strand TSS. Identifiers and reasons are in `manifest.json`. Retained sequences and coordinates were not changed. Labels were checked against all starts recorded in the curated raw source table, including starts omitted from the positive-window selection. A negative label means no recorded same-strand TSS in the interval, not experimentally confirmed absence of transcription.

The split targets are 60/20/20. Group sizes make exact fractions impossible in general. The generator chooses among 512 seeded candidate partitions using per-class sample counts only, without model metrics. It assigns test first, then validation from the remaining groups. The exported files are reproducible byte for byte.

From the repository root in WSL:

```bash
conda activate ece657d
PYTHONPATH=src python scripts/grouped_train_test_split.py
PYTHONPATH=src python scripts/audit_dataset.py
```

The generator requires the local interim FASTA/cluster files; the exported dataset is tracked so training does not require those generated inputs. The manifest records their hashes. The loader fails on missing splits, duplicate identifiers, forbidden cross-split overlaps/groups/ORFs, and inconsistent TSS labels.

Use validation for development and test for final reporting. These are new partitions of previously inspected course windows, not a new biological cohort. This dataset establishes the documented strand-aware split guarantees; it does not establish absence of arbitrary remote homology or overlap with foundation-model pretraining data.

See [the repair report](../../../documents/repair-2026-10-13/README.md) for causes, verification and model changes.
