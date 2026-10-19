# First clean viral baseline experiment

Protocol fixed on 2026-10-19 before inspecting experiment performance.

## Question and scope

Do frozen NTv1 or NTv2 CLS representations improve TSS discrimination over simple sequence-composition baselines on the repaired AcMNPV split? This evaluates within-genome held-out groups, not transfer to an unseen virus or a human-versus-virus comparison.

Use clean_v2 unchanged: 233 train, 77 validation, 78 test. Keep the established strand policy and all existing groups. Do not rerun the split or tune to test performance.

## Comparisons

- Constant training-prevalence predictor.
- Logistic regression on normalized oriented DNA k-mer frequencies for k=1..4. Standardize using training statistics only. Select C from [0.01, 0.1, 1, 10, 100] by validation log loss; ties use the smaller C. Fit on training only, without adding validation after selection.
- Frozen NTv1 500M human-reference and NTv2 500M multispecies encoders, pinned to the revisions already recorded in the repaired loader. Use raw first-token representation and the existing 512-unit GELU head. Run seeds 42, 43, 44, keeping identical data splits. Learning rate 0.0003, 50 epochs, batch size 50; existing evaluation/checkpoint intervals are retained. Select each run's saved checkpoint by validation loss. No backbone adaptation or hyperparameter search in this round.

Report log loss, MCC, accuracy, precision, recall, ROC AUC and average precision where prediction arrays permit. Use the existing argmax/0.5 decision threshold; no test-derived threshold. Show all three seeds, not only the best test run. Seed variability measures optimization variability, not independent biological replication.

For the deterministic baselines, choose settings on validation and then predict test once. For the NT runs, the existing Trainer performs one final test prediction after validation checkpoint selection. The engineering smoke job also invokes that existing path; its test metrics are excluded from model selection and scientific comparisons. Smoke results must not be used to adjust the experiment based on test performance.

## Execution and limits

Start with one short H100 1g.10gb GPU smoke job (6 epochs, seed 42) to validate CUDA, model loading, training, saving and checkpoint recovery. Submit real experiments only after engineering checks pass. Diagnose environmental failures without changing scientific settings based on performance. Request only small MIG GPU slices; no full-size GPU sweep. Bound each experimental job to one hour initially and the overall work window to approximately ten hours, ending around 2026-10-20 06:56 UTC (02:56 Toronto). Adjust resource limits based on observed runtime/memory, recording any change. Training goes through SLURM, never the login node.

Preserve existing untracked server files. Save configurations, package/Git/dataset metadata, prediction arrays and job logs. Commit code, protocol and compact evidence; leave large models under ignored runs/. Record queued/unfinished work honestly if the window ends first.

## Interpretation

A single small test split cannot establish broad generalization. Same-strand genomic separation and ORF grouping reduce the previously observed leakage; opposite-strand overlaps remain by explicit policy. Pretraining overlap and distant homology are not ruled out. A winning frozen probe would justify a subsequent matched-data transfer or carefully scoped adaptation experiment, rather than an immediate larger-model sweep.
