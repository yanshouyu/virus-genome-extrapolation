# Validation-only representation/readout protocol — October 20, 2026

Authorized by the user's instruction to proceed after reviewing the proposed ten-hour plan. No test inference, test selection, adaptation, or new model families in this round.

Question: how do frozen representation pooling and regularized linear readouts affect discrimination on clean_v2? NTv1 human-reference and NTv2 multispecies differ in both generation and corpus; these are not matched architecture controls.

Keep clean_v2 training (233) and validation (77) unchanged. Read only these two CSVs in the new experiment. Extract pinned pretrained encoders in eval/inference mode. Cache final-layer raw CLS and mean over attended non-special tokens. Record row identifiers, labels, groups, source/data hashes and package versions. Do not truncate silently. Exclude empty mean masks. Run an extraction replay check on a small batch.

Candidates: NTv1 CLS, NTv1 mean, NTv2 CLS, NTv2 mean, and existing oriented 1–4-mer features. For each, use StandardScaler fitted only on fitting rows and L2 logistic regression with C in [0.01, 0.1, 1, 10, 100], max_iter=5000. Choose minimum validation log loss, ties smaller C. No refit on validation. Fixed threshold probability >0.5. Report log loss, MCC, accuracy, ROC AUC, AP, precision and recall, and all candidate scores. Export fitted scaler/readout and replay probabilities. Historical nonlinear-head validation losses are contextual, not a matched search-budget comparison.

Stability: five outer StratifiedGroupKFold folds on training only (shuffle, seed 42); select C by pooled out-of-fold log loss in three inner grouped folds (shuffle, seed 43). Fit scaling separately within every fitting fold. Record fold membership, class/group counts, selected C, fold metrics and pooled outer predictions. No validation or test in nested selection. This estimates the declared regularization-selection procedure; it does not provide independent biological replication. Treat rankings as exploratory, especially with five correlated training folds.

Minimum: verified caches and all validation comparisons. Follow-on backlog: nested grouped validation; sampling audit; independent evaluation requirements and accession/relative exclusion policy; report and reproduction notes. Complete these if minimum finishes early. Do not expand the candidate grid to fill time. GPU work uses SLURM H100 MIG slices, 16 GB RAM, four CPUs, one-hour bound, at most two extraction jobs. CPU analysis also uses SLURM. Preserve existing files. Record unfinished work and actual runtime.
