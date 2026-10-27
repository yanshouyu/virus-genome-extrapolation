# Handoff after the October 20 validation-only work

Read [the completed report](experiments-2026-10-20/README.md), [fixed protocol](experiments-2026-10-20/protocol.md), [reproduction notes](experiments-2026-10-20/reproduction.md) and [external evaluation proposal](experiments-2026-10-20/independent-evaluation-plan.md). The morning discussion remains in handoff-2026-10-20.md; this records subsequent execution authorized by the user.

## Completed scope

Frozen NTv1/NTv2 CLS and masked-mean train/validation embeddings; standardized regularized logistic readouts at five predeclared C values; fixed-validation selection; five outer / three inner grouped nested CV on training only; k-mer under the same readout/CV procedure; sampling audit; sourced external-evaluation requirements and target-exclusion proposal; exports, replay checks and regression suite.

No new real clean_v2 test inference/scoring/selection; no adaptation; no broad sweep; no optional DeepSeek/provider access. Actual GPU extraction jobs took 39 seconds each including startup, CPU analysis 23 seconds. All backlog items finished well before ten hours; no cluster jobs or scheduled automations remain.

## Findings

- Fixed validation log loss: k-mer .449, NTv1 CLS .439, NTv1 mean .467, NTv2 CLS .655, NTv2 mean .524.
- Nested pooled outer log loss: k-mer .512, NTv1 CLS .535, NTv1 mean .600, NTv2 CLS .844, NTv2 mean .541.
- NTv2 mean beats CLS in all outer folds. NTv1's small fixed-validation lead over k-mer reverses in pooled nested CV. Do not call this a resolved superiority comparison.
- All selected C values are .01, the strongest regularization tested; no post-result grid expansion was made.
- Linear CLS does not improve historical nonlinear CLS validation losses. Differences include scaling/regularization/optimization, so do not assert head nonlinearity or width caused the earlier result.
- Sampling context is asymmetric: 0/111 train and 0/37 validation negatives overlap same-strand CDS; 65/122 and 25/40 positives do. Original negative sampling excludes same-strand CDS. This is a plausible shortcut, not proof the fitted models exploit it. Preserve clean_v2 and define a separate improved benchmark in a future protocol.
- Model generation and pretraining corpus remain confounded. Opposite-strand overlaps remain permitted by user policy.

## Verification and engineering

19 local tests passed, 2 large opt-in tests skipped. First-batch embedding replay difference zero; all five exported fixed-validation readouts replay locally within 3.34e-16. Source and artifact hashes, pinned model revisions, package versions and metric recomputation are recorded. All-tensor reference weight comparison was not repeated in this round; use October 19's verified loader evidence with that limitation.

Initial extraction failed from inconsistent tokenizer special masks; fixed masks derive from actual IDs. Initial analysis failed a strict replay check because float32 scaling differed from manual float64 scaling by 4.63e-8; corrected by promoting all embedding features to float64 before fitting. Both failed attempts are preserved separately and excluded. Successful extraction: 24697844_0/1. Successful analysis: 24697924. Cancelled dependency job: 24697824. Full accounting in report directory.

## Artifacts and next direction

Nibi experiment root: `/home/shouyu/projects/def-aucoin/shouyu/virus-genome-extrapolation/runs/2026-10-20/`. Local full cache/export copies: `runs/2026-10-20/collected/`. Compact report/probabilities/provenance: `documents/experiments-2026-10-20/`.

Next prioritize evaluation design, not adaptation: context-matched negatives and window rules; reference/annotation reconciliation for a second AcMNPV study; verify a compatible different-virus TSS dataset before claiming unseen-virus transfer. The 2018 long-read study has AcMNPV data under ENA PRJEB24943/PRJEB27818 and reference KM667940.1; it is an annotation-validation lead, not unseen-virus evidence. A BmNPV transcriptome lead is not yet certified as genome-wide experimental TSS truth. No external predictions/download campaign occurred.

Any future adaptation needs accession-preserving corpus construction, genome/group-level partitioning, predeclared related-genome exclusions and repaired token budgeting. The historical 44 target-genome windows preclude a target-excluded claim as-is. The proposed policy is documented, not implemented or a new adaptation authorization.

Preserve pre-existing local untracked scripts/run_training.slurm and all existing untracked Nibi files. Shared SSH connection worked during this session; verify it afresh next time. No secrets/provider configuration were read.
