# Completed handoff — October 23, 2026

The authorized two-hour window began October 24 at 01:43:37 UTC (October 23, 21:43:37 Toronto); deadline 03:43:37 UTC. All experiments, analysis and verification finished by 02:50 UTC; final documentation and repository synchronization followed within the window. This supersedes handoff-adaptation-2026-10-23-progress.md.

Read [the complete report](adaptation-2026-10-23/README.md) and [reproduction instructions](adaptation-2026-10-23/reproduction.md).

## Completed scope

All 12 valid NTv2 scientific runs completed: 100M/500M multispecies, baculovirus/general control, seeds 42/43/44, 512 steps per run. The 100M jobs used 20GB H100 MIG; 500M used full H100 for scheduling availability. A frozen NTv1 2.5B multispecies reference also completed. No 2.5B adaptation was attempted. At most two GPU jobs ran concurrently. Source fidelity, selected-state reload and fresh-process embedding replay all passed. All 16 readouts replay locally within 4.45e-16; 25 regression tests passed and 2 opt-in large-model tests were skipped.

Engineering corrections preceded adapted downstream outcome inspection: use no_grad for evaluation in a trainable NTv2 process; repair general sampling to select loci then both strands. All original general runs were excluded and preserved, and controls were rerun from the pinned pretrained states. Full job accounting and exclusions are in adaptation-2026-10-23/window-record.md and jobs.psv.

## Scientific result

Validation MCC: 100M unadapted .584, general .550, baculovirus .627; 500M unadapted .506, general .575, baculovirus .575. Values for adapted conditions are means across three adaptation seeds. K-mer MCC is .565 and frozen 2.5B MCC .513. K-mers retain the best log loss (.449) and ROC AUC (.882).

All primary paired group-bootstrap 95% intervals include zero. The study does not establish that baculovirus-specific adaptation is necessary. Modest point gains justify investigation, but the near-identical 500M general-control MCC gain prevents attribution to baculovirus exposure alone. This is validation on 77 windows/64 groups of one AcMNPV genome, not external biological replication.

## Data and reproducibility

Downstream clean_v2 remains unchanged (233 train / 77 validation / 78 test). Real test files were hashed for identity only: no real test predictions were generated. Pooling and C were frozen from training-only grouped selection before adapted outcomes: masked mean and C=.01 for both sizes; k-mer C=.01. No further tuning is justified from this report.

Nibi root: /home/shouyu/projects/def-aucoin/shouyu/virus-genome-extrapolation. Full selected checkpoints: runs/2026-10-21/scientific/. Compact local copies: runs/2026-10-23/collected/. Final CPU development outputs: remote runs/2026-10-21/development/, local runs/2026-10-23/development/. Corpus: runs/2026-10-21/corpus/. Valid paired general SHA256: 6455320fd95ae86d27d32c80c101b18c9529cc064f252e7972181a947d852e04.

Valid baculovirus arrays: 24896408 and 24896596, tasks 0/2/4. Corrected general arrays: 24897074 and 24897075, tasks 1/3/5. Fresh replays: 24897149/24897151. Full development: 24897153. Frozen 2.5B extraction/readout: 24897906/24897926. All finished. No scientific jobs remain queued or running and no monitor is installed.

## Next decision

Prioritize a larger independent annotated evaluation to resolve whether these effects generalize. Alternatively, predeclare a distinct adaptation-budget hypothesis before further training. Keep the real test separate from model selection. Do not describe this validation-only result as proof of either necessity or ineffectiveness, or use the 2.5B comparison as a causal model-size experiment.

Changes are saved on codex/leakage-and-model-fixes and synchronized to the same branch on Nibi. Prior source commit was 65c888d; repair commit was 02820c7; the final results commit contains this handoff. Preserve unrelated scripts/run_training.slurm and remote untracked logs/scripts.
