# NTv2 500M multispecies retention study - in progress

The approved study evaluates the original model and six existing 4096-budget full-adaptation states on all18 tasks in the authors' revised NT benchmark. No new backbone training and no AcMNPV test inference. [Frozen protocol](protocol.md), [execution and jobs](execution.md), and [dataset audit](dataset-audit.json) provide details.

Data audit passed: all18 official train/test splits are chromosome-disjoint, with no cross-split coordinate overlap or exact/reverse-complement sequences. Every original labelled example is preserved. Exact sequence reuse reduces inference to434501training and32388test sequences per state.

The training-only smoke passed for original and baculovirus-adapted500M, with exact repeats and checkpoint reload. Existing states remain fixed; layer/C selection uses only original-model training examples. All benchmark test inference is gated on completing/fixing those18 selections.

GPU extraction is running on Nibi; dependent selection, benchmark-test extraction,126 readouts, fresh-process replay and checked reporting are submitted. Do not interpret job submission as completed results. The final results/COMPLETE marker is written only after all checks pass. No recurring monitor or automation is installed.
