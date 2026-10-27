# Handoff: multispecies adaptation prepared October 21

The user authorized proceeding with paired NTv2 multispecies 100M and 500M adaptation and advised using larger Nibi MIG slices where needed. Primary goal: estimate whether baculovirus context adaptation is useful beyond broad-genome pretrained models, while preserving the current downstream data strategy.

Read documents/adaptation-2026-10-21/README.md and protocol.md first. **No new scientific performance results yet. Nibi reports every CPU/GPU partition down.** SSH works; both models are cached and the audited corpora/scripts are on Nibi.

## Ready and pending

- GPU engineering jobs 24757107 (100M, 20GB MIG) and 24757108 (500M, 40GB MIG), each 5 optimizer steps, 4 CPUs/32GB host/15min, are queued with PartitionDown. They may execute when partitions return. They write runs/2026-10-21/smoke-100m and smoke-500m.
- No scientific arrays, downstream evaluation jobs or scheduled chat automations have been submitted. The queued jobs are engineering only. Do not claim that twelve adaptation runs are already underway.
- Cache-preparation CPU job 24757086 was cancelled because of PartitionDown. Model-file downloading only was completed on the login host; no model loading, inference or training was performed there.
- Local regression suite: 23 passed, 2 large opt-in checks skipped. Four new checks cover masks/positions, actual tiny NTv2 backward/reload, and a complete synthetic paired evaluation without test.csv. Full-size GPU behavior remains unverified until smoke completes.

## Fixed scientific plan

Target-excluded baculovirus training versus equal-budget continued pretraining on a three-organism general-genome subset, and unadapted starting checkpoint. Sizes 100M and 500M; adaptation seeds 42/43/44 for both conditions. Full-parameter AdamW, lr1e-5, 512 steps, effective batch32 (micro4/accum8), 1536-base ACGT windows, BF16 autocast/FP32 parameters and optimizer. Same fixed held-out baculovirus MLM masks for both conditions; selected trained checkpoint at steps128/256/384/512 by MLM loss, initial loss reported separately. No TSS-performance-based adaptation checkpoint selection.

Downstream pooling and regularization are chosen using training-only grouped CV on unadapted embeddings at each size and held fixed across before/after arms. Development evaluates 77 validation windows. Final secondary test only after complete development freeze and checks; it is the previously inspected cohort, not an independent external test. No adaptation seed selection. Full protocol is authoritative; do not silently expand budget, corpus or model panel.

## Corpus provenance

97 source GenBank files; excluded AcMNPV NC_001623.1. Remaining records form 91 species/duplicate groups; 9 held out. 14,164 train and 1,448 held-out windows with both orientations; fixed MLM eval 128. General control has 14,164 windows from Saccharomyces cerevisiae, Caenorhabditis elegans and Drosophila melanogaster versioned assemblies present in the original NT source list. These are not a reconstruction of all 850 pretraining genomes. No close-species exclusion beyond target identities/duplicates; do not claim unseen-clade transfer.

All corpus hashes transferred match. Full ignored data: runs/2026-10-21/corpus/ locally and on Nibi. Pinned 100M revision f34324c6fde36a4f635f0f1f06cac5d25acd6798; 500M revision 06615c1660c892fc199840c18123f8385b3542a8. Revisions and manifest/source hashes are committed in report directory.

## Resume procedure

Inspect squeue/sacct and smoke logs. If completed, read both metadata.json files and verify actual peak memory/reference fidelity/parameter change/reload equality. Check timing to choose scientific walltime. Then activate the prepared environment and run scripts/submit_controlled_adaptation.sh. The launcher gates on smoke completion, reference fidelity, exact reload and reserved GPU memory below 95% of actual device memory. It queues 6 tasks per size, concurrency1 each; total at most 2 active training jobs. Use fresh directories for engineering retries and preserve failures. No actual memory estimate should be invented while smoke is pending.

After all 12 scientific runs complete, run scripts/evaluate_controlled_adaptation.py in a CPU allocation. It produces development selection, per-arm metrics, readout exports and paired validation-group uncertainty. scripts/evaluate_adaptation_secondary_test.py is a separate gated GPU pass for the fixed secondary test; no refitting. Final report/plots/interpretation are still outstanding.

## Data-policy correction to preserve

User clarified other-gene CDS overlap in positives is acceptable. The October21 gene-specific audit finds 8 one-base assigned-CDS boundaries plus one compound-CDS overlap; no dataset changes were authorized or made. Do not revive the earlier blanket proposal to generate new negatives from aggregate CDS counts. Current goal is multispecies adaptation, not a human-model comparison.

Preserve pre-existing local scripts/run_training.slurm and remote logs/untracked scripts. Shared SSH socket /home/s34yan/.ssh/nibi-codex.sock was working; verify anew if reconnecting. No credentials/provider configuration were read, and no other LLM was used.
