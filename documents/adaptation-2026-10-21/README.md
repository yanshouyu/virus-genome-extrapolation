# Controlled multispecies adaptation — preparation status

**October 23 update:** The complete experiment and verification are now reported in [the October 23 results](../adaptation-2026-10-23/README.md). The text below records the earlier preparation state; it is no longer the current job status.

As of October 21, 2026, 16:48 Toronto: **prepared and locally tested; GPU execution is blocked by Nibi's scheduler. No new adaptation or downstream scientific results exist yet.** All partitions report down; the two GPU smoke jobs are pending with PartitionDown. This is cluster availability, not GPU memory exhaustion or an SSH failure.

## Completed preparation

- Fixed the [protocol](protocol.md) before training or performance inspection: NTv2 multispecies 100M/500M, unadapted versus baculovirus adaptation versus matched-budget general-genome continued pretraining; three adaptation seeds per size/condition.
- Pinned and cached [both source revisions](sources.json). The 500M revision is unchanged from prior verified work; 100M is f34324c6fde36a4f635f0f1f06cac5d25acd6798.
- Audited the existing 97 baculovirus GenBank records. Excluded NC_001623.1 / AcMNPV; the remaining genomes form 91 species/duplicate groups, with 9 groups held out. There are 14,164 training and 1,448 validation windows, including both orientations. Each is 1536 ACGT bases; no circular wrapping or padding of genomic tails. Fixed MLM evaluation uses 128 windows sampled from the held-out set.
- Downloaded versioned yeast, C. elegans and D. melanogaster FASTA files from the original NT source manifest. The general control has exactly 14,164 windows, allocated 4722/4721/4721 across the three organisms. It is a limited general-genome subset, not the full original multispecies distribution. File hashes, sources and exclusions are in [corpus-report.json](corpus-report.json).
- Transferred matching corpus files to Nibi; all four JSONL hashes match the local manifest. Existing clean_v2, source notebooks and previous experiments remain unchanged.
- Implemented full-parameter MLM training with special-token-safe masking, actual model/tokenizer budget checks, fixed validation masks, synchronized selection among saved trained candidates, finite loss/gradient checks, initial backbone reference fidelity checks in smoke, and exact selected-state reload checks.
- Implemented training-only pooling/readout selection, paired validation-group bootstrap comparisons, fixed readout coefficients across before/after arms, and a separately gated secondary-test evaluator. No new real test file has been used by these scripts.
- Local regression suite: **23 passed, 2 opt-in large pretrained checks skipped**. New tests cover actual tiny NTv2 backpropagation/reload and complete paired development evaluation with no test dataset present. GPU-specific paths and full-size training are not yet validated.

## Queued jobs and resource sizing

| Job | Purpose | GPU request | State |
|---|---|---|---|
| 24757107 | Five-step 100M engineering smoke | h100_2g.20gb:1 | Pending: PartitionDown |
| 24757108 | Five-step 500M engineering smoke | h100_3g.40gb:1 | Pending: PartitionDown |
| 24757086 | Original CPU cache preparation | None | Cancelled; file-download-only caching completed separately |

Both smoke requests have four CPUs, 32GB host RAM and a 15-minute limit. The saved Nibi documentation confirms the 20GB slice provides 2/8 compute and the 40GB slice 3/8 compute. No scientific training arrays have been submitted. NTv1 2.5B remains a possible later robustness check; it has not been launched or assumed to fit these allocations.

The pending smoke jobs can start automatically when scheduling returns. They only run engineering checks. There is no automatic monitoring/wakeup, no recurring automation, and no automatic scientific-array submission. Do not mistake these queued engineering jobs for completion of the adaptation study.

## Next execution steps

1. Check the pending job status and logs. After both finish, inspect loss/gradient finiteness, reference-weight fidelity, actual parameter changes, exact reload, peak memory and measured training time. If either fails, preserve it and diagnose before scientific runs. Use a fresh output directory for any retry.
2. Activate the prepared Nibi environment and run scripts/submit_controlled_adaptation.sh only after inspecting the smoke results. It refuses missing/failed fidelity or memory checks. It submits separate six-task arrays for 100M and 500M, one active task per size, covering both corpus conditions and seeds 42/43/44. Initial scientific walltime is two hours per task; adjust from smoke timings if needed, without changing the scientific budget.
3. After all twelve runs have COMPLETE markers, run scripts/evaluate_controlled_adaptation.py in a CPU allocation. It selects pooling and C using training-only grouped validation and holds them fixed across adaptation arms, then produces the fixed 77-window validation comparison and conditional bootstrap intervals. No downstream test selection occurs.
4. Inspect/freeze development outputs and engineering evidence. Only then run scripts/evaluate_adaptation_secondary_test.py in a small GPU allocation for the fixed secondary test comparison. It refuses to run without completed development selection and refuses to overwrite its output. The already-inspected test remains secondary, not a new external cohort.
5. Collect compact evidence and present per-size paired improvements, every seed, uncertainty, MLM loss, token exposure and GPU costs. Include both corpus arms even if results are negative. Do not add a sweep based on test outcomes.

## Reproduction entry points and limitations

Working root on Nibi: /home/shouyu/projects/def-aucoin/shouyu/virus-genome-extrapolation. Experiment root: runs/2026-10-21/. Local corpus copy is under the same relative path. Source modules: src/baculo_adapt/. Main jobs and analysis are separate scripts, preserving the historical adaptation branch.

Typical environment setup is module load python; module load arrow/23.0.1; source ~/projects/def-aucoin/shouyu/venvs/huggingface/bin/activate; export PYTHONPATH=$PWD/src. Submit GPU work through SLURM only. Corpus preparation used the existing local cpen_14 environment for Biopython; regression tests used ece657d. No packages were upgraded.

The corpus excludes the target genome, but related different species are retained. This does not establish strict unseen-clade independence. The general control represents only three organisms. The fixed 512-step budget tests one declared adaptation regime; a null result would not prove all adaptation methods unnecessary. The frozen linear probe measures representation utility, not maximum supervised fine-tuning performance. No performance conclusion is warranted while execution remains pending.
