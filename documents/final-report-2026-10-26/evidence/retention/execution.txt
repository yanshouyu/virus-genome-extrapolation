# NTv2 500M multispecies retention execution

Dataset: authors' revised18-task benchmark, revision851f9946252e90c665cdb3cc3eedb78f1f26197c. Official data unchanged. Audit: no train/test shared chromosomes, coordinate overlap, or exact/reverse-complement sequences in any task; global cross-split exact/RC matches0. Deduplicated inference inventory434501train/32388test; all labelled rows retained.

Nibi root /home/shouyu/projects/def-aucoin/shouyu/virus-genome-extrapolation; local root /home/s34yan/projects/260308_virus-genome-extrapolation. Artifacts runs/2026-10-25-retention; compact evidence documents/retention-2026-10-25. Same SSH socket /home/s34yan/.ssh/nibi-codex.sock. Existing Python environment and pinned model/source used, no upgrades.

Initial audit jobs25004286 and25004391 failed before data processing because the wrapper replaced the Arrow module's Python path. Failed logs preserved. Corrected wrapper preserves module paths; audit25004436 completed. This does not change data or scientific outcomes.

Training-only original/adapted smoke jobs25004444/25004455 completed, exact same-batch repeat difference0. Layers resolved from configuration to[8,15,22,29], hidden1024. Batch64; BF16 autocast with FP32 weights and pooled sums. Peak reserved3.55GB.1024-row extraction1.919s(original) /2.534s(adapted), projecting14.6–19.3H100min per fullstate before startup/I/O. Seven-state estimated1.7–2.3GPUhours, up to2 simultaneous GPUs. Four layers are cached for each state for simple reproducibility (about54GB across training/test); only the train-selected layer is used per task.

| Role | Job |
|---|---|
| Fixed common genomic blocks/bootstrap weights |25004589|
| Original training features |25004631|
| Original-only training readout selection,18tasks |25004632|
| Freeze all18 settings |25004633|
| Six adapted training feature sets |25004643|
| Seven benchmark-test feature sets |25004644|
|18 fixed readouts and paired bootstrap tasks |25004662|

Adapted train array starts at concurrency1 while original extraction occupies the other slot. After original completes and no other GPU work is active, its throttle may increase to2. Benchmark test array has concurrency2 and waits for all training extractions AND setting freeze. CPU readout waits for test array. A first readout submission referenced an already-aged-out completed block job and was rejected; only the corrected readout job above was submitted. No extraction duplication.

Data preparation scripts use official parquet. Selection uses original-model train only,3 chromosome-grouped folds, layer quartiles and C grid. Readouts fit independent training-only scalers/classifiers with fixed per-task settings. Benchmark test access is intentional and authorized; the AcMNPV downstream test remains excluded. Raw model states are never modified. Existing unrelated local/remote files remain untouched.

Fresh-process verification job25004863 waits for all benchmark-test extraction. It reloads all seven states and compares all four pooled layers for the first64 rows of both train and test against their caches; no fitting. Final report25004813 waits for BOTH fixed readouts25004662 and replay25004863. It independently checks every feature-file hash, raw dataset hash, source and frozen setting, full-training scaler and full-test coefficient replay before writing COMPLETE. The pure block-bootstrap helpers pass explicit-resampling binary/multiclass checks, including tied probabilities and missing-class resamples.

At20:06Toronto, all18 settings were frozen and a separate local pre-test snapshot was collected (pre-test-freeze-verification.json; frozen-settings-before-test.json). No benchmark test caches existed at that stage. Four tasks choose layer8, fourteen layer15; these choices are not changed for adapted models. Original full training feature extraction finished in695.85s, first adapted in657.78s. CPU release job25005147 raised adapted array concurrency to2 only after the original GPU job completed. Large-cache storage remains on Nibi.

Nibi lacks Matplotlib. Scientific packages are unchanged: the server finalizer performs verification/report tables, and scripts/plot_nt_retention.py renders the figures locally using the existing ece657d environment. This changes no feature extraction, fitting, selection, metric, or bootstrap rule. Local scripts/verify_nt_retention_local.py will independently recompute all metrics and paired intervals from collected compact probabilities and check the pre-test snapshot.

The initial full verifier25004813 was CPU-light but I/O-bound while revisiting memory-mapped feature files in task-first order. Its source and resource snapshot were preserved under runs/2026-10-25-retention/verification-attempts/ and its log retained; it was cancelled before writing results. Replacement verifier25009860 uses32GiB and sequential full-array reads in backbone-first order, checking the same rows, scaler statistics, coefficients, hashes and metrics. No extraction, classifier fitting, model selection or bootstrap generation was repeated or changed. Final reporting now refers to25009860;25004813 remains an explicit engineering attempt in accounting.

All18 task readouts and fresh replay completed successfully. Replacement full verifier25009860 passed every complete-data check; local independent reproduction regenerated the genomic blocks and20000 weights and reproduced all metrics/intervals. Raw completed results are in results/; interpretation.md is the main report. Figures were rendered locally and visually checked. GPU allocation1.813hours, maximum simultaneous study GPUs2. All actual attempts remain in jobs.psv; the two early data-audit failures and cancelled first verifier are explicitly preserved. No additional backbone training or AcMNPV test inference occurred.
