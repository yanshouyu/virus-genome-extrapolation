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
