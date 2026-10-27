# Project wrap-up — October 27, 2026

The completed scientific work has been consolidated onto `main`. The portable final report remains unchanged; its 78 package files and 88 HTML links were checked again without training or model inference.

## Recovery locations

- Nibi bulk archives: `/nearline/6002642/shouyu/virus-genome-extrapolation/2026-10-27-wrap-up/experiments/`.
- Nibi archive control, inventories, per-file SHA-256 manifests, Git bundles, unique scripts, and verification logs: `/project/6002642/shouyu/virus-genome-extrapolation-archive/2026-10-27-wrap-up/`.
- Readily accessible selected adaptation weights: the control folder's `checkpoints/2026-10-24/` subtree.
- Local recovery files: ignored `archives/2026-10-27-wrap-up/`, including the local ignored-file tar, pre-cleanup Git bundles from both machines, receipt copies, 14 selected checkpoint copies, and the cache needed to load those selected models offline.
- Published branch recovery points: [branch-archive.json](branch-archive.json).

The initial 171 GiB estimate covered only dated runs. The full inventory also found roughly 334 GiB of historical trainer directories. All historical trainer checkpoints are included in the preservation scope. Tar hard links preserve shared adaptation weight files without duplicating their contents. Uncompressed archives avoid recompressing model and array files; actual sizes and verification status are recorded in the manifests.

## Branch decisions

`dev` and `codex/project-audit` are ancestors of `codex/leakage-and-model-fixes`. The completed branch incorporates the dataset repair, model fidelity fixes, verified baselines, controlled adaptation, retention study, and final report.

`feat/domain-adaptation` has eight unique historical commits ending at `ebc5fbb`. Its source is fully retained by `archive/2026-10-27/domain-adaptation` and both Git bundles. It is superseded by the controlled adaptation pipeline. The [historical failure audit](../experiments-2026-10-19/historical-cuda/README.md) records positional overflow, overlapping-window validation, and target-genome exposure concerns. Its pipeline is archived rather than reintroduced into the supported implementation.

Nibi had an existing stash, `Preserve deployed retention-study files before verified fast-forward`. It is preserved in `nibi-before.bundle` (including `refs/stash`) and left intact on Nibi.

## Checkpoint guarantees

Every archived regular-file payload is SHA-256 hashed while copied, then independently hashed by reading the finished tar. The whole tar has its own SHA-256. File membership and original file identity/size/modification time are checked before originals can be removed. Sources are not deleted by the archive tool.

Historical Trainer checkpoints retain all existing model, head, optimizer, scheduler, random-state, tokenizer, configuration, and trainer-state files. Their historical scientific limitations remain unchanged.

The completed controlled/duration adaptation programs saved selected weight dictionaries and LoRA adapters, not optimizer/scheduler/random-state snapshots. These support evaluation and warm starts using the pinned upstream model; they do not support exact interrupted-training resumption. All existing files and metadata are retained. Cached tokenizer, configuration, custom model code, and 16,462,217,726 bytes of upstream weights are included in the recovery package; access credentials are excluded. The upstream weights support reconstruction of LoRA adapters and pinned reference models without a future download. The [accessible checkpoint index](accessible-checkpoints.json) records the 14 endpoint weight hashes.

Current environment inventories for local ece657d/cpen_14 and Nibi huggingface are preserved with the private recovery files. They accompany historical per-run metadata; they do not establish the versions of older runs that did not record their environments.

## Restore an experiment

Locate the archive and member path in the JSON receipts. Check the tar's SHA-256 against its receipt before restoring. From the Nibi repository root:

```bash
tar -xf /nearline/6002642/shouyu/virus-genome-extrapolation/2026-10-27-wrap-up/experiments/experiment-2026-10-24.tar runs/2026-10-24
```

The compact local analysis inputs retain their original collected/ layout in the local recovery tar. From the local WSL repository root, restore the dated inputs with:

```bash
tar -xf archives/2026-10-27-wrap-up/payload/local-ignored-artifacts.tar --wildcards 'runs/2026-*'
```

The local tar's detailed original manifest is also retained under the Nibi control folder's manifests/local/ directory.

For an offline model rebuild, restore the cached upstream models and custom code from the combined archive, then point Hugging Face at that cache:

```bash
tar -xf /nearline/6002642/shouyu/virus-genome-extrapolation/2026-10-27-wrap-up/experiments/baseline-and-recovery.tar archives/2026-10-27-wrap-up/model-support
export HF_HOME="$PWD/archives/2026-10-27-wrap-up/model-support"
```

Use an empty destination when testing recovery to avoid overwriting working files. Member paths preserve the original repository-relative layout. The combined `baseline-and-recovery.tar` also contains the copied local tar and pre-cleanup bundles under `archives/2026-10-27-wrap-up/`. The local tar has its own manifest and original local repository-relative paths.

For a Git recovery checkout:

```bash
git clone /path/to/repository-before.bundle recovered-project
git switch -c historical-adaptation ebc5fbb2edb4896bcbcf78e5960d9ed364230ce0
```

The pre-cleanup bundle predates the new archive tags, so the command above uses the full preserved commit. Published archive tags remain available from GitHub.

The archive tool can recheck a cold archive after originals have been removed:

```bash
python3 archive_project_artifacts.py archive-plan.json --verify-only --without-source
```

Run this from the Nibi archive-control folder. It reads every tar and writes separate reverification receipts, preserving the original verification receipts.

## Storage behavior

Nearline is intended for inactive data. It may take more than 24 hours to create Nibi's initial tape copy, and small files/indexes are not backed up there. Detailed manifests and Git bundles therefore remain in project storage and locally; the compact index is versioned on GitHub. Tape migration status must be reported separately from successful archive verification.

Project originals are eligible for removal only after the corresponding archive is fully verified and its receipt is copied locally. Selected endpoint adaptation weights stay readily accessible under the project archive control folder. Reproducible Python caches are excluded from the permanent local archive.

## Validation

- [Regression suite](regression-tests.txt): 43 passed, two opt-in pretrained-weight tests skipped.
- [Archive tool tests](archive-tool-tests.txt): 10 passed: restoration, symbolic/hard links, corruption detection, source mutation, path traversal, off-host receipt matching, tracked-file protection, and idempotent retirement.
- [Report integrity](report-integrity.json): all 78 report files match their recorded hashes; all 88 local HTML links resolve.
- [Standard tar restoration](restore-smoke.json): archived model provenance restored with the expected checksum.
- [Checkpoint provenance](checkpoint-provenance-verification.json): all 82 endpoint/budget states match the hashes recorded in the completed studies.
- No new scientific training, feature extraction, benchmark inference, or public model publication.

## Completion

Both checkouts and GitHub are on main; the four stale branches have been retired with published recovery tags. All 11 archive packages have been independently verified and their receipts saved locally before the corresponding Nibi originals were retired. The [archive catalog](archive-catalog.json) records exact destinations, sizes, checksums, source paths, and completion times. The archived payload totals 517.54 GiB (555.71 GB). The 14 selected endpoint checkpoints remain accessible in project storage. Tape migration is tracked separately in [tape-status.json](tape-status.json): 5 of 11 tape copies were confirmed at 2026-10-27T16:51:27.070867+00:00.

The [final quota report](storage-after.txt) records project usage falling from 515 to 29 GiB, approximately 486 GiB (522 GB) reclaimed, and nearline usage of 518 GiB. The archive-file payload itself is 555,706,746,880 bytes.

The [independent local checkpoint copy](local-checkpoint-verification.json) contains 14 selected states and 47 model-support files (16.70 GB), all matching their original hashes. The full upstream cache, including historical reference models, remains in the combined nearline archive. The selected states also remain directly accessible in Nibi project storage.

The [local cleanup receipt](local-cleanup.json) records 2.29 GB of unchanged bulky originals retired after verifying both recovery copies. Small local working datasets, editor settings, and notebook notes remain available with archived snapshots.

Final Git recovery bundles are stored under the control folder’s git/ directory and the local recovery folder’s git/ directory: repository-final.bundle and nibi-final.bundle. Their standalone restore/object checks and SHA-256 values are recorded in git-bundle-verification.json beside those bundles.
