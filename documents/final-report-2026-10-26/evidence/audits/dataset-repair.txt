# Repairs completed on 2026-10-13

The active dataset is [clean_v2](../../data/processed/clean_v2/README.md). The user's policy permits opposite-strand overlap and prohibits same-strand overlap between splits. The course source files and the original untracked `scripts/run_training.slurm` were preserved. No HPC connection, job submission, or biological-model training was performed.

## Why the course split overlapped

The extraction created a 300-base window for each retained TSS, with placement chosen within annotation-derived boundaries. Distinct TSS can lie in the same regulatory region, including alternative starts associated with the same ORF. Nearby genes also yield overlapping windows. Variable placement means TSS separated by more than 300 bases can still have intersecting windows.

All 23 forbidden train/test interval pairs in the old split are positive-positive pairs. Linking the zero-based identifier anchors back to the one-based annotation table shows:

- 18 pairs have the same associated ORF.
- 5 pairs have different nearby associated ORFs.
- The largest overlap is 297 bases, from starts four bases apart associated with ORF 116492-117391.

Thus, close genes contribute, but most observed leakage pairs arise from multiple starts associated with one ORF. This is an annotation-based explanation; it does not independently establish the functional relationship of each transcript. [Pair-level evidence](overlap-origin.csv).

The software cause was that interval independence was never enforced. The old split grouped per-class sequence clusters. Clustering required full-length coverage, so shifted 300-base windows could share almost every base and remain separate clusters. A 297-base overlap is still less than 300-base coverage. Merely lowering sequence identity cannot solve that coordinate constraint.

## Dataset repair

The generator now creates connected groups before partitioning. Same-strand overlap joins groups transitively; same associated ORF, same-strand duplicate/reverse-complement sequences and known same-strand cluster membership also join them. The grouping is joint across labels. Opposite-strand overlaps are allowed explicitly.

Label checking discovered an additional extraction problem:

| Excluded source identifier | Reason |
|---|---|
| `129683|129704:130004|+|1` | Its TSS is before the window; no recorded same-strand TSS is inside. |
| `132967|132651:132951|-|1` | Its TSS is after the window; no recorded same-strand TSS is inside. |
| `neg_15|129388:129688|+|0` | Contains the recorded positive-strand TSS at one-based position 129684. |

The notebook's placement function does not explicitly require the named TSS to lie inside the final window. The negative sampler excludes extracted positive windows rather than every actual TSS point; the first misplaced positive therefore leaves its TSS available to a negative window. The clean generator validates against the curated raw TSS table and excludes the three invalid rows, rather than silently moving boundaries or changing labels. The historical notebook and extracted source FASTA remain unchanged; rerunning them is still guarded by the new export validation.

The final export contains 388 unchanged samples in 319 groups: train 233 (122/111), validation 77 (40/37), test 78 (41/37). All 300-base sequences match the downloaded reference after strand orientation. Every retained positive includes its named recorded start; no negative contains a recorded same-strand start. Independent checks across all three splits find no shared ORF/group, same-strand interval overlap, duplicate identifier or same-strand exact/reverse-complement duplicate. Opposite-strand overlap remains in 97 pairs, as permitted.

[Manifest and source hashes](../../data/processed/clean_v2/manifest.json), [audit](dataset-audit.json), [generation summary](split-generation.json). Export reproduction is tested byte for byte. Arbitrary distant homology and foundation-model pretraining overlap have not been exhaustively audited; these checks are the documented dataset guarantees, not a claim of independent unseen viral cohorts.

## Model and training repairs

- NTv2 reload now reconstructs the custom encoder and its gated feed-forward layers instead of standard ESM. The saved source revision is used to retrieve its custom code.
- Legacy API handling is confined to NTv2's custom classes. Importing the module no longer patches `PreTrainedModel` globally; the obsolete pruning import stub exists only while loading custom code.
- The custom encoder's initialization protects loaded tensors inside the local classifier wrapper. In Transformers 5.3, the wrapper otherwise causes initialization to treat its remote-code children as local, permitting NTv2's direct tensor writes to overwrite checkpoint weights. Fixing shapes alone was insufficient; exact parameter and logit comparison detected this.
- New probes use raw first-token embeddings consistently for both NTv1 and NTv2, with no newly initialized frozen pooler. Existing saved checkpoints retain their legacy representation policy when that policy is absent from old configuration files. This is a changed experimental baseline; compare new results under the new policy, not as continuations of course curves.
- Loaded encoders are reused directly, eliminating the temporary second full-sized backbone. Frozen encoders stay in evaluation mode even during classifier training.
- The two tested 500M model revisions are pinned by default; `--model-revision` permits an explicit override. Checkpoint loading still needs the pinned custom code cached or reachable on Hugging Face.
- Seeding occurs before model/head initialization. Run IDs are generated per instance, and new outputs go under ignored `runs/` directories.
- Training uses separate validation data for model selection by validation loss. It selects among saved checkpoints, then performs one final test prediction. Metrics, prediction arrays, model/tokenizer files, Git state, package versions, dataset fingerprints and manifest are recorded.
- Variable token lengths use dynamic padding. Metadata fields such as group and name are removed before model input. Tokenizers are saved with checkpoints and final models; old checkpoints can fall back to the recorded original tokenizer source.
- The missing Trainer dependency, `accelerate`, is declared. Existing dependency pins were otherwise preserved, and the local environment was not altered. The tested local package versions still differ from the server-oriented pins; a clean install of those exact pins was not tested. BioPython is absent from this local environment, so the legacy extraction scripts were not rerun.

For human data, the official test split remains reserved for final prediction. If an official validation split is absent, validation is stratified from official training data. Because genomic coordinates are not provided by this loader, that human validation split does not carry the same region-independence guarantee as the viral export.

The validated cached NTv1 revision is `f87b5d7233295242e79c951873d290f4cf992045`; the NTv2 revision is `06615c1660c892fc199840c18123f8385b3542a8`. The initial assessment's NTv1 revision string had extra trailing characters from combined command output; that documentation was corrected.

## Verification

The standard regression suite passes 17 tests; its two large pretrained checks are opt-in and were also run separately, both passing. Coverage includes transitive overlap, opposite-strand permission, mixed labels, ORF grouping, exact/reverse-complement duplicates, source-label validation, preserved retained samples, reference-genome extraction, deterministic byte-identical exports, loader rejection of invalid data, and dynamic tokenization.

Both tiny built-in and custom encoders preserve all saved parameters and logits exactly, and only classifier parameters update in a synthetic optimizer step. A tiny synthetic Trainer run verifies checkpoint selection on validation, exactly one final-test prediction, tokenizer exports and an unchanged encoder. Both actual cached 500M encoders match every pretrained encoder parameter exactly; their wrapper CLS predictions agree with directly computed encoder representations and remain identical after full checkpoint save/reload.

[Regression output](tests.txt), [actual pretrained checks](pretrained-checks.txt). The standard suite's CPU-only pinned-memory warnings are expected. Commands from WSL repository root:

```bash
conda activate ece657d
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 python -m pytest -q
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RUN_PRETRAINED_CHECKS=1 OMP_NUM_THREADS=2 python -m pytest tests/test_pretrained_models.py -q
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/check_nt_checkpoint.py
```

## Nibi preparation and remaining work

The user's saved `Nibi - Alliance Doc.html` was read as reference material. It confirms the `h100_1g.10gb` instance name, 10 GB GPU memory, public internet access on all nodes, and interactive access through SSH or Open OnDemand with multifactor authentication.

A separate [job template](../../scripts/run_probe.slurm) is ready for manual review/submission. It uses the course account/environment/module choices, checks the dataset and CUDA availability, and propagates Python failures. Shell syntax and failure propagation were tested with simulated commands locally. Module `arrow/23.0.1`, the environment path and resource sufficiency need verification on the actual cluster; they are not established by a local shell test. The saved manual does not specify those Python/Arrow module versions. No emails, logins or jobs were initiated.

Submit from the repository root after preparing the HPC environment:

```bash
sbatch scripts/run_probe.slurm --organism virus --epochs 50
# Different prepared environment:
PROBE_ENV=/path/to/environment sbatch scripts/run_probe.slurm --organism virus --epochs 50
```

The next useful external step is a short GPU smoke job and environment snapshot, followed by an agreed experiment protocol and resource budget. No claim of improved biological performance is made from these software checks. The earlier adaptation CUDA failure still requires its original job traceback, and the separate adaptation branch's overlapping-window validation and shell issues have not been merged into this work. Revising that pipeline should follow the evaluation decision rather than revive it wholesale.
