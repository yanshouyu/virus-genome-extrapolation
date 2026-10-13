# Project assessment and continuation plan

Date: 2026-10-13. Baseline: `dev` at `42b39b5`. Scope: proposal and course report, tracked source and Git history, local data/results, cached model smoke checks, and a targeted literature check. No HPC login, training, dependency installation, or dataset regeneration was performed. Documented future work in the PDFs is background, not an instruction to execute it now.

## Assessment

The project has a useful initial dataset and a working frozen-model probe. Its current evidence supports an exploratory difference between two constructed classification tasks. It does **not** yet establish that phylogenetic distance causes the gap, that NTv2 is unsuitable, or that domain adaptation is necessary. The highest-value next work is evaluation validity and model fidelity, before larger models or more training.

The existing directory layout is adequate. Keep it. Do not add a framework, workflow engine, model registry, or parallel training system at this stage.

## What exists

The proposal envisioned comparative probing followed by optional interpretation and adaptation. The report delivered human/virus probing with NTv1 (two head sizes), NTv2, and an extended viral training run. Adaptation remained unfinished.

The raw TSS table comes from Chen et al. (2013), supplement S2. The fixed coordinates and downloaded reference are local. Current positives comprise 205 windows; negatives comprise 186. The processed split is 312 training rows (164 positive, 148 negative) and 79 test rows (41 positive, 38 negative), all 300 bases. There are 97 local GenBank files; this count alone does not establish independent species or nonredundant genomes.

The notebook documents 216 source records, then four within-associated-ORF removals, yielding 212, then deduplication to 205. Local interim files confirm 212 and 205. Reconcile this with the report's wording about 220/216. Also, the notebook notes 64 retained TSS inside another same-strand ORF: the filtering was not a genome-wide exclusion of all coding sequence. Negatives exclude same-strand CDS and positive windows, so opposite-strand coding sequence is allowed. These are sampling definitions requiring explicit reporting, not necessarily errors to silently correct.

### Course results recovered from saved histories

These are **last logged evaluations**, not selected best checkpoints or new experiments.

| Backbone / head | Task | Accuracy | MCC | Last step |
|---|---|---:|---:|---:|
| NTv1 / 1280 | Human | 0.8535 | 0.7073 | 3000 |
| NTv1 / 1280 | Virus | 0.6962 | 0.3951 | 350 |
| NTv1 / 512 | Human | 0.8529 | 0.7059 | 3000 |
| NTv1 / 512 | Virus | 0.7342 | 0.4984 | 350 |
| NTv2 / 512 | Human | 0.5688 | 0.1377 | 3000 |
| NTv2 / 512 | Virus | 0.5443 | 0.0856 | 350 |
| NTv1 / 512, extended | Virus | 0.7342 | 0.4688 | 3500 |

Sources: local `notebooks/trainer_states/`, currently Git-ignored. The extended run's minimum logged evaluation loss is at step 640, not exactly 600. With seven updates per epoch, 640 is about 91.4 epochs. Archive original logs and associated code/environment before interpreting exact historical behavior; current source alone cannot prove which implementation each job executed.

## Findings that affect publication

1. **Overlapping genomic regions cross the split.** The read-only audit finds 65 train/test interval pairs sharing bases; 23 are same-strand pairs. 41/79 test windows overlap training territory, including 20 on the same strand. Maximum overlap is 297/300 bases. There are no exact sequence duplicates across the split. Opposite-strand overlap is shared genomic territory, not necessarily identical model input. The strongest same-strand example has train interval `[116182,116482)` and test interval `[116185,116485)`. See the hashed [audit result](audit-2026-10-13/dataset.json).
2. **The named test set is used for periodic evaluation.** `prep_ds()` returns train/test and `train.py` passes test as `eval_dataset`. Inspecting this curve to choose training duration or architecture uses it for development. Preserve it as the course evaluation set; establish independent validation and final evaluation for new claims.
3. **Data volume and task construction are confounded with domain.** Human training has 30,000 samples versus 312 viral samples. Matching optimizer steps does not match sample diversity. Positive placement, negative definitions, annotation uncertainty and within-genome correlation also differ. Separate heads trained in each domain measure representation usefulness, not zero-shot transfer of a human-trained classifier.
4. **Clustering did not establish independence.** Positive and negative classes were clustered separately, with full-length coverage required. Offset windows can share nearly all bases yet fail that coverage requirement. Cluster disjointness is weaker than region/homology disjointness. MMseqs2 `--cov-mode 0` uses the longer sequence denominator, contrary to the old README. [MMseqs2 documentation](https://github.com/soedinglab/MMseqs2/wiki#how-to-set-the-right-alignment-coverage-to-cluster).
5. **Uncertainty is not quantified.** One split and one recorded seed with 79 correlated test samples cannot support precise generalization claims. Confidence intervals must respect the unit of independence, rather than treating overlapping windows as independent observations.

## Nucleotide Transformer compatibility

### Confirmed locally

Environment: Python 3.14.3; Transformers 5.3.0; PyTorch 2.9.1; datasets 4.7.0; NumPy 2.4.2; pandas 2.2.3; accelerate 1.12.0. `pip check` reports no broken installed requirements. This is not proof of compliance with `pyproject.toml`: torch, datasets and pandas differ from its declared requirements, and the project is not importable without `PYTHONPATH=src` in this environment. Trainer's accelerate dependency is also absent from the project dependency declaration. No environment was altered.

Cached revisions:

- NTv1 human-ref: `f87b5d7233295242e79c951873d290f4cf99204598`.
- NTv2 500M multispecies: `06615c1660c892fc199840c18123f8385b3542a8`.

Using the current project loader, both cached 500M models loaded offline on CPU and returned finite logits for two short synthetic inputs. This checks basic execution, not scientific correctness, pretrained-weight parity, training, or historical HPC behavior.

The current global patches address removed pruning/head-mask APIs and missing tied-weight metadata. The upstream [NTv2 custom source](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species/blob/main/modeling_esm.py) still contains legacy API calls. Tiny MLM construction via `AutoModelForMaskedLM.from_config` fails on missing `all_tied_weights_keys`, even though the patched pretrained-loading path succeeds. These are different paths; do not generalize success from one to the other.

**Checkpoint reconstruction is broken for NTv2.** `SequenceClassification.__init__` rebuilds `backbone_config` through Transformers' built-in `CONFIG_MAPPING['esm']`. NTv2's custom gated feed-forward architecture is lost. A tiny cached-code encoder with hidden size 16 and intermediate size 32 has intermediate weights `[64,16]`; the reconstructed standard encoder expects `[32,16]`. Saving the tiny wrapper succeeds, but reloading raises a shape-mismatch error. Do not bypass this using `ignore_mismatched_sizes=True`: it would discard learned parameters. See [reproducer](../scripts/check_nt_checkpoint.py).

**Pooling differs across models.** NTv1 loading reports missing pooler weights, initializes them randomly, and then freezes them with the backbone. The classifier uses that pooler. NTv2 has no pooler and uses the first-token representation. This is an uncontrolled difference, not proof of the cause of NTv2's weak course results. The classifier is also created before Trainer sets its seed. Seed before initialization and make the representation policy explicit in the next implementation.

**Other concrete risks:** initialization builds a full temporary backbone before replacing it (avoidable memory overhead); freeze-by-`requires_grad` does not itself keep the encoder in evaluation mode; default collation assumes equal token lengths; tokenizer files were absent in the locally saved checkpoint examined, while `model_path` loads its tokenizer from that folder. Resolve these in a small, separately validated repair, not with additional broad monkey patches.

### Recommended compatibility decision

Keep Hugging Face as the first route to validate. Test an isolated Transformers 4.x environment against the pinned cached NTv2 revision before choosing a supported version; no exact 4.x version has been validated here. Staying on 5.3 requires architecture-aware construction and loading plus tightly scoped compatibility handling. Either route still needs the wrapper reload repair. Switching to the authors' `nucleotide_transformer` implementation is a larger backend migration, not an automatic fix; evaluate it only if the smaller route cannot satisfy parity checks. [Official implementation](https://github.com/instadeepai/nucleotide-transformer).

Acceptance checks: pinned model/code revisions; fixed synthetic inputs; reference-versus-wrapper hidden-state agreement; no unexplained missing backbone weights; finite forward/backward; only intended head parameters updated; deterministic evaluation; checkpoint reload preserves logits and tokenizer behavior. Follow with an HPC smoke job only when access and resources are agreed.

## Git, local artifacts and HPC

At audit start, `dev` and remote `dev` were `42b39b5`; `main` was `65d92db`. The local adaptation branch was `17952bf`, two commits behind its remote (`ebc5fbb`). Remote refs were verified read-only. Adaptation source is on that separate branch; local `src/nt_domain_adapt/__pycache__` is residue, not source to restore from. No branches were merged.

The existing untracked `scripts/run_training.slurm` was preserved. It uses a personal environment and currently requests a 500-epoch viral run. Its final shell status can mask a failed Python command. On the remote adaptation branch, a commented argument interrupts a backslash-continued Python command; subsequent arguments become a separate shell command. `bash -n` alone does not catch this semantic issue. Repair and smoke-check scripts when preparing the next authorized HPC run.

Historical adaptation code randomly splits overlapping windows and drops accession metadata before splitting. Its validation therefore does not establish held-out-genome generalization. A separate original CUDA error is **unresolved**: there is no traceback here sufficient to identify its cause. The report mentions a 2.5B attempt; later branch history switches to 500M. Recover the exact job log and commit before attributing the error to CUDA, token limits, or hardware.

The supplied [Nibi manual](https://docs.alliancecan.ca/wiki/Nibi) returned an access-denied page to automated retrieval. Current module versions and resource names were not independently verified. No HPC login or job submission occurred.

## Publication-oriented next steps

### 1. Preserve and validate the evidence

Archive the course histories/configurations with checksums, run-to-commit mapping where recoverable, and environment snapshots. Retain the course dataset as version 1. Reconcile source-table counts and annotation definitions against the original supplement. Recover HPC logs, especially the CUDA failure, when convenient.

Deliverable: a compact provenance table distinguishing known facts, inferred settings and missing metadata.

### 2. Repair model fidelity before running new comparisons

Choose and verify one software stack; repair architecture-preserving reloads, seeding, representation policy and tokenizer saving. Use small synthetic checks before any expensive run. Do not silently change the historical baseline or claim that a successful forward pass validates its results.

Deliverable: passing compatibility and checkpoint round-trip checks with pinned revisions and documented pooling.

### 3. Agree an evaluation protocol before further modeling

Define independent units and train/validation/final-test responsibilities. Audit genomic overlap, reverse complements and homology jointly across labels. Decide how to handle already-inspected course test data. Use group-aware uncertainty estimates and distinguish within-genome evaluation from claims about other viruses. Include controls for training-set size, class balance and sampling definitions, with simple non-neural references. Predeclare primary metrics and a limited comparison budget.

Deliverable: a short protocol with acceptance criteria; no new dataset is generated during this audit.

### 4. Reassess the research claim and novelty

A defensible initial question is whether the apparent cross-domain performance gap remains after accounting for evaluation leakage, sample size and representation choices. A negative result can be useful if the benchmark and controls are rigorous. A single-genome demonstration alone will not establish broad viral generalization; independent annotations or an external evaluation would materially strengthen it.

The targeted literature search is not a novelty clearance. Older baculovirus promoter prediction already exists ([2005 study](https://pubmed.ncbi.nlm.nih.gov/15908030/)). Modern cross-species DNA-model benchmarks also exist, including a 2025 study showing substantial dependence on embedding pooling ([Feng et al.](https://www.nature.com/articles/s41467-025-65823-8)). Homology leakage is an established concern ([2025 preprint](https://www.biorxiv.org/content/10.1101/2025.01.22.634321v1.full)). Position any paper around the controlled evaluation and reusable evidence, not merely applying NT to a new organism. Read these in detail before claiming novelty.

Only after these milestones should we decide whether additional models, domain adaptation or interpretation are justified. Do not revive the larger adaptation pipeline merely because it exists.

## Decisions for the next conversation

No further user input was required to complete this audit. Before new experiments, agree on available independent annotations, the intended breadth of the claim, how much course evidence can be archived publicly, and an HPC resource budget. The recommended immediate task is the small model-fidelity repair and its acceptance checks.
