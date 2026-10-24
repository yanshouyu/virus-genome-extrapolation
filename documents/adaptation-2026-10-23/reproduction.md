# Reproduce the October 23 adaptation results

Use the committed protocol plus October 23 amendments. The canonical corpus/source root remains runs/2026-10-21/; output directories refuse overwrites. Keep the excluded individual-strand controls separate. All commands below are from the project root.

## Environment and data identity

Nibi used the existing environment at ~/projects/def-aucoin/shouyu/venvs/huggingface after module load python and module load arrow/23.0.1. No package upgrades were made. Training metadata records torch 2.10.0+computecanada, transformers 5.3.0+computecanada and numpy 2.4.2+computecanada. [analysis-environment.json](analysis-environment.json) records the fitting libraries and analysis-source hashes. Preserve these environments for exact reproduction rather than assuming an arbitrary latest install.

Pinned sources are recorded in the preparation sources.json and each run's metadata; the exploratory capacity metadata records the 2.5B revision and both source-shard hashes. Original pretrained files must be cached for offline jobs.

Reuse the preserved JSONL corpora in runs/2026-10-21/corpus/. The valid general_train.jsonl hash is 6455320fd95ae86d27d32c80c101b18c9529cc064f252e7972181a947d852e04. The old unpaired file remains as general_train-unpaired-v1.jsonl and must not be used. The current corpus-report.json contains the corrected control and original inputs; the committed [v2 report](corpus-report-v2.json) provides its identity. Raw GenBank versions/hashes and public general-source URLs/archive hashes are retained there.

For a clean corpus rebuild, use scripts/prepare_adaptation_corpus.py with a new --output directory and --general-manifest documents/adaptation-2026-10-21/original-general-source-manifest.txt, in the existing local cpen_14 environment with Biopython. The corrected implementation samples loci first and includes both strands. Compare resulting hashes before substituting data into the canonical training root. Re-downloads can change record annotations; saved corpora and manifests define this experiment. scripts/repair_general_control_pairing.py documents the one-time v1-to-v2 correction and intentionally refuses to repeat that move.

## Scientific training

Use SLURM only. The 100M configuration is microbatch 4, accumulation 8 on a 20GB H100 MIG; 500M is microbatch 16, accumulation 2 on a full H100. Both keep effective batch 32 and 512 optimizer steps. Repeat each condition for seeds 42/43/44. A fresh example:

```bash
sbatch --gpus=h100_2g.20gb:1 --time=00:30:00 \
  scripts/run_controlled_adaptation.slurm \
  --size 100m --condition baculo --seed 42 --microbatch 4 \
  --output runs/reproduction/100m-baculo-seed42
```

For 500M substitute --gpus=h100:1, --size 500m and --microbatch 16. For the control use --condition general. Use fresh output paths for every run. The runner loads pinned sources.json and corrected corpus files from runs/2026-10-21, masks only ordinary tokens, uses fixed held-out masks for checkpoint selection, and never opens the downstream test CSV.

Do not rerun the old array launchers blindly into completed scientific directories. The valid completed task map is:

- 100M baculovirus: 24896408_0, _2, _4.
- 500M baculovirus: 24896596_0, _2, _4.
- 100M corrected general: 24897074_1, _3, _5.
- 500M corrected general: 24897075_1, _3, _5.

All earlier general-control outputs are excluded and preserved under runs/2026-10-21/excluded-general-unpaired/ on Nibi; local copies are in runs/2026-10-23/excluded-general-unpaired/. Original array aggregate CANCELLED states do not invalidate completed baculovirus tasks.

Each valid scientific directory contains selected_state.pt, history.json, metadata.json, before/after training and validation embeddings, and COMPLETE. Source/selected-state hashes, selected step, token exposure, timing and peak memory are recorded. Only compact artifacts were copied locally; large weights remain on Nibi.

## Verification and development analysis

Source checks: scripts/check_adaptation_source_fidelity.py in a CPU allocation. Fresh adapted replay: scripts/check_adapted_embeddings.py --size 100m or 500m in a GPU allocation. These scripts use the canonical completed run root, compare state tensors and regenerate all train/validation embeddings. Saved fresh-replay JSON evidence is committed.

Readouts were independently frozen by scripts/freeze_adaptation_readouts.py using only unadapted training rows, before adapted TSS outcomes were inspected. Final development uses scripts/evaluate_controlled_adaptation.py. It recomputes training-only nested pooling/C selection and asserts that the choices match frozen-readout-selection.json. It then fits each fixed-C readout on training only, computes validation predictions and paired group-bootstrap intervals. A partial --sizes 100m panel writes SIZE_PANEL_COMPLETE; the full panel writes COMPLETE. Partial/full reporting did not lead to retuning.

Compact Nibi outputs are in runs/2026-10-21/development/. Local copies are in runs/2026-10-23/development/ and runs/2026-10-23/collected/. To verify and regenerate reporting locally with the existing ece657d environment:

```bash
python scripts/verify_adaptation_readouts.py \
  --collected runs/2026-10-23/collected \
  --development runs/2026-10-23/development \
  --capacity-root runs/2026-10-23/capacity-2.5b \
  --output documents/adaptation-2026-10-23/readout-verification.json

python scripts/summarize_controlled_adaptation.py \
  --collected runs/2026-10-23/collected \
  --development runs/2026-10-23/development \
  --capacity-root runs/2026-10-23/capacity-2.5b \
  --output documents/adaptation-2026-10-23

python scripts/complete_adaptation_checks.py
```

The summarizer requires corpus-report-v2.json in its output directory, rejects old control hashes, checks selected steps and paired token counts, and writes the main table, figures, compact prediction CSVs and provenance. The supplementary checker uses saved predictions for fixed-model k-mer comparisons; it does not fit new models. Its data-identity check hashes the test CSV but performs no test inference.

## Exploratory larger-model reference

scripts/extract_capacity_reference.py loads the cached NTv1 2.5B multispecies model, verifies all source parameters/shard hashes and extracts frozen train/validation embeddings. It ran as job 24897906 on a full H100 with 64GB host memory. scripts/evaluate_capacity_reference.py chooses pooling/C with the same training-only grouped procedure and evaluates validation; CPU job 24897926. This is a practical unadapted reference, not an architecture-matched capacity experiment or a 2.5B adaptation result. Existing capacity output directories also refuse overwrites.

## Test restriction

No real downstream test predictions were made in this window. The secondary-test script remains uninvoked. Do not treat a development COMPLETE marker as authorization to expand the scope or tune to the existing test; any later evaluation should preserve a frozen protocol and the cohort's previously inspected status.
