# Reproduction notes

From the project root in WSL, using the existing `ece657d` environment:

```bash
PYTHONPATH=src OMP_NUM_THREADS=2 python -m pytest -q
OMP_NUM_THREADS=2 python scripts/run_sequence_baselines.py --output-dir runs/2026-10-19/sequence-baselines-new
```

On Nibi, use the prepared Python/Arrow environment and submit from the repository root:

```bash
mkdir -p runs/2026-10-19
sbatch scripts/run_baseline_array.slurm
# Substitute the returned six-task array ID below.
sbatch --dependency=afterok:BASELINE_ARRAY_ID scripts/check_baseline_reloads.slurm
```

The launcher refuses to overwrite an existing run directory. Corrected scientific runs use `runs/2026-10-19/verified/`. For a deliberate rerun, set `EXPERIMENT_ROOT` to a fresh path for both launchers. It performs frozen-encoder probing, not backbone fine-tuning. The dataset, source model revisions, head settings and seed schedule are fixed by [protocol.md](protocol.md).

The original engineering launch used two disjoint arrays with a dependency. After correcting checkpoint selection, the scientific launch used one six-task array with a concurrency limit of two. The two initial smoke jobs were separately bounded engineering checks. The historical position diagnostic was a tiny CPU job. Initial engineering NT results are excluded from the scientific comparison.

Collect each run's `config.json`, `run_metadata.json`, `trainer_history.json`, `test_metrics.json`, `test_predictions.npz` and `gpu_reload_check.json`. Also copy the numerically last retained checkpoint's `trainer_state.json` to the collected run directory as `trainer_state.json`. Trainer saves at the 30-step intervals and also at the final step 250; the summary includes that final checkpoint when verifying validation selection. Keep the full selected/final models and tokenizer on Nibi under the original ignored `runs/` paths.

Then, from the local repository root:

```bash
python scripts/summarize_baselines.py \
  --run-dir runs/2026-10-19/verified \
  --baseline-dir runs/2026-10-19/sequence-baselines-verified \
  --output-dir documents/experiments-2026-10-19 --plot
```

The summary checks prediction label order and saved checkpoint selection, recalculates test metrics from stored probabilities/logits, and bootstraps the 64 test groups together across models. No test-based model selection or threshold tuning occurs. Its checkpoint-step reconstruction intentionally matches this fixed experiment schedule; it is not a generic analysis tool for arbitrary training settings.

Within a CPU SLURM allocation in the prepared Nibi environment (allow at least 16 GB RAM for comfortable headroom), verify the saved encoders against the pinned cached references:

```bash
python scripts/check_frozen_backbones.py --run-root runs/2026-10-19/verified
```

This uses only locally cached weights, compares every saved backbone tensor with its pretrained counterpart, and writes one `frozen_encoder_check.json` per run. The recorded CPU job used 12 GB and completed, with peak resident memory close to that limit. The separate GPU replay checks load full saved models and compare logits against the original prediction arrays; they do not tune or refit a model.

The simple sequence baseline exports standardization statistics and logistic coefficients to `kmer_model.npz`. Features are ordered by k=1,2,3,4 and then Cartesian product order over A,C,G,T. Apply `(x-mean)/scale`, then the stored linear coefficients/intercept and sigmoid. The exported parameters were replayed against saved validation/test probabilities; a deterministic rerun reproduced the original metrics exactly.

Local CPU and Nibi package versions differ. The sequence baseline records its own NumPy/scikit-learn versions; each Nibi run records its Python and training-package versions. Interpret reproducibility against those recorded environments rather than assuming an exact installation of every declared dependency was tested.
