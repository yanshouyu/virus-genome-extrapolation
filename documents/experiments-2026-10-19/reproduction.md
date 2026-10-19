# Reproduction notes

From the project root in WSL, using the existing `ece657d` environment:

```bash
PYTHONPATH=src OMP_NUM_THREADS=2 python -m pytest -q
OMP_NUM_THREADS=2 python scripts/run_sequence_baselines.py --output-dir runs/2026-10-19/sequence-baselines-new
```

On Nibi, use the prepared Python/Arrow environment and submit from the repository root:

```bash
mkdir -p runs/2026-10-19
sbatch --array=0-2%2 scripts/run_baseline_array.slurm
# Substitute the returned NTv1 array ID below.
sbatch --dependency=afterok:NTV1_ARRAY_ID --array=3-5%2 scripts/run_baseline_array.slurm
# Substitute the returned NTv2 array ID below.
sbatch --dependency=afterok:NTV2_ARRAY_ID scripts/check_baseline_reloads.slurm
```

The launcher refuses to overwrite an existing run directory. For a deliberate rerun, first change the dated output root in a copy of the launcher. It performs frozen-encoder probing, not backbone fine-tuning. The dataset, source model revisions, head settings and seed schedule are fixed by [protocol.md](protocol.md).

The recorded launch sequence used two disjoint arrays with a dependency, maintaining at most two active experimental jobs. The two initial smoke jobs were separately bounded engineering checks. The historical position diagnostic was a tiny CPU job.

Collect each run's `config.json`, `run_metadata.json`, `trainer_history.json`, `test_metrics.json`, `test_predictions.npz` and `gpu_reload_check.json`. Also copy the numerically last retained checkpoint's `trainer_state.json` to the collected run directory as `trainer_state.json`. Trainer saves at the 30-step intervals and also at the final step 250; the summary includes that final checkpoint when verifying validation selection. Keep the full selected/final models and tokenizer on Nibi under the original ignored `runs/` paths.

Then, from the local repository root:

```bash
python scripts/summarize_baselines.py \
  --run-dir runs/2026-10-19 \
  --baseline-dir runs/2026-10-19/sequence-baselines-verified \
  --output-dir documents/experiments-2026-10-19
```

The summary checks prediction label order and saved checkpoint selection, recalculates test metrics from stored probabilities/logits, and bootstraps the 64 test groups together across models. No test-based model selection or threshold tuning occurs. Its checkpoint-step reconstruction intentionally matches this fixed experiment schedule; it is not a generic analysis tool for arbitrary training settings.

The simple sequence baseline exports standardization statistics and logistic coefficients to `kmer_model.npz`. Features are ordered by k=1,2,3,4 and then Cartesian product order over A,C,G,T. Apply `(x-mean)/scale`, then the stored linear coefficients/intercept and sigmoid. The exported parameters were replayed against saved validation/test probabilities; a deterministic rerun reproduced the original metrics exactly.

Local CPU and Nibi package versions differ. The sequence baseline records its own NumPy/scikit-learn versions; each Nibi run records its Python and training-package versions. Interpret reproducibility against those recorded environments rather than assuming an exact installation of every declared dependency was tested.
