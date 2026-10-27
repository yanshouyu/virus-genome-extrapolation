# Reproducibility and evidence index

This is a portable reporting package assembled from completed evidence on October 26, 2026. It contains no new training, feature extraction or test prediction. Scientific source state before assembly: commit `9b67df4`; adaptation interpretation: `f5c94dd`; retention implementation/freeze: `4bf23e2` and `6a40782`.

## Verified results

- [Adaptation independent verification](evidence/adaptation/analysis-verification.json): 14 scientific runs, 59 readouts, replay tolerance and intervals.
- [Adaptation server verification](evidence/adaptation/server-verification.json) and [analysis manifest](evidence/adaptation/analysis-manifest.json).
- [Retention server verification](evidence/retention/server-verification.json), [local independent verification](evidence/retention/local-verification.json), and [fresh-process replay](evidence/retention/fresh-replay.json): 126 readouts and all statistical bounds.
- [Retention server manifest](evidence/retention/server-manifest.json), [dataset audit](evidence/retention/dataset-audit.json), [pre-test freeze](evidence/retention/pre-test-freeze-verification.json), and [frozen settings](evidence/retention/frozen-settings-before-test.json).
- [Figures and all result tables](assets.md).

## Protocol, execution and costs

- [Adaptation protocol](evidence/adaptation/protocol.md), [execution snapshot](evidence/adaptation/execution.txt), and [complete job accounting](evidence/adaptation/jobs-completed.psv).
- [Retention protocol](evidence/retention/protocol.md), [execution snapshot](evidence/retention/execution.txt), [cost breakdown](evidence/retention/compute-cost.md), and [UTC accounting](evidence/retention/jobs.psv).
- [Dataset repair](evidence/audits/dataset-repair.txt), [CDS annotation audit](evidence/audits/cds-audit.txt), and [sampling audit](evidence/audits/sampling-audit.json). These are historical audits; their test-coordinate inspection does not imply new test predictions.
- Original detailed interpretations: [adaptation](evidence/adaptation/interpretation.txt) and [retention](evidence/retention/interpretation.txt). Plain-text snapshots preserve their original references to the full repository without treating those paths as portable package links.

## Reproduction boundaries

The repository is `/home/s34yan/projects/260308_virus-genome-extrapolation` locally. Nibi's study root is `/project/6002642/shouyu/virus-genome-extrapolation`; large states and feature caches remain there under `runs/2026-10-24` and `runs/2026-10-25-retention`. Original manifests retain exact remote paths and hashes. They describe full scientific inputs, not paths guaranteed to exist within this portable reporting subset.

Saved numerical summaries are in `tables/`. Full per-window probabilities, bootstrap weights and supporting evidence remain in the repository's `documents/retention-2026-10-25/results/`; adaptation raw readout evidence remains in `documents/adaptation-2026-10-24/results/`. Large weights, features, raw datasets and those prediction arrays are intentionally omitted from this review package. This package is an evidence index and source snapshot, not a self-contained rerun environment.

From the full repository with the original saved inputs and existing analysis environment, the completed adaptation interpretation is reproduced by `scripts/interpret_adaptation_duration.py`; retention local replay uses `scripts/verify_nt_retention_local.py`. Read their input paths and existing execution records before running them. Neither is needed merely to open this report. Scripts for original preparation/extraction/evaluation are included for provenance, and are not rerun by this assembly.

Selected source snapshots:

- [interpret_adaptation_duration.py](sources/scripts/interpret_adaptation_duration.py)
- [summarize_adaptation_duration.py](sources/scripts/summarize_adaptation_duration.py)
- [prepare_nt_retention.py](sources/scripts/prepare_nt_retention.py)
- [extract_nt_retention.py](sources/scripts/extract_nt_retention.py)
- [select_nt_retention.py](sources/scripts/select_nt_retention.py)
- [evaluate_nt_retention.py](sources/scripts/evaluate_nt_retention.py)
- [replay_nt_retention.py](sources/scripts/replay_nt_retention.py)
- [finalize_nt_retention.py](sources/scripts/finalize_nt_retention.py)
- [plot_nt_retention.py](sources/scripts/plot_nt_retention.py)
- [verify_nt_retention_local.py](sources/scripts/verify_nt_retention_local.py)
- [collect_nt_retention_accounting.sh](sources/scripts/collect_nt_retention_accounting.sh)
- [summarize_nt_retention_cost.py](sources/scripts/summarize_nt_retention_cost.py)
- [assemble_final_report.py](sources/scripts/assemble_final_report.py)

[Statistical helpers](sources/src/baculo_adapt/retention.py) · [Focused statistical tests](sources/tests/test_retention.py) · [Project dependencies](sources/pyproject.toml)

## Package integrity

[Manifest](package-manifest.json) records every copied artifact, original relative path and SHA-256. [Assembly verification](assembly-verification.json) records local link, figure and key-result checks. Report rendering uses the existing markdown-it environment; no package installation is required.

[Return to final report](report.md)
