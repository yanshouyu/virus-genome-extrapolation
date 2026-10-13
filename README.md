# Project orientation

This began as an ECE657D course project testing frozen Nucleotide Transformer representations on human promoter and AcMNPV TSS classification. The continuation goal is a publishable, reproducible study of generalization. Existing results are exploratory; the current split and model comparisons need validation before stronger claims.

Start with [the project assessment and continuation plan](documents/project-assessment.md).

## Layout

- `data/raw/baculovirus/`: accession manifest, curated source TSS table, locally downloaded GenBank records.
- `data/interim/`: fixed positive-window coordinates plus generated FASTA and clustering files.
- `data/processed/`: generated course train/test CSV and FASTA files.
- `src/baculo_tss_probe/`: configuration, data loading, frozen-backbone classifier, training and checkpoint callback.
- `scripts/`: extraction, clustering, splitting and read-only dataset audit.
- `notebooks/`: data preparation and course-result plotting. Local `trainer_states/` and `figures/` are ignored by Git.
- `runs/`: local checkpoints; not a complete record of the HPC experiments.
- `documents/`: assessment, next steps and compact audit evidence.

## Local inspection (WSL Ubuntu)

```bash
cd /home/s34yan/projects/260308_virus-genome-extrapolation
conda activate ece657d
python scripts/audit_dataset.py
PYTHONPATH=src python -m baculo_tss_probe.train --help
```

The existing environment differs from the package pins; see the assessment before installing or upgrading dependencies. The commands above do not train a model. Running training still uses the course test set for periodic evaluation and should not be treated as a publication-ready experiment.

The course work is on `dev`; `main` is older. Domain-adaptation source lives on `feat/domain-adaptation`, not in this checkout. See the assessment before reviving it. The local `scripts/run_training.slurm` is an existing, untracked HPC example; no HPC access is required for this audit.

## Contributors

Keep changes small and evidence-driven. Preserve the course data and outputs when introducing a revised experiment. This repository uses nbstripout to keep committed notebooks clean:

```bash
pip install nbstripout
nbstripout --install
```
