# virus-genome-extrapolation

This began as an ECE657D course project probing frozen Nucleotide Transformer representations on human promoter and AcMNPV TSS classification. The continuation goal is a publishable, reproducible study of generalization.

The current [repair report](documents/repair-2026-10-13/README.md) explains the clean dataset, overlap causes, model compatibility fixes and verification. The [initial assessment](documents/project-assessment.md) records the course baseline and publication plan.

## Current workflow (WSL Ubuntu)

```bash
cd /home/s34yan/projects/260308_virus-genome-extrapolation
conda activate ece657d
PYTHONPATH=src python scripts/audit_dataset.py
PYTHONPATH=src python -m baculo_tss_probe.train --help
```

The active [clean dataset](data/processed/clean_v2/README.md) has 233 training, 77 validation and 78 test samples. Same-strand overlaps and related groups stay together; opposite-strand overlap is allowed. Three source rows with inconsistent TSS labels were excluded and documented. Original course files remain one directory above the versioned export.

Training now selects checkpoints by validation loss and predicts the final test once. New probes use an explicit raw first-token representation for both model versions, so their results form a new baseline. Model revisions, seeds, package versions and dataset metadata are saved with each run.

```bash
# Runs training; use a suitable compute node/environment.
PYTHONPATH=src python -m baculo_tss_probe.train --organism virus --epochs 50 --seed 42
# Prepared Nibi template for manual submission from the repository root:
sbatch scripts/run_probe.slurm --organism virus --epochs 50
```

The local environment differs from the project dependency pins. See the repair report before installing or upgrading dependencies. No local environment changes were made during the repairs.

## Layout

- `data/raw/baculovirus/`: curated source TSS table, accession manifest and local GenBank records.
- `data/interim/`: fixed course positive coordinates and generated FASTA/clustering inputs.
- `data/processed/clean_v2/`: tracked, versioned train/validation/test export and manifest.
- `src/baculo_tss_probe/`: splitting, validation, frozen-encoder classifier and training.
- `scripts/`: extraction, split generation, auditing, compatibility check and Nibi template.
- `tests/`: small regression tests plus opt-in cached pretrained checks.
- `notebooks/`: course preparation and plotting; local `trainer_states/` and `figures/` are ignored.
- `runs/`: ignored training outputs.
- `documents/`: assessment, repair notes and compact verification evidence.

Course work is on `dev`; `main` is older. Domain-adaptation code remains on `feat/domain-adaptation` and has not been merged. The original local `scripts/run_training.slurm` remains untracked; the new template is `scripts/run_probe.slurm`.

## Contributors

Keep changes small and evidence-driven. Preserve the course artifacts when introducing revised experiments. Notebook output stripping:

```bash
pip install nbstripout
nbstripout --install
```
