#!/bin/bash
#SBATCH --job-name=adapt-source-check
#SBATCH --account=def-aucoin
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=runs/2026-10-21/source-check-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load python
module load arrow/23.0.1
source "$HOME/projects/def-aucoin/shouyu/venvs/huggingface/bin/activate"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
python scripts/check_adaptation_source_fidelity.py
