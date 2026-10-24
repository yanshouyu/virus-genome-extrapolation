#!/bin/bash
#SBATCH --job-name=freeze-adapt-readouts
#SBATCH --account=def-aucoin
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=runs/2026-10-21/freeze-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load python
module load arrow/23.0.1
source "$HOME/projects/def-aucoin/shouyu/venvs/huggingface/bin/activate"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
python scripts/freeze_adaptation_readouts.py
