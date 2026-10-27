#!/bin/bash
#SBATCH --job-name=baculo-readouts
#SBATCH --account=def-aucoin
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=runs/2026-10-20/readouts-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load python
module load arrow/23.0.1
source "$HOME/projects/def-aucoin/shouyu/venvs/huggingface/bin/activate"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
python scripts/analyze_validation_readouts.py --embedding-root runs/2026-10-20/embeddings --output runs/2026-10-20/readouts
