#!/bin/bash
#SBATCH --job-name=nt-duration-replay
#SBATCH --account=def-aucoin_gpu
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=runs/2026-10-24/replay-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load python
module load arrow/23.0.1
source "$HOME/projects/def-aucoin/shouyu/venvs/huggingface/bin/activate"
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1
python scripts/check_adaptation_duration.py "$@"
