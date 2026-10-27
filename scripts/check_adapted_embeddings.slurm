#!/bin/bash
#SBATCH --job-name=adapt-fresh-replay
#SBATCH --account=def-aucoin_gpu
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=runs/2026-10-21/replay-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load python
module load arrow/23.0.1
source "$HOME/projects/def-aucoin/shouyu/venvs/huggingface/bin/activate"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
python scripts/check_adapted_embeddings.py --size "${1:?Pass size}"
