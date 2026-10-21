#!/bin/bash
#SBATCH --job-name=baculo-adapt
#SBATCH --account=def-aucoin_gpu
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=runs/2026-10-21/adapt-%j.log
# Choose --gpus=h100_2g.20gb:1 for 100M or h100_3g.40gb:1 for 500M.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load python
module load arrow/23.0.1
source "$HOME/projects/def-aucoin/shouyu/venvs/huggingface/bin/activate"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1
python scripts/run_controlled_adaptation.py "$@"
