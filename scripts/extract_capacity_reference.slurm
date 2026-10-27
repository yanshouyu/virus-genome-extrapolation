#!/bin/bash
#SBATCH --job-name=nt-2p5b-reference
#SBATCH --account=def-aucoin_gpu
#SBATCH --time=00:20:00
#SBATCH --gpus=h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=runs/2026-10-21/capacity-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load python
module load arrow/23.0.1
source "$HOME/projects/def-aucoin/shouyu/venvs/huggingface/bin/activate"
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
python scripts/extract_capacity_reference.py
