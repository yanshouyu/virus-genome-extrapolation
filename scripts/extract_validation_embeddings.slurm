#!/bin/bash
#SBATCH --job-name=baculo-embeddings
#SBATCH --account=def-aucoin_gpu
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1%2
#SBATCH --output=runs/2026-10-20/extract-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load python
module load arrow/23.0.1
source "$HOME/projects/def-aucoin/shouyu/venvs/huggingface/bin/activate"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
names=(ntv1 ntv2)
name="${names[$SLURM_ARRAY_TASK_ID]}"
python scripts/extract_validation_embeddings.py --model "$name" --output "runs/2026-10-20/embeddings/$name"
