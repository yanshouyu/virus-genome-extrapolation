#!/bin/bash
#SBATCH --account=def-aucoin_gpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=h100:1
#SBATCH --output=runs/2026-10-25-retention/array-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load python
module load arrow/23.0.1
source "$HOME/projects/def-aucoin/shouyu/venvs/huggingface/bin/activate"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
names=(500m-full-baculo-seed42 500m-full-baculo-seed43 500m-full-baculo-seed44 500m-full-general-seed42 500m-full-general-seed43 500m-full-general-seed44)
if [[ "$1" == test ]];then names=(original "${names[@]}");fi
python scripts/extract_nt_retention.py --backbone "${names[$SLURM_ARRAY_TASK_ID]}" --split "$1"
