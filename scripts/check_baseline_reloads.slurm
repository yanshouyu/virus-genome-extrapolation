#!/bin/bash
#SBATCH --job-name=baculo-reload-check
#SBATCH --account=def-aucoin_gpu
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --output=runs/2026-10-19/reload-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load python arrow/23.0.1
source "${PROBE_ENV:-$HOME/projects/def-aucoin/shouyu/venvs/huggingface}/bin/activate"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export HF_HUB_DISABLE_XET=1
for model in ntv1 ntv2; do
    for seed in 42 43 44; do
        python scripts/check_probe_reload.py "${EXPERIMENT_ROOT:-runs/2026-10-19/verified}/${model}-seed${seed}"
    done
done
echo 'All six saved probes reproduced their recorded test logits on CUDA.'
