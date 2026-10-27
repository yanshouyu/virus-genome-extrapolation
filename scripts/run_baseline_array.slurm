#!/bin/bash
#SBATCH --job-name=baculo-baselines
#SBATCH --account=def-aucoin_gpu
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-5%2
#SBATCH --output=runs/2026-10-19/baseline-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
index="${SLURM_ARRAY_TASK_ID:?}"
seeds=(42 43 44)
seed="${seeds[$((index % 3))]}"
if ((index < 3)); then
    name=ntv1
    model=InstaDeepAI/nucleotide-transformer-500m-human-ref
else
    name=ntv2
    model=InstaDeepAI/nucleotide-transformer-v2-500m-multi-species
fi
output="${EXPERIMENT_ROOT:-runs/2026-10-19/verified}/${name}-seed${seed}"
[[ ! -e "$output" ]] || { echo "Refusing to overwrite $output" >&2; exit 1; }
bash scripts/run_probe.slurm --organism virus --base-model "$model" --epochs 50 --seed "$seed" --output-dir "$output"
