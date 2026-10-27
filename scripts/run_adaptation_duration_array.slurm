#!/bin/bash
#SBATCH --job-name=nt-duration-array
#SBATCH --account=def-aucoin_gpu
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=runs/2026-10-24/array-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
size="${DURATION_SIZE:?}"
method="${DURATION_METHOD:-full}"
index="${SLURM_ARRAY_TASK_ID:?}"
seeds=(42 43 44)
seed="${seeds[$((index / 2))]}"
if ((index % 2 == 0)); then condition=baculo; else condition=general; fi
[[ -f "runs/2026-10-24/smoke-$size-$method/COMPLETE" ]]
bash scripts/run_adaptation_duration.slurm --size "$size" --method "$method" --condition "$condition" --seed "$seed" --output "runs/2026-10-24/scientific/$size-$method-$condition-seed$seed"
