#!/bin/bash
#SBATCH --job-name=baculo-adapt-array
#SBATCH --account=def-aucoin_gpu
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=runs/2026-10-21/adapt-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
size="${ADAPT_SIZE:?}"
index="${SLURM_ARRAY_TASK_ID:?}"
seeds=(42 43 44)
seed="${seeds[$((index / 2))]}"
if ((index % 2 == 0)); then condition=baculo; else condition=general; fi
smoke_tag="${ADAPT_SMOKE_TAG:--v2}"
[[ -f "runs/2026-10-21/smoke-$size$smoke_tag/COMPLETE" ]]
bash scripts/run_controlled_adaptation.slurm --size "$size" --condition "$condition" --seed "$seed" --microbatch "${ADAPT_MICROBATCH:-4}" --output "runs/2026-10-21/scientific/$size-$condition-seed$seed"
