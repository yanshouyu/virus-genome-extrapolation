#!/bin/bash
#SBATCH --job-name=nt-duration-smoke
#SBATCH --account=def-aucoin_gpu
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=runs/2026-10-24/smoke-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
size="${DURATION_SIZE:?}"
bash scripts/run_adaptation_duration.slurm --size "$size" --method full --smoke --output "runs/2026-10-24/smoke-$size-full"
if [[ "$size" = 100m ]]; then
 bash scripts/run_adaptation_duration.slurm --size 100m --method lora --smoke --output runs/2026-10-24/smoke-100m-lora
fi
