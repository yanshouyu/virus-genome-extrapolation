#!/bin/bash
#SBATCH --account=def-aucoin
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --output=runs/2026-10-25-retention/release-%j.log
set -euo pipefail
# Submitted afterok the original GPU extraction. The test/replay GPU jobs still
# depend on the complete adapted array, so this preserves at most two GPUs.
scontrol update JobId=25004643 ArrayTaskThrottle=2
scontrol show job 25004643
