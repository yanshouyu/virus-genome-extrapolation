#!/bin/bash
#SBATCH --job-name=nt-duration-report
#SBATCH --account=def-aucoin
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=runs/2026-10-24/report-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load python
module load arrow/23.0.1
source "$HOME/projects/def-aucoin/shouyu/venvs/huggingface/bin/activate"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python scripts/finalize_adaptation_duration.py
TZ=UTC sacct -j 24919204,24919211,24919261,24919263,24919264,24919265,24919266,24919267,24919268,24919269,"$SLURM_JOB_ID" --format=JobID,State,Start,End,Elapsed,ElapsedRaw,MaxRSS,ExitCode,AllocTRES --parsable2 > documents/adaptation-2026-10-24/results/jobs.psv
