#!/bin/bash
#SBATCH --job-name=baculo-probe
#SBATCH --account=def-aucoin_gpu
#SBATCH --time=0-07:00
#SBATCH --mem=16G
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4

# Submit from the repository root. SLURM keeps that directory as the workdir.
# Example: sbatch scripts/run_probe.slurm --organism virus --epochs 50
# Set PROBE_ENV to the prepared environment directory if it differs below.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the repository root.}"
[[ -f pyproject.toml ]] || { echo 'Submit from the repository root.' >&2; exit 1; }
module load python
module load arrow/23.0.1
PROBE_ENV="${PROBE_ENV:-$HOME/projects/def-aucoin/shouyu/venvs/huggingface}"
[[ -f "$PROBE_ENV/bin/activate" ]] || { echo "Missing environment: $PROBE_ENV" >&2; exit 1; }
source "$PROBE_ENV/bin/activate"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python scripts/audit_dataset.py
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
echo "Job $SLURM_JOB_ID on $(hostname)"
if [[ $# -eq 0 ]]; then
    set -- --organism virus --epochs 50
fi
python -m baculo_tss_probe.train "$@"
echo 'Training and final evaluation completed.'
