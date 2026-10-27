#!/bin/bash
#SBATCH --job-name=adapt-development
#SBATCH --account=def-aucoin
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=runs/2026-10-21/development-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load python
module load arrow/23.0.1
source "$HOME/projects/def-aucoin/shouyu/venvs/huggingface/bin/activate"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
python - <<'PY'
import hashlib,importlib.metadata,json
from pathlib import Path
sources=['scripts/evaluate_controlled_adaptation.py','scripts/analyze_validation_readouts.py','scripts/run_sequence_baselines.py']
Path('runs/2026-10-21/analysis-environment.json').write_text(json.dumps(dict(packages={k:importlib.metadata.version(k) for k in ['numpy','scikit-learn','scipy']},source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}),indent=2)+'\n')
PY
[[ -f runs/2026-10-21/frozen-readout-selection.json ]]
python scripts/evaluate_controlled_adaptation.py "$@"
