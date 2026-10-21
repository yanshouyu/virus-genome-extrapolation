set -euo pipefail
module load python
module load arrow/23.0.1
source "$HOME/projects/def-aucoin/shouyu/venvs/huggingface/bin/activate"
export HF_HUB_DISABLE_XET=1
# File transfer/cache preparation only; no model inference or training on login.
python scripts/cache_adaptation_sources.py
