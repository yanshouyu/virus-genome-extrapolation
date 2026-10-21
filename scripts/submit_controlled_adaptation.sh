#!/bin/bash
# Submit only after both smoke jobs completed and their results were inspected.
set -euo pipefail
python - <<'PY'
import json
from pathlib import Path
for size in ['100m','500m']:
    root=Path('runs/2026-10-21/smoke-'+size)
    assert (root/'COMPLETE').is_file(),root
    m=json.loads((root/'metadata.json').read_text())
    assert m['saved_state_exact_replay'] is True
    assert m['initial_backbone_reference_exact'] is True
    assert m['embedding_parameter_max_change']>0
    assert m['max_gpu_reserved_bytes']<.95*m['gpu_total_memory_bytes']
    print(size,'memory fraction',m['max_gpu_reserved_bytes']/m['gpu_total_memory_bytes'],'5-step training seconds',m['training_seconds'])
PY
for size in 100m 500m; do
    if [[ "$size" == 100m ]]; then gpu=h100_2g.20gb; else gpu=h100_3g.40gb; fi
    sbatch --parsable --array=0-5%1 --gpus="$gpu:1" --export="ALL,ADAPT_SIZE=$size" scripts/run_controlled_adaptation_array.slurm
done
