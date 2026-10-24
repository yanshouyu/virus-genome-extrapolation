#!/bin/bash
# Inspect both successful smoke metadata files before calling this launcher.
set -euo pipefail
smoke_tag="${ADAPT_SMOKE_TAG:--v2}"
export ADAPT_SMOKE_TAG="$smoke_tag"
export ADAPT_SIZES="${ADAPT_SIZES:-100m 500m}"
python - <<'PY'
import json,os
from pathlib import Path
for size in os.environ['ADAPT_SIZES'].split():
    assert size in ['100m','500m']
    root=Path('runs/2026-10-21/smoke-'+size+os.environ['ADAPT_SMOKE_TAG'])
    assert (root/'COMPLETE').is_file(),root
    m=json.loads((root/'metadata.json').read_text())
    assert m['saved_state_exact_replay'] is True
    assert m['initial_backbone_reference_exact'] is True
    assert m['embedding_parameter_max_change']>0
    assert m['max_gpu_reserved_bytes']<.95*m['gpu_total_memory_bytes']
    print(size,'memory fraction',m['max_gpu_reserved_bytes']/m['gpu_total_memory_bytes'],'5-step training seconds',m['training_seconds'])
PY
for size in $ADAPT_SIZES; do
    if [[ "$size" == 100m ]]; then gpu=h100_2g.20gb; else gpu=h100_3g.40gb; fi
    microbatch=$(python -c "import json; print(json.load(open('runs/2026-10-21/smoke-$size$smoke_tag/metadata.json'))['args']['microbatch'])")
    sbatch --parsable --array=0-5%1 --time="${ADAPT_WALLTIME:-02:00:00}" --gpus="${ADAPT_GPU:-$gpu}:1" --export="ALL,ADAPT_SIZE=$size,ADAPT_MICROBATCH=$microbatch,ADAPT_SMOKE_TAG=$smoke_tag" scripts/run_controlled_adaptation_array.slurm
done
