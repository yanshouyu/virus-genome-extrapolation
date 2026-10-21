"""Resolve and download pinned source checkpoints before GPU jobs."""
import json
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
out=Path('runs/2026-10-21');out.mkdir(parents=True,exist_ok=True)
result={}
for size in ['100m','500m']:
    name='InstaDeepAI/nucleotide-transformer-v2-'+size+'-multi-species'
    revision='06615c1660c892fc199840c18123f8385b3542a8' if size=='500m' else HfApi().model_info(name).sha
    snapshot_download(name,revision=revision,allow_patterns=['*.json','*.py','*.txt','*.safetensors'])
    result[size]=dict(source=name,revision=revision)
(out/'sources.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
