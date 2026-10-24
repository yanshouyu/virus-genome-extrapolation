"""Verify all loaded MLM parameters against pinned safetensors references on CPU."""
import json
from pathlib import Path
import torch
from transformers import AutoConfig
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from baculo_tss_probe.models import custom_nt_classes,_normalize_backbone_config_fields

torch.set_num_threads(4)
results={}
for size,source in json.loads(Path('runs/2026-10-21/sources.json').read_text()).items():
    cfg=AutoConfig.from_pretrained(source['source'],revision=source['revision'],trust_remote_code=True,local_files_only=True)
    _normalize_backbone_config_fields(cfg);cls,_=custom_nt_classes(source['source'],source['revision'],True)
    model=cls.from_pretrained(source['source'],revision=source['revision'],config=cfg,local_files_only=True)
    reference=hf_hub_download(source['source'],'model.safetensors',revision=source['revision'],local_files_only=True)
    names=[]
    with safe_open(reference,framework='pt',device='cpu') as state:
        for key,value in model.named_parameters():
            assert torch.equal(value.detach(),state.get_tensor(key)),key
            names.append(key)
    results[size]=dict(source=source,all_mlm_parameters_exact=True,unique_parameter_tensors=len(names),includes_lm_head=any(n.startswith('lm_head.') for n in names))
    del model
Path('runs/2026-10-21/source-fidelity.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
