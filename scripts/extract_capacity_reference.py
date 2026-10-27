"""Exploratory frozen NTv1 2.5B multispecies capacity reference; no test."""
import hashlib,importlib.metadata,json,time
from pathlib import Path
import torch
from transformers import AutoConfig,AutoModelForMaskedLM,AutoTokenizer
from huggingface_hub import hf_hub_download
from baculo_tss_probe.models import _normalize_backbone_config_fields
from run_controlled_adaptation import extract

started=time.time();root=Path('runs/2026-10-21/capacity-2.5b');root.mkdir(parents=True,exist_ok=False)
source='InstaDeepAI/nucleotide-transformer-2.5b-multi-species';revision='b746b125aacd1b0970c05a32fd71ba726754542e'
cfg=AutoConfig.from_pretrained(source,revision=revision,local_files_only=True);_normalize_backbone_config_fields(cfg)
model=AutoModelForMaskedLM.from_pretrained(source,revision=revision,config=cfg,local_files_only=True).cuda().eval()
model.requires_grad_(False)
tokenizer=AutoTokenizer.from_pretrained(source,revision=revision,local_files_only=True)
index=json.loads(Path(hf_hub_download(source,'pytorch_model.bin.index.json',revision=revision,local_files_only=True)).read_text())['weight_map']
parameters=dict(model.named_parameters());checked=0;shard_hashes={}
for shard_name in sorted(set(index.values())):
    path=Path(hf_hub_download(source,shard_name,revision=revision,local_files_only=True))
    state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    for key,value in parameters.items():
        if index[key]==shard_name:
            assert torch.equal(value.detach().cpu(),state[key]),key
            checked+=1
    hasher=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):hasher.update(chunk)
    shard_hashes[shard_name]=hasher.hexdigest();del state
assert checked==len(parameters)
torch.cuda.reset_peak_memory_stats();replay=extract(model,tokenizer,root,'before')
metadata=dict(source=source,revision=revision,parameters=sum(p.numel() for p in model.parameters()),
    all_source_parameters_exact=True,checked_parameter_tensors=checked,shard_sha256=shard_hashes,
    embedding_replay_max_abs=replay,gpu=torch.cuda.get_device_name(),gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
    max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.time()-started,
    packages={p:importlib.metadata.version(p) for p in ['torch','transformers','numpy']},
    scope='Frozen exploratory reference only; no adaptation and no downstream test access.',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(root/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');(root/'COMPLETE').write_text('Verified source and frozen embeddings.\n')
print(json.dumps(metadata,indent=2))
