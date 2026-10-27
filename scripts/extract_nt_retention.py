"""Frozen NTv2 500M benchmark embedding extraction; no optimization or AcMNPV access."""
import argparse, hashlib, importlib.metadata, json, math, shutil, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoConfig,AutoTokenizer
from baculo_tss_probe.models import custom_nt_classes,_normalize_backbone_config_fields
from baculo_adapt.masking import check_token_budget
ROOT=Path('runs/2026-10-25-retention')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def model_load(name):
 source=json.loads(Path('runs/2026-10-21/sources.json').read_text())['500m']
 cfg=AutoConfig.from_pretrained(source['source'],revision=source['revision'],trust_remote_code=True,local_files_only=True);_normalize_backbone_config_fields(cfg)
 cls,_=custom_nt_classes(source['source'],source['revision'],True)
 model=cls.from_pretrained(source['source'],revision=source['revision'],config=cfg,local_files_only=True)
 state_hash=None
 if name!='original':
  run=Path('runs/2026-10-24/scientific')/name/'budget-4096';v=json.loads((run/'selection.json').read_text());state_hash=sha(run/'selected_state.pt');assert state_hash==v['state_sha256']
  state=torch.load(run/'selected_state.pt',map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True)
  assert all(torch.equal(model.state_dict()[k],v) for k,v in state.items());del state
 model.requires_grad_(False);model.eval();model.cuda()
 tok=AutoTokenizer.from_pretrained(source['source'],revision=source['revision'],trust_remote_code=True,local_files_only=True)
 return model,tok,source,state_hash

def main():
 p=argparse.ArgumentParser();p.add_argument('--backbone',default='original');p.add_argument('--split',choices=['train','test','both'],default='train');p.add_argument('--smoke',action='store_true');p.add_argument('--batch-size',type=int,default=64);a=p.parse_args()
 assert (ROOT/'dataset/COMPLETE').exists()
 if a.split in ['test','both']:
  assert (ROOT/'selection/COMPLETE').exists(),'Freeze all18 readout settings before test inference'
 started=time.time();model,tok,source,state_hash=model_load(a.backbone)
 layers=[math.ceil(model.config.num_hidden_layers*v/4) for v in [1,2,3,4]]
 out=ROOT/('smoke' if a.smoke else 'embeddings')/a.backbone;out.mkdir(parents=True,exist_ok=True)
 splits=['train','test'] if a.split=='both' else [a.split]
 for split in splits:
  assert not (out/(split+'-COMPLETE')).exists()
  sequences=[json.loads(v) for v in (ROOT/'dataset'/(split+'-sequences.jsonl')).read_text().splitlines()]
  if a.smoke:
   assert split=='train';ix=np.random.default_rng(42).choice(len(sequences),min(1024,len(sequences)),replace=False);sequences=[sequences[i] for i in ix]
  n=len(sequences);dim=model.config.hidden_size
  dest=np.lib.format.open_memmap(out/(split+'.npy'),mode='w+',dtype=np.float32,shape=(n,4,dim))
  torch.cuda.reset_peak_memory_stats();begin=time.time();tokens_count=0;repeat=None;max_tokens=0
  for start in range(0,n,a.batch_size):
   tokens=tok(sequences[start:start+a.batch_size],padding=True,truncation=False,return_tensors='pt')
   check_token_budget(tokens['input_ids'],tokens['attention_mask'],model.config,tok)
   mask=tokens['attention_mask'].bool()&~torch.isin(tokens['input_ids'],torch.tensor(tok.all_special_ids));assert (mask.sum(1)>0).all()
   max_tokens=max(max_tokens,int(tokens['attention_mask'].sum(1).max()));tokens_count+=int(tokens['attention_mask'].sum())
   inputs={k:v.cuda() for k,v in tokens.items() if k in ['input_ids','attention_mask']};mask=mask.cuda()
   def forward():
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
     result=model.esm(**inputs,output_hidden_states=True)
     return torch.stack([(result.hidden_states[layer].float()*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True) for layer in layers],dim=1)
   vectors=forward()
   if start==0:
    repeat=float((vectors-forward()).abs().max());assert repeat==0
   dest[start:start+len(vectors)]=vectors.cpu().numpy()
   if start%(a.batch_size*100)==0:print(json.dumps(dict(backbone=a.backbone,split=split,completed=start+len(vectors),total=n,seconds=time.time()-begin)),flush=True)
  dest.flush();del dest;torch.cuda.synchronize()
  meta=dict(backbone=a.backbone,split=split,smoke=a.smoke,rows=n,hidden_size=dim,layers=layers,batch_size=a.batch_size,source=source,state_sha256=state_hash,repeat_max_abs=repeat,seconds=time.time()-begin,total_seconds=time.time()-started,tokens=tokens_count,max_tokens=max_tokens,gpu=torch.cuda.get_device_name(),peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,peak_reserved_gb=torch.cuda.max_memory_reserved()/1e9,output_sha256=sha(out/(split+'.npy')),dataset_sha256=sha(ROOT/'dataset'/(split+'-sequences.jsonl')),protocol_sha256=sha('documents/retention-2026-10-25/protocol.md'),source_sha256={str(q):sha(q) for q in [Path(__file__),Path('src/baculo_tss_probe/models.py')]},packages={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy']},precision='FP32 weights; BF16 autocast; FP32 mean pooling')
  write(out/(split+'-metadata.json'),meta);shutil.copy2(__file__,out/(split+'-source.py'));(out/(split+'-COMPLETE')).write_text('Frozen extraction, exact same-batch repeat, no optimization.\n');print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
