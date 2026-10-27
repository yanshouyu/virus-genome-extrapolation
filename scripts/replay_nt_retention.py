"""Fresh-process replay of fixed batches from all seven completed feature caches."""
import gc,json,math,time
from pathlib import Path
import numpy as np
import torch
from extract_nt_retention import ROOT,model_load,sha,write
from baculo_adapt.masking import check_token_budget
NAMES=['original']+[f'500m-full-{c}-seed{s}' for c in ['baculo','general'] for s in [42,43,44]]
started=time.time();checks={}
for name in NAMES:
 model,tok,source,state_hash=model_load(name);layers=[math.ceil(model.config.num_hidden_layers*v/4) for v in [1,2,3,4]];checks[name]={}
 for split in ['train','test']:
  root=ROOT/'embeddings'/name;assert (root/(split+'-COMPLETE')).exists()
  with (ROOT/'dataset'/(split+'-sequences.jsonl')).open() as f:seq=[json.loads(next(f)) for _ in range(64)]
  tokens=tok(seq,padding=True,truncation=False,return_tensors='pt');check_token_budget(tokens['input_ids'],tokens['attention_mask'],model.config,tok)
  mask=(tokens['attention_mask'].bool()&~torch.isin(tokens['input_ids'],torch.tensor(tok.all_special_ids))).cuda();inputs={k:v.cuda() for k,v in tokens.items() if k in ['input_ids','attention_mask']}
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   h=model.esm(**inputs,output_hidden_states=True).hidden_states
   values=torch.stack([(h[layer].float()*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True) for layer in layers],dim=1).cpu().numpy()
  cached=np.load(root/(split+'.npy'),mmap_mode='r')[:64];delta=float(abs(values-cached).max());assert delta==0,(name,split,delta)
  checks[name][split]=dict(rows=64,all_four_layers_max_abs=delta,state_sha256=state_hash)
  del h,values,cached,inputs,mask,tokens
 del model,tok;gc.collect();torch.cuda.empty_cache();print(name,checks[name],flush=True)
write(ROOT/'fresh-replay.json',dict(backbones=checks,max_abs=0.0,source_sha256={str(p):sha(p) for p in [Path(__file__),Path('scripts/extract_nt_retention.py')]},seconds=time.time()-started))
(ROOT/'REPLAY_COMPLETE').write_text('All7 backbones/train+test first64, all4 layers, fresh process exact.\n')
