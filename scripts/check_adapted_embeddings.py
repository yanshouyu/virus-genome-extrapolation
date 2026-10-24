"""Fresh-process replay of selected adapted encoders against stored embeddings."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoConfig,AutoTokenizer
from baculo_tss_probe.models import custom_nt_classes,_normalize_backbone_config_fields

p=argparse.ArgumentParser();p.add_argument('--size',choices=['100m','500m'],required=True);args=p.parse_args()
root=Path('runs/2026-10-21');source=json.loads((root/'sources.json').read_text())[args.size]
cfg=AutoConfig.from_pretrained(source['source'],revision=source['revision'],trust_remote_code=True,local_files_only=True)
_normalize_backbone_config_fields(cfg);cls,_=custom_nt_classes(source['source'],source['revision'],True)
model=cls.from_pretrained(source['source'],revision=source['revision'],config=cfg,local_files_only=True).cuda().eval()
tokenizer=AutoTokenizer.from_pretrained(source['source'],revision=source['revision'],trust_remote_code=True,local_files_only=True)
results={}
for condition in ['baculo','general']:
    for seed in [42,43,44]:
        name=f'{args.size}-{condition}-seed{seed}';run=root/'scientific'/name
        assert (run/'COMPLETE').is_file()
        meta=json.loads((run/'metadata.json').read_text())
        assert hashlib.sha256((run/'selected_state.pt').read_bytes()).hexdigest()==meta['state_sha256']
        state=torch.load(run/'selected_state.pt',map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True)
        assert all(torch.equal(value.detach().cpu(),state[key]) for key,value in model.state_dict().items());del state
        checks={}
        for split in ['train','validation']:
            rows=list(csv.DictReader(Path('data/processed/clean_v2',split+'.csv').open()))
            arrays={'cls':[],'mean':[]}
            with torch.no_grad():
                for start in range(0,len(rows),32):
                    tokens=tokenizer([r['sequence'] for r in rows[start:start+32]],padding=True,truncation=False,return_tensors='pt')
                    special=torch.isin(tokens['input_ids'],torch.tensor(tokenizer.all_special_ids))
                    inputs={k:v.cuda() for k,v in tokens.items() if k in ['input_ids','attention_mask']}
                    h=model.esm(**inputs).last_hidden_state
                    mask=(tokens['attention_mask'].bool()&~special).cuda()
                    pooled=(h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
                    arrays['cls'].append(h[:,0].cpu().numpy());arrays['mean'].append(pooled.cpu().numpy())
            with np.load(run/('after_'+split+'.npz')) as original:
                assert np.array_equal(original['names'],[r['name'] for r in rows])
                checks[split]={}
                for pooling in arrays:
                    replay=np.concatenate(arrays[pooling]);expected=original[pooling]
                    delta=float(np.max(np.abs(replay-expected)));assert np.allclose(replay,expected,rtol=1e-5,atol=1e-6),(name,split,pooling,delta)
                    checks[split][pooling]=delta
        results[name]=dict(selected_state_exact=True,embedding_replay_max_abs=checks)
        print(name,checks,flush=True)
(root/('fresh-replay-'+args.size+'.json')).write_text(json.dumps(results,indent=2)+'\n')
