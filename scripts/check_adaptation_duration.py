"""Fresh-process replay of all predeclared budget checkpoint embeddings."""
import argparse, json, tempfile
from pathlib import Path
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from baculo_tss_probe.models import custom_nt_classes, _normalize_backbone_config_fields
from baculo_adapt.duration import inject_lora, load_training_state
from run_adaptation_duration import digest, frozen_digest
from run_controlled_adaptation import extract


def main():
    p=argparse.ArgumentParser(); p.add_argument('--size',choices=['100m','500m'],required=True)
    p.add_argument('--method',choices=['full','lora'],default='full')
    p.add_argument('--root',default='runs/2026-10-24/scientific'); p.add_argument('--output',required=True)
    args=p.parse_args(); root=Path(args.root)
    assert not Path(args.output).exists()
    source=json.loads(Path('runs/2026-10-21/sources.json').read_text())[args.size]
    cfg=AutoConfig.from_pretrained(source['source'],revision=source['revision'],trust_remote_code=True,local_files_only=True)
    _normalize_backbone_config_fields(cfg); cls,_=custom_nt_classes(source['source'],source['revision'],True)
    model=cls.from_pretrained(source['source'],revision=source['revision'],config=cfg,local_files_only=True).cuda()
    if args.method=='lora': inject_lora(model,8,16)
    tokenizer=AutoTokenizer.from_pretrained(source['source'],revision=source['revision'],trust_remote_code=True,local_files_only=True)
    result={}
    for condition in ['baculo','general']:
        for seed in ([42,43,44] if args.method=='full' else [42]):
            name=f'{args.size}-{args.method}-{condition}-seed{seed}'; run=root/name
            assert (run/'COMPLETE').exists()
            meta=json.loads((run/'metadata.json').read_text())
            if args.method=='lora': assert frozen_digest(model)==meta['frozen_parameters_sha256']
            checks={}
            for budget in [512,1024,2048,4096]:
                location=run/f'budget-{budget}'
                assert digest(location/'selected_state.pt')==meta['budgets'][str(budget)]['state_sha256']
                state=torch.load(location/'selected_state.pt',map_location='cpu',weights_only=True)
                load_training_state(model,state,args.method); del state
                with tempfile.TemporaryDirectory(prefix='duration-replay-',dir=run) as temp:
                    extract(model,tokenizer,Path(temp),'replay')
                    checks[str(budget)]={}
                    for split in ['train','validation']:
                        with np.load(Path(temp)/f'replay_{split}.npz') as actual,np.load(location/f'after_{split}.npz') as expected:
                            for key in ['names','groups','labels']: assert np.array_equal(actual[key],expected[key])
                            for pooling in ['cls','mean']:
                                delta=float(np.abs(actual[pooling]-expected[pooling]).max())
                                assert np.allclose(actual[pooling],expected[pooling],rtol=1e-5,atol=1e-6),(name,budget,split,pooling,delta)
                                checks[str(budget)][split+'_'+pooling]=delta
                print(name,budget,checks[str(budget)],flush=True)
            result[name]=dict(selected_states_exact=True,embedding_replay_max_abs=checks)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main()
