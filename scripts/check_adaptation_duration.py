"""Fresh-process replay of all predeclared budget checkpoint embeddings."""
import argparse, json, tempfile
from pathlib import Path
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from baculo_tss_probe.models import custom_nt_classes, _normalize_backbone_config_fields
from baculo_adapt.duration import inject_lora, load_training_state
from baculo_adapt.audit import independent_audit_rows
from run_adaptation_duration import digest, frozen_digest
from run_controlled_adaptation import extract, read_rows
from baculo_adapt.masking import mask_tokens, check_token_budget
from collections import Counter
import hashlib


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
    corpus=Path('runs/2026-10-21/corpus')
    audit=independent_audit_rows(read_rows(corpus/'baculo_validation.jsonl'),read_rows(corpus/'mlm_validation.jsonl'))
    tokens=tokenizer([r['sequence'] for r in audit],padding=True,truncation=False,return_tensors='pt')
    check_token_budget(tokens['input_ids'],tokens['attention_mask'],cfg,tokenizer)
    masked,labels=mask_tokens(tokens['input_ids'],tokenizer,torch.Generator().manual_seed(2718))
    def audit_loss():
        model.eval(); total=0.; count=0
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for start in range(0,len(audit),16):
                target=labels[start:start+16].cuda(); n=int((target!=-100).sum())
                loss=model(input_ids=masked[start:start+16].cuda(),attention_mask=tokens['attention_mask'][start:start+16].cuda(),labels=target).loss
                assert torch.isfinite(loss); total+=float(loss)*n; count+=n
        return total/count
    original_audit_loss=audit_loss()
    result={'_verification':dict(source_sha256={str(p):digest(p) for p in [Path(__file__),Path('src/baculo_adapt/duration.py'),Path('src/baculo_adapt/audit.py')]},
        amendment_sha256=digest('documents/adaptation-2026-10-24/verification-amendment.md'),
        audit_panel=[{k:v for k,v in row.items() if k!='sequence'} for row in audit],
        audit_panel_sha256=hashlib.sha256(json.dumps(audit,sort_keys=True).encode()).hexdigest(),
        audit_mask_sha256=hashlib.sha256(masked.numpy().tobytes()+labels.numpy().tobytes()).hexdigest(),
        audit_groups=dict(Counter(row['group'] for row in audit)),original_audit_mlm_loss=original_audit_loss)}
    for condition in ['baculo','general']:
        for seed in ([42,43,44] if args.method=='full' else [42]):
            name=f'{args.size}-{args.method}-{condition}-seed{seed}'; run=root/name
            assert (run/'COMPLETE').exists()
            meta=json.loads((run/'metadata.json').read_text())
            if args.method=='lora': assert frozen_digest(model)==meta['frozen_parameters_sha256']
            checks={}; audit_losses={}
            for budget in [512,1024,2048,4096]:
                location=run/f'budget-{budget}'
                assert digest(location/'selected_state.pt')==meta['budgets'][str(budget)]['state_sha256']
                state=torch.load(location/'selected_state.pt',map_location='cpu',weights_only=True)
                load_training_state(model,state,args.method); del state
                audit_losses[str(budget)]=audit_loss()
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
            result[name]=dict(selected_states_exact=True,embedding_replay_max_abs=checks,independent_audit_mlm_losses=audit_losses)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main()
