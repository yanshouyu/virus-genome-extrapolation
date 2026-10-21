"""Run one controlled NTv2 continued-pretraining arm; no downstream test access."""
import argparse,csv,hashlib,importlib.metadata,json,math,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoConfig,AutoTokenizer
from baculo_tss_probe.models import custom_nt_classes,_normalize_backbone_config_fields
from baculo_adapt.masking import mask_tokens,check_token_budget


def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def read_rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]

def extract(model,tokenizer,out,prefix):
    """Cache both pooling rules on train/validation, retaining identifiers."""
    model.eval()
    for split in ['train','validation']:
        rows=list(csv.DictReader(Path('data/processed/clean_v2',split+'.csv').open()))
        values={'cls':[],'mean':[]};replay=None
        for start in range(0,len(rows),32):
            tokens=tokenizer([r['sequence'] for r in rows[start:start+32]],padding=True,truncation=False,return_tensors='pt')
            special=torch.isin(tokens['input_ids'],torch.tensor(tokenizer.all_special_ids))
            check_token_budget(tokens['input_ids'],tokens['attention_mask'],model.config,tokenizer)
            inputs={k:v.cuda() for k,v in tokens.items() if k in ['input_ids','attention_mask']}
            with torch.inference_mode():
                h=model.esm(**inputs).last_hidden_state
                mask=(tokens['attention_mask'].bool()&~special).cuda()
                assert (mask.sum(1)>0).all()
                mean=(h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
                if start==0:
                    replay=float((h-model.esm(**inputs).last_hidden_state).abs().max().cpu())
                    assert replay==0
            values['cls'].append(h[:,0].cpu().numpy());values['mean'].append(mean.cpu().numpy())
        np.savez_compressed(out/(prefix+'_'+split+'.npz'),**{k:np.concatenate(v) for k,v in values.items()},
            names=np.array([r['name'] for r in rows]),groups=np.array([r['group'] for r in rows]),labels=np.array([int(r['label']) for r in rows]))
    return replay


def main():
    p=argparse.ArgumentParser();p.add_argument('--size',choices=['100m','500m'],required=True)
    p.add_argument('--condition',choices=['baculo','general'],default='baculo');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--microbatch',type=int,default=4)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    started=time.time();torch.manual_seed(args.seed);np.random.seed(args.seed)
    assert torch.cuda.is_available();assert 32%args.microbatch==0
    source=json.loads(Path('runs/2026-10-21/sources.json').read_text())[args.size]
    cfg=AutoConfig.from_pretrained(source['source'],revision=source['revision'],trust_remote_code=True,local_files_only=True)
    _normalize_backbone_config_fields(cfg)
    cls,_=custom_nt_classes(source['source'],source['revision'],True)
    model=cls.from_pretrained(source['source'],revision=source['revision'],config=cfg,local_files_only=True).cuda()
    tokenizer=AutoTokenizer.from_pretrained(source['source'],revision=source['revision'],trust_remote_code=True,local_files_only=True)
    model.requires_grad_(True)
    if args.smoke:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        weights=hf_hub_download(source['source'],'model.safetensors',revision=source['revision'],local_files_only=True)
        with safe_open(weights,framework='pt',device='cpu') as reference:
            for key,parameter in model.esm.named_parameters():
                assert torch.equal(parameter.detach().cpu(),reference.get_tensor('esm.'+key)),key
    corpus=Path('runs/2026-10-21/corpus')
    train_rows=read_rows(corpus/(args.condition+'_train.jsonl'))
    val_rows=read_rows(corpus/'mlm_validation.jsonl')[:16 if args.smoke else 128]
    def tokenize(rows):
        tokens=tokenizer([r['sequence'] for r in rows],padding=True,truncation=False,return_tensors='pt')
        limit=check_token_budget(tokens['input_ids'],tokens['attention_mask'],cfg,tokenizer)
        assert torch.all(tokens['attention_mask'].sum(1)==257)
        return tokens['input_ids'],tokens['attention_mask'],limit
    train,train_attention,limit=tokenize(train_rows);val,val_attention,_=tokenize(val_rows)
    fixed_generator=torch.Generator().manual_seed(1729)
    val_masked,val_labels=mask_tokens(val,tokenizer,fixed_generator)
    def evaluate():
        model.eval();total=0.;n=0
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for start in range(0,len(val),args.microbatch):
                labels=val_labels[start:start+args.microbatch].cuda();count=int((labels!=-100).sum())
                loss=model(input_ids=val_masked[start:start+args.microbatch].cuda(),attention_mask=val_attention[start:start+args.microbatch].cuda(),labels=labels).loss
                assert torch.isfinite(loss);total+=float(loss)*count;n+=count
        return total/n
    initial=evaluate()
    metadata=dict(args=vars(args),source=source,initial_mlm_loss=initial,token_limit=limit,
        packages={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy']},
        gpu=torch.cuda.get_device_name(),gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,parameters=sum(p.numel() for p in model.parameters()),
        corpus_sha256=hashlib.sha256((corpus/(args.condition+'_train.jsonl')).read_bytes()).hexdigest(),
        eval_sha256=hashlib.sha256((corpus/'mlm_validation.jsonl').read_bytes()).hexdigest(),
        initial_backbone_reference_exact=(True if args.smoke else 'verified by required smoke job'),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/baculo_adapt/masking.py'),Path('src/baculo_tss_probe/models.py')]},
        sampling='uniform windows with replacement; independent seeded CPU generator')
    if not args.smoke:extract(model,tokenizer,out,'before')
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01)
    generator=torch.Generator().manual_seed(args.seed+1000)
    steps=5 if args.smoke else 512;accumulation=32//args.microbatch
    best=float('inf');history=[];masked_count=0;input_count=0
    tracked=model.esm.embeddings.word_embeddings.weight
    initial_weight=tracked.detach().cpu().clone()
    training_started=time.time();torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(args.seed)
    for step in range(1,steps+1):
        model.train();optimizer.zero_grad(set_to_none=True);loss_sum=0.
        for micro in range(accumulation):
            ix=torch.randint(len(train),(args.microbatch,),generator=generator)
            ids,labels=mask_tokens(train[ix],tokenizer,generator)
            masked_count+=int((labels!=-100).sum());input_count+=int(train_attention[ix].sum())
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=model(input_ids=ids.cuda(),attention_mask=train_attention[ix].cuda(),labels=labels.cuda()).loss
            assert torch.isfinite(loss);(loss/accumulation).backward();loss_sum+=float(loss.detach())/accumulation
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        scale=min(step/26.,.5*(1+math.cos(math.pi*max(0,step-26)/(512-26))))
        for group in optimizer.param_groups:group['lr']=1e-5*scale
        optimizer.step()
        if step%16==0:print(json.dumps(dict(step=step,train_loss=loss_sum,elapsed=time.time()-training_started)),flush=True)
        if step%128==0 or step==steps:
            score=evaluate();history.append(dict(step=step,mlm_loss=score,train_loss=loss_sum,gradient_norm=float(grad),seconds=time.time()-training_started))
            if score<best:
                best=score;metadata['selected_step']=step
                torch.save(model.state_dict(),out/'selected_state.pt')
            write(out/'history.json',history);print(json.dumps(history[-1]),flush=True)
    delta=float((tracked.detach().cpu()-initial_weight).abs().max());assert delta>0
    metadata.update(selected_mlm_loss=best,embedding_parameter_max_change=delta,training_seconds=time.time()-training_started,
        masked_tokens=masked_count,input_tokens=input_count,max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),max_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
    del optimizer
    torch.cuda.empty_cache()
    selected=torch.load(out/'selected_state.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(selected,strict=True);del selected
    if not args.smoke:metadata['after_embedding_replay_max_abs']=extract(model,tokenizer,out,'after')
    # Compare reloaded tensors exactly to the saved selected checkpoint.
    saved=torch.load(out/'selected_state.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(v.detach().cpu(),saved[k]) for k,v in model.state_dict().items())
    metadata.update(saved_state_exact_replay=True,total_seconds=time.time()-started,
                    state_sha256=hashlib.sha256((out/'selected_state.pt').read_bytes()).hexdigest())
    write(out/'metadata.json',metadata);(out/'COMPLETE').write_text('Verified training and selected-state reload.\n')
    print(json.dumps(metadata,indent=2),flush=True)

if __name__=='__main__':main()
