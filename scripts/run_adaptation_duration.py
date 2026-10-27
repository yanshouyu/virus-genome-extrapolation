"""Bounded long-schedule MLM training; train/validation probing only."""
import argparse, hashlib, importlib.metadata, json, os, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from baculo_tss_probe.models import custom_nt_classes, _normalize_backbone_config_fields
from baculo_adapt.masking import mask_tokens, check_token_budget
from baculo_adapt.duration import (inject_lora, state_to_save, load_training_state,
                                   learning_rate_scale, plateau_summary)
from run_controlled_adaptation import extract, read_rows, write


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def frozen_digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if not p.requires_grad:
            h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--size', choices=['100m','500m'], required=True)
    p.add_argument('--condition', choices=['baculo','general'], default='baculo')
    p.add_argument('--method', choices=['full','lora'], default='full')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=4096)
    p.add_argument('--lr', type=float)
    p.add_argument('--rank', type=int, default=8); p.add_argument('--alpha', type=float, default=16)
    p.add_argument('--microbatch', type=int, default=16)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--output', required=True)
    args=p.parse_args()
    assert args.steps>=512 and args.steps%256==0
    assert args.microbatch>0 and 32%args.microbatch==0
    if args.lr is None: args.lr=1e-5 if args.method=='full' else 1e-4
    out=Path(args.output); out.mkdir(parents=True, exist_ok=False)
    started=time.time(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    assert torch.cuda.is_available()
    source=json.loads(Path('runs/2026-10-21/sources.json').read_text())[args.size]
    cfg=AutoConfig.from_pretrained(source['source'], revision=source['revision'], trust_remote_code=True, local_files_only=True)
    _normalize_backbone_config_fields(cfg)
    cls,_=custom_nt_classes(source['source'],source['revision'],True)
    model=cls.from_pretrained(source['source'],revision=source['revision'],config=cfg,local_files_only=True)
    ref=hf_hub_download(source['source'],'model.safetensors',revision=source['revision'],local_files_only=True)
    with safe_open(ref,framework='pt',device='cpu') as state:
        for name,param in model.named_parameters():
            assert torch.equal(param.detach(),state.get_tensor(name)),name
    model=model.cuda(); model.requires_grad_(True)
    targets=inject_lora(model,args.rank,args.alpha) if args.method=='lora' else []
    frozen_before=frozen_digest(model) if args.method=='lora' else None
    tokenizer=AutoTokenizer.from_pretrained(source['source'],revision=source['revision'],trust_remote_code=True,local_files_only=True)
    corpus=Path('runs/2026-10-21/corpus')
    report=json.loads((corpus/'corpus-report.json').read_text())
    assert digest(corpus/(args.condition+'_train.jsonl'))==report['files'][args.condition+'_train.jsonl']
    assert report['files']['general_train.jsonl']=='6455320fd95ae86d27d32c80c101b18c9529cc064f252e7972181a947d852e04'
    def tokenize(rows):
        t=tokenizer([r['sequence'] for r in rows],padding=True,truncation=False,return_tensors='pt')
        limit=check_token_budget(t['input_ids'],t['attention_mask'],cfg,tokenizer)
        assert torch.all(t['attention_mask'].sum(1)==257)
        return t['input_ids'],t['attention_mask'],limit
    train,attention,limit=tokenize(read_rows(corpus/(args.condition+'_train.jsonl')))
    val,val_attention,_=tokenize(read_rows(corpus/'mlm_validation.jsonl')[:16 if args.smoke else 128])
    val_masked,val_labels=mask_tokens(val,tokenizer,torch.Generator().manual_seed(1729))
    def evaluate():
        model.eval(); total=0.; n=0
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for start in range(0,len(val),args.microbatch):
                labels=val_labels[start:start+args.microbatch].cuda(); count=int((labels!=-100).sum())
                loss=model(input_ids=val_masked[start:start+args.microbatch].cuda(),attention_mask=val_attention[start:start+args.microbatch].cuda(),labels=labels).loss
                assert torch.isfinite(loss); total+=float(loss)*count; n+=count
        return total/n
    initial=evaluate()
    metadata=dict(args=vars(args),source=source,pretraining_scope='multispecies',initial_mlm_loss=initial,
        all_source_parameters_exact=True,token_limit=limit,lora_targets=targets,
        parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        packages={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy']},
        gpu=torch.cuda.get_device_name(),gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        corpus_sha256=digest(corpus/(args.condition+'_train.jsonl')),eval_sha256=digest(corpus/'mlm_validation.jsonl'),
        source_sha256={str(q):digest(q) for q in [Path(__file__),Path('src/baculo_adapt/duration.py'),Path('src/baculo_adapt/masking.py'),Path('scripts/run_controlled_adaptation.py'),Path('src/baculo_tss_probe/models.py')]},
        protocol_sha256=digest('documents/adaptation-2026-10-24/protocol.md'),budgets={})
    write(out/'metadata-initial.json',metadata)
    if not args.smoke: extract(model,tokenizer,out,'before')
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=args.lr,weight_decay=.01)
    generator=torch.Generator().manual_seed(args.seed+1000)
    steps=5 if args.smoke else args.steps
    budgets=[steps] if args.smoke else [b for b in [512,1024,2048,args.steps] if b<=steps]
    best=float('inf'); history=[]; masked_count=0; input_count=0
    initial_trainable={k:p.detach().cpu().clone() for k,p in model.named_parameters() if p.requires_grad and (args.method=='lora' or k=='esm.embeddings.word_embeddings.weight')}
    training_started=time.time(); torch.cuda.reset_peak_memory_stats(); torch.manual_seed(args.seed)
    for step in range(1,steps+1):
        model.train(); optimizer.zero_grad(set_to_none=True); loss_sum=0.
        for micro in range(32//args.microbatch):
            ix=torch.randint(len(train),(args.microbatch,),generator=generator)
            ids,labels=mask_tokens(train[ix],tokenizer,generator)
            masked_count+=int((labels!=-100).sum()); input_count+=int(attention[ix].sum())
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=model(input_ids=ids.cuda(),attention_mask=attention[ix].cuda(),labels=labels.cuda()).loss
            assert torch.isfinite(loss)
            (loss/(32//args.microbatch)).backward(); loss_sum+=float(loss.detach())/(32//args.microbatch)
        grad=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        for group in optimizer.param_groups: group['lr']=args.lr*learning_rate_scale(step,args.steps)
        optimizer.step()
        if step%128==0: print(json.dumps(dict(step=step,train_loss=loss_sum,elapsed=time.time()-training_started)),flush=True)
        if step==128 or step%256==0 or step==steps:
            score=evaluate(); history.append(dict(step=step,mlm_loss=score,train_loss=loss_sum,gradient_norm=float(grad),seconds=time.time()-training_started))
            if score<best:
                best=score; metadata['selected_step']=step
                torch.save(state_to_save(model,args.method),out/'selected_state-next.pt')
                os.replace(out/'selected_state-next.pt',out/'selected_state.pt')
            write(out/'history.json',history); print(json.dumps(history[-1]),flush=True)
        if step in budgets:
            budget=out/('budget-'+str(step)); budget.mkdir()
            os.link(out/'selected_state.pt',budget/'selected_state.pt')
            # Restore the current training state after extracting the best checkpoint.
            current={k:v.detach().cpu().clone() for k,v in state_to_save(model,args.method).items()}
            saved=torch.load(budget/'selected_state.pt',map_location='cpu',weights_only=True)
            load_training_state(model,saved,args.method); del saved
            if not args.smoke: extract(model,tokenizer,budget,'after')
            load_training_state(model,current,args.method); del current
            detail=dict(budget=step,selected_step=metadata['selected_step'],selected_mlm_loss=best,
                        endpoint_mlm_loss=history[-1]['mlm_loss'],input_tokens=input_count,masked_tokens=masked_count,
                        state_sha256=digest(budget/'selected_state.pt'),state_exact_reload=True)
            metadata['budgets'][str(step)]=detail; write(budget/'selection.json',detail)
            (budget/'COMPLETE').write_text('MLM-only checkpoint selection and exact state reload.\n')
    changed=max(float((p.detach().cpu()-initial_trainable[k]).abs().max()) for k,p in model.named_parameters() if k in initial_trainable)
    assert changed>0
    frozen_after=frozen_digest(model) if args.method=='lora' else None
    assert frozen_after==frozen_before
    metadata.update(selected_mlm_loss=best,trainable_parameter_max_change=changed,
        frozen_parameters_exact=(frozen_before==frozen_after) if args.method=='lora' else None,
        frozen_parameters_sha256=frozen_before,training_seconds=time.time()-training_started,
        input_tokens=input_count,masked_tokens=masked_count,max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        max_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),plateau=plateau_summary(history),
        total_seconds=time.time()-started,state_sha256=digest(out/'selected_state.pt'),saved_state_exact_replay=True)
    write(out/'metadata.json',metadata); (out/'COMPLETE').write_text('Fixed-budget study complete; no downstream test.\n')
    print(json.dumps(metadata,indent=2),flush=True)

if __name__=='__main__': main()
