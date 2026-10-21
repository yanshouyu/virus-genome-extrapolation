"""One fixed secondary test pass after development selection is complete."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoConfig,AutoTokenizer
from baculo_tss_probe.models import custom_nt_classes,_normalize_backbone_config_fields
from run_sequence_baselines import features,metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/2026-10-21');args=p.parse_args();root=Path(args.root)
    assert (root/'development/COMPLETE').is_file()
    selection=json.loads((root/'development/selection.json').read_text())
    output=root/'secondary-test';output.mkdir(parents=True,exist_ok=False)
    rows=list(csv.DictReader(Path('data/processed/clean_v2/test.csv').open()))
    y=np.array([int(r['label']) for r in rows]);sources=json.loads((root/'sources.json').read_text());results={}
    def predict(name,x):
        with np.load(root/'development'/(name+'_readout.npz')) as m:
            z=((x-m['mean'])/m['scale'])@m['coefficients'].T+m['intercept']
            prob=(1/(1+np.exp(-z))).ravel()
        results[name]=metrics(y,prob)
        np.savez_compressed(output/(name+'.npz'),names=np.array([r['name'] for r in rows]),labels=y,probabilities=prob)
    for size in ['100m','500m']:
        source=sources[size];cfg=AutoConfig.from_pretrained(source['source'],revision=source['revision'],trust_remote_code=True,local_files_only=True)
        _normalize_backbone_config_fields(cfg);cls,_=custom_nt_classes(source['source'],source['revision'],True)
        model=cls.from_pretrained(source['source'],revision=source['revision'],config=cfg,local_files_only=True).cuda().eval()
        tokenizer=AutoTokenizer.from_pretrained(source['source'],revision=source['revision'],trust_remote_code=True,local_files_only=True)
        def extract():
            arrays=[]
            with torch.inference_mode():
                for start in range(0,len(rows),32):
                    tokens=tokenizer([r['sequence'] for r in rows[start:start+32]],padding=True,truncation=False,return_tensors='pt')
                    special=torch.isin(tokens['input_ids'],torch.tensor(tokenizer.all_special_ids))
                    inputs={k:v.cuda() for k,v in tokens.items() if k in ['input_ids','attention_mask']}
                    h=model.esm(**inputs).last_hidden_state
                    if selection[size]['pooling']=='cls':pooled=h[:,0]
                    else:
                        mask=(tokens['attention_mask'].bool()&~special).cuda()
                        pooled=(h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
                    arrays.append(pooled.cpu().numpy().astype(np.float64))
            return np.concatenate(arrays)
        predict(size+'_before',extract())
        for condition in ['baculo','general']:
            for seed in [42,43,44]:
                name=size+'_'+condition+'_seed'+str(seed)
                run=root/'scientific'/(size+'-'+condition+'-seed'+str(seed));assert (run/'COMPLETE').exists()
                state=torch.load(run/'selected_state.pt',map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True);del state
                predict(name,extract())
        del model;torch.cuda.empty_cache()
    predict('kmer',features([r['sequence'] for r in rows]))
    (output/'metrics.json').write_text(json.dumps(results,indent=2)+'\n');(output/'COMPLETE').write_text('Fixed secondary evaluation; no refitting.\n')
if __name__=='__main__':main()
