"""Training-only readout selection for the frozen 2.5B multispecies reference."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from evaluate_controlled_adaptation import screen_pooling,select_c
from analyze_validation_readouts import fit
from run_sequence_baselines import metrics
root=Path('runs/2026-10-21/capacity-2.5b');assert (root/'COMPLETE').exists()
metadata=json.loads((root/'metadata.json').read_text());assert metadata['all_source_parameters_exact']
out=root/'readout';out.mkdir(exist_ok=False)
rows={s:list(csv.DictReader(Path('data/processed/clean_v2',s+'.csv').open())) for s in ['train','validation']}
y={s:np.array([int(r['label']) for r in rows[s]]) for s in rows};x={}
groups=np.array([r['group'] for r in rows['train']])
for split in rows:
    with np.load(root/('before_'+split+'.npz')) as a:
        assert np.array_equal(a['names'],[r['name'] for r in rows[split]])
        assert np.array_equal(a['labels'],y[split])
        x[split]={p:a[p].astype(np.float64) for p in ['cls','mean']}
pooling,screen=screen_pooling(x['train'],y['train'],groups)
c,losses=select_c(x['train'][pooling],y['train'],groups)
model=fit(x['train'][pooling],y['train'],c)
prob=model.predict_proba(x['validation'][pooling])[:,1]
np.savez_compressed(out/'readout.npz',mean=model[0].mean_,scale=model[0].scale_,coefficients=model[1].coef_,intercept=model[1].intercept_)
z=((x['validation'][pooling]-model[0].mean_)/model[0].scale_)@model[1].coef_.T+model[1].intercept_
replay=(1/(1+np.exp(-z))).ravel();delta=float(np.abs(replay-prob).max());assert delta<1e-10
np.savez_compressed(out/'validation.npz',names=np.array([r['name'] for r in rows['validation']]),labels=y['validation'],probabilities=prob)
result=dict(pooling=pooling,C=c,training_C_losses=losses,unadapted_nested_pooling_screen=screen,
            validation=metrics(y['validation'],prob),readout_replay_max_abs=delta,
            dataset_sha256={s:hashlib.sha256(Path('data/processed/clean_v2',s+'.csv').read_bytes()).hexdigest() for s in rows},
            exploratory=True,source=metadata['source'],revision=metadata['revision'])
(out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');(out/'COMPLETE').write_text('Exploratory validation reference; no test.\n')
print(json.dumps(result,indent=2))
