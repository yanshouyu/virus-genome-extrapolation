"""Independent local replay of exported adaptation readouts and validation metrics."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from run_sequence_baselines import features,metrics

p=argparse.ArgumentParser();p.add_argument('--collected',required=True);p.add_argument('--development',required=True);p.add_argument('--output',required=True);p.add_argument('--capacity-root');args=p.parse_args()
root=Path(args.collected);dev=Path(args.development);record=json.loads((dev/'metrics.json').read_text())
rows={s:list(csv.DictReader(Path('data/processed/clean_v2',s+'.csv').open())) for s in ['train','validation']}
y=np.array([int(r['label']) for r in rows['validation']]);results={}

def verify(name,x,model_path,prediction_path,expected):
    with np.load(model_path) as m,np.load(prediction_path) as a:
        assert np.array_equal(a['names'],[r['name'] for r in rows['validation']]);assert np.array_equal(a['labels'],y)
        np.testing.assert_allclose(m['mean'],x['train'].mean(axis=0),rtol=1e-12,atol=1e-12)
        scale=x['train'].std(axis=0,ddof=0);scale[scale==0]=1.
        np.testing.assert_allclose(m['scale'],scale,rtol=1e-10,atol=1e-12)
        z=((x['validation']-m['mean'])/m['scale'])@m['coefficients'].T+m['intercept']
        probs=(1/(1+np.exp(-z))).ravel();delta=float(np.abs(probs-a['probabilities']).max());assert delta<1e-10
        score=metrics(y,a['probabilities']);assert all(abs(score[k]-expected[k])<1e-12 for k in score)
    results[name]=dict(train_only_scaling_verified=True,probability_replay_max_abs=delta,metrics_recomputed=True)
for name,score in record['models'].items():
    if name=='kmer':x={s:features([r['sequence'] for r in rows[s]]) for s in rows}
    else:
        tokens=name.split('_');size,condition=tokens[:2];pooling=record['selection'][size]['pooling']
        run=root/(size+'-baculo-seed42' if condition=='before' else size+'-'+condition+'-'+tokens[2])
        prefix='before' if condition=='before' else 'after';x={}
        for split in rows:
            with np.load(run/(prefix+'_'+split+'.npz')) as a:
                assert np.array_equal(a['names'],[r['name'] for r in rows[split]])
                x[split]=a[pooling].astype(np.float64)
    verify(name,x,dev/(name+'_readout.npz'),dev/(name+'_validation.npz'),score)
if args.capacity_root:
    cap=Path(args.capacity_root);r=json.loads((cap/'readout/metrics.json').read_text());x={}
    for split in rows:
        with np.load(cap/('before_'+split+'.npz')) as a:x[split]=a[r['pooling']].astype(np.float64)
    verify('capacity_2.5b',x,cap/'readout/readout.npz',cap/'readout/validation.npz',r['validation'])
Path(args.output).write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
