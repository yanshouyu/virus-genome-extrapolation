"""Fixed-readout development analysis for the predeclared duration study."""
import argparse, csv, hashlib, json, importlib.metadata
from pathlib import Path
import numpy as np
from analyze_validation_readouts import fit
from evaluate_controlled_adaptation import basic_scores
from run_sequence_baselines import metrics, features


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='runs/2026-10-24/scientific')
    p.add_argument('--output',required=True); p.add_argument('--include-pilot',action='store_true')
    args=p.parse_args(); root=Path(args.root); out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    frozen=json.loads(Path('runs/2026-10-21/frozen-readout-selection.json').read_text())['selection']
    rows={s:list(csv.DictReader(Path('data/processed/clean_v2',s+'.csv').open())) for s in ['train','validation']}
    labels={s:np.array([int(r['label']) for r in rows[s]]) for s in rows}
    names={s:np.array([r['name'] for r in rows[s]]) for s in rows}
    groups={s:np.array([r['group'] for r in rows[s]]) for s in rows}
    result=dict(models={},comparison={},runs={},selection=frozen,pilot_included=args.include_pilot,
                dataset_sha256={s:hashlib.sha256(Path('data/processed/clean_v2',s+'.csv').read_bytes()).hexdigest() for s in rows},
                source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [Path(__file__),Path('scripts/analyze_validation_readouts.py'),Path('scripts/evaluate_controlled_adaptation.py'),Path('scripts/run_sequence_baselines.py')]},
                frozen_readout_sha256=hashlib.sha256(Path('runs/2026-10-21/frozen-readout-selection.json').read_bytes()).hexdigest(),
                packages={name:importlib.metadata.version(name) for name in ['numpy','scikit-learn','scipy']})
    predictions={}; budgets=[512,1024,2048,4096]
    def read(path,prefix,split,pooling):
        with np.load(path/(prefix+'_'+split+'.npz')) as a:
            assert np.array_equal(a['names'],names[split]) and np.array_equal(a['labels'],labels[split])
            assert np.array_equal(a['groups'],groups[split])
            return a[pooling].astype(np.float64)
    def evaluate(name,x,c):
        model=fit(x['train'],labels['train'],c)
        prob=model.predict_proba(x['validation'])[:,1]
        z=((x['validation']-model[0].mean_)/model[0].scale_)@model[1].coef_.T+model[1].intercept_
        replay=(1/(1+np.exp(-z))).ravel(); delta=float(np.abs(replay-prob).max()); assert delta<1e-10
        assert np.allclose(model[0].mean_,x['train'].mean(0),rtol=1e-12,atol=1e-12)
        np.savez_compressed(out/(name+'_readout.npz'),mean=model[0].mean_,scale=model[0].scale_,coefficients=model[1].coef_,intercept=model[1].intercept_)
        np.savez_compressed(out/(name+'_validation.npz'),names=names['validation'],labels=labels['validation'],probabilities=prob)
        with (out/(name+'_validation.csv')).open('w',newline='') as f:
            writer=csv.writer(f,lineterminator='\n'); writer.writerow(['name','label','probability'])
            writer.writerows(zip(names['validation'],labels['validation'],prob))
        result['models'][name]=dict(**metrics(labels['validation'],prob),readout_replay_max_abs=delta)
        predictions[name]=prob
    for size in ['100m','500m']:
        assert frozen[size]['pooling']=='mean' and frozen[size]['C']==.01
        before={s:read(root/f'{size}-full-baculo-seed42','before',s,'mean') for s in rows}
        evaluate(size+'_before',before,.01)
        methods=['full','lora'] if args.include_pilot and size=='100m' else ['full']
        for method in methods:
            for condition in ['baculo','general']:
                for seed in ([42,43,44] if method=='full' else [42]):
                    name=f'{size}-{method}-{condition}-seed{seed}'; run=root/name
                    assert (run/'COMPLETE').exists(),run
                    meta=json.loads((run/'metadata.json').read_text())
                    assert meta['all_source_parameters_exact'] and meta['saved_state_exact_replay']
                    if method=='lora': assert meta['frozen_parameters_exact']
                    for split in rows:
                        baseline=read(run,'before',split,'mean')
                        assert np.allclose(baseline,before[split],rtol=1e-5,atol=1e-6)
                    result['runs'][name]=meta
                    for budget in budgets:
                        assert (run/f'budget-{budget}'/'COMPLETE').exists()
                        x={s:read(run/f'budget-{budget}','after',s,'mean') for s in rows}
                        evaluate(name+f'-budget{budget}',x,.01)
        for seed in [42,43,44]:
            a=result['runs'][f'{size}-full-baculo-seed{seed}']; b=result['runs'][f'{size}-full-general-seed{seed}']
            assert a['input_tokens']==b['input_tokens']==4096*32*257
            assert a['masked_tokens']==b['masked_tokens']
            for budget in budgets:
                assert a['budgets'][str(budget)]['masked_tokens']==b['budgets'][str(budget)]['masked_tokens']
    evaluate('kmer',{s:features([r['sequence'] for r in rows[s]]) for s in rows},frozen['kmer']['C'])
    rng=np.random.default_rng(42); unique=np.unique(groups['validation'])
    blocks=[np.where(groups['validation']==g)[0] for g in unique]
    samples=[np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))]) for _ in range(2000)]
    all_indices=np.arange(len(labels['validation']))
    def compare(key,left,right):
        def delta(ix):
            def avg(keys): return np.mean([basic_scores(labels['validation'][ix],predictions[k][ix]) for k in keys],axis=0)
            return avg(left)-avg(right)
        point=delta(all_indices); boot=np.stack([delta(ix) for ix in samples])
        result['comparison'][key]=dict(log_loss_delta=float(point[0]),mcc_delta=float(point[1]),
            log_loss_interval=np.quantile(boot[:,0],[.025,.975]).tolist(),mcc_interval=np.quantile(boot[:,1],[.025,.975]).tolist())
    for size in ['100m','500m']:
        for budget in budgets:
            arms={c:[f'{size}-full-{c}-seed{s}-budget{budget}' for s in [42,43,44]] for c in ['baculo','general']}
            compare(f'{size}-budget{budget}-baculo-minus-general',arms['baculo'],arms['general'])
            compare(f'{size}-budget{budget}-baculo-minus-before',arms['baculo'],[size+'_before'])
        for condition in ['baculo','general']:
            compare(f'{size}-{condition}-4096-minus-512',[f'{size}-full-{condition}-seed{s}-budget4096' for s in [42,43,44]],
                    [f'{size}-full-{condition}-seed{s}-budget512' for s in [42,43,44]])
    if args.include_pilot:
        for condition in ['baculo','general']:
            compare(f'pilot-100m-{condition}-lora-minus-full-seed42',
                    [f'100m-lora-{condition}-seed42-budget4096'],[f'100m-full-{condition}-seed42-budget4096'])
    result['bootstrap']=dict(resamples=2000,groups=len(unique),conditional=True)
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'COMPLETE').write_text('Fixed-readout development analysis, no test inference.\n')
    print(json.dumps(result['comparison'],indent=2))

if __name__=='__main__': main()
