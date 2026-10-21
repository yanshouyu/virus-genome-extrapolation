"""Training-only readout selection and paired adaptation development evaluation."""
import argparse,csv,json,hashlib,warnings
from pathlib import Path
import numpy as np
from analyze_validation_readouts import fit,folds,CS
from run_sequence_baselines import features,metrics


def cv_predictions(x,y,groups,c,n=3,seed=43):
    result=np.full(len(y),np.nan)
    for a,b in folds(y,groups,n,seed):result[b]=fit(x[a],y[a],c).predict_proba(x[b])[:,1]
    assert np.isfinite(result).all()
    return result


def select_c(x,y,groups):
    losses=[metrics(y,cv_predictions(x,y,groups,c))['log_loss'] for c in CS]
    return CS[min(range(len(CS)),key=lambda i:(losses[i],CS[i]))],losses


def screen_pooling(x,y,groups):
    result={}
    for pooling in ['cls','mean']:
        prediction=np.full(len(y),np.nan);details=[]
        for a,b in folds(y,groups,5,42):
            c,losses=select_c(x[pooling][a],y[a],groups[a])
            prediction[b]=fit(x[pooling][a],y[a],c).predict_proba(x[pooling][b])[:,1]
            details.append(dict(C=c,inner_losses=losses,metrics=metrics(y[b],prediction[b])))
        result[pooling]=dict(pooled=metrics(y,prediction),folds=details)
    chosen=min(['cls','mean'],key=lambda p:(result[p]['pooled']['log_loss'],p!='cls'))
    return chosen,result


def mcc(y,p):
    q=p>.5;positive=y==1
    tp=np.sum(q&positive);tn=np.sum(~q&~positive);fp=np.sum(q&~positive);fn=np.sum(~q&positive)
    denominator=np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return float((tp*tn-fp*fn)/denominator) if denominator else 0.


def basic_scores(y,p):
    p=np.clip(p,np.finfo(float).eps,1-np.finfo(float).eps)
    return np.array([-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)),mcc(y,p)])


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/2026-10-21/scientific')
    p.add_argument('--data-dir',default='data/processed/clean_v2');p.add_argument('--output',default='runs/2026-10-21/development')
    p.add_argument('--bootstrap',type=int,default=2000)
    args=p.parse_args();root=Path(args.root);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    rows={s:list(csv.DictReader(Path(args.data_dir,s+'.csv').open())) for s in ['train','validation']}
    y={s:np.array([int(r['label']) for r in rows[s]]) for s in rows}
    groups={s:np.array([r['group'] for r in rows[s]]) for s in rows}
    names={s:np.array([r['name'] for r in rows[s]]) for s in rows}
    predictions={};result={'selection':{},'models':{},'comparison':{},'dataset_sha256':{s:hashlib.sha256(Path(args.data_dir,s+'.csv').read_bytes()).hexdigest() for s in rows}}
    def read(run,prefix,split):
        with np.load(run/(prefix+'_'+split+'.npz')) as a:
            assert np.array_equal(a['names'],names[split]);assert np.array_equal(a['labels'],y[split]);assert np.array_equal(a['groups'],groups[split])
            return {p:a[p].astype(np.float64) for p in ['cls','mean']}
    def evaluate(name,x,c):
        model=fit(x['train'],y['train'],c);probs=model.predict_proba(x['validation'])[:,1]
        predictions[name]=probs
        np.savez_compressed(out/(name+'_readout.npz'),mean=model[0].mean_,scale=model[0].scale_,coefficients=model[1].coef_,intercept=model[1].intercept_)
        replay=(1/(1+np.exp(-(((x['validation']-model[0].mean_)/model[0].scale_)@model[1].coef_.T+model[1].intercept_)))).ravel()
        assert np.max(np.abs(replay-probs))<1e-10
        np.savez_compressed(out/(name+'_validation.npz'),names=names['validation'],labels=y['validation'],probabilities=probs)
        result['models'][name]=metrics(y['validation'],probs)
    for size in ['100m','500m']:
        reference=root/(size+'-baculo-seed42')
        before={s:read(reference,'before',s) for s in rows}
        pooling,screen=screen_pooling(before['train'],y['train'],groups['train'])
        c,losses=select_c(before['train'][pooling],y['train'],groups['train'])
        result['selection'][size]=dict(pooling=pooling,C=c,C_inner_losses=losses,unadapted_nested_pooling_screen=screen)
        evaluate(size+'_before',{s:before[s][pooling] for s in rows},c)
        for condition in ['baculo','general']:
            for seed in [42,43,44]:
                run=root/(size+'-'+condition+'-seed'+str(seed));assert (run/'COMPLETE').is_file(),run
                for split in rows:
                    baseline=read(run,'before',split)
                    assert all(np.array_equal(baseline[p],before[split][p]) for p in ['cls','mean'])
                after={s:read(run,'after',s)[pooling] for s in rows}
                evaluate(size+'_'+condition+'_seed'+str(seed),after,c)
    x={s:features([r['sequence'] for r in rows[s]]) for s in rows}
    c,losses=select_c(x['train'],y['train'],groups['train']);result['selection']['kmer']=dict(C=c,C_inner_losses=losses)
    evaluate('kmer',x,c)
    # Selection is finalized before any possible secondary test evaluation.
    (out/'selection.json').write_text(json.dumps(result['selection'],indent=2)+'\n')
    rng=np.random.default_rng(42);unique=np.unique(groups['validation'])
    blocks=[np.where(groups['validation']==g)[0] for g in unique]
    samples=[np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))]) for _ in range(args.bootstrap)]
    def scores(key,indices):return basic_scores(y['validation'][indices],predictions[key][indices])
    all_indices=np.arange(len(y['validation']))
    for size in ['100m','500m']:
        for left,right in [('baculo','before'),('general','before'),('baculo','general')]:
            def difference(indices):
                def condition_score(condition):
                    if condition=='before':return scores(size+'_before',indices)
                    return np.mean([scores(size+'_'+condition+'_seed'+str(seed),indices) for seed in [42,43,44]],axis=0)
                return condition_score(left)-condition_score(right)
            boot=np.stack([difference(ix) for ix in samples]);point=difference(all_indices)
            result['comparison'][size+'_'+left+'_minus_'+right]=dict(log_loss_delta=float(point[0]),mcc_delta=float(point[1]),
                log_loss_interval=np.quantile(boot[:,0],[.025,.975]).tolist(),mcc_interval=np.quantile(boot[:,1],[.025,.975]).tolist())
    result['bootstrap']=dict(resamples=args.bootstrap,groups=len(unique),conditional=True)
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');(out/'COMPLETE').write_text('Development comparison frozen.\n')
    print(json.dumps(result['comparison'],indent=2))
if __name__=='__main__':main()
