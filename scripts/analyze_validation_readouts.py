"""Linear readouts with validation and nested grouped CV; never reads test."""
import argparse, csv, hashlib, importlib.metadata, json, warnings
from pathlib import Path
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from run_sequence_baselines import features, metrics

CS = [0.01, 0.1, 1., 10., 100.]

def fit(x, y, c):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, solver='lbfgs', max_iter=5000, random_state=42))
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x, y)
    return model


def folds(y, groups, n, seed):
    result = list(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed).split(np.zeros(len(y)),y,groups))
    for a,b in result:
        assert not set(groups[a]) & set(groups[b])
        assert len(np.unique(y[a])) == len(np.unique(y[b])) == 2
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--embedding-root', required=True)
    p.add_argument('--output', required=True)
    args=p.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    root=Path('data/processed/clean_v2')
    rows={s:list(csv.DictReader((root/(s+'.csv')).open())) for s in ['train','validation']}
    y={s:np.array([int(r['label']) for r in rows[s]]) for s in rows}
    names={s:np.array([r['name'] for r in rows[s]]) for s in rows}
    groups=np.array([r['group'] for r in rows['train']])
    candidates={'kmer':{s:features([r['sequence'] for r in rows[s]]) for s in rows}}
    for family in ['ntv1','ntv2']:
        for pooling in ['cls','mean']:
            candidates[family+'_'+pooling]={}
        for s in rows:
            with np.load(Path(args.embedding_root)/family/(s+'.npz')) as data:
                assert np.array_equal(data['names'],names[s])
                assert np.array_equal(data['labels'],y[s])
                assert np.array_equal(data['groups'],[r['group'] for r in rows[s]])
                for pooling in ['cls','mean']:
                    candidates[family+'_'+pooling][s]=data[pooling].astype(np.float64)
    outer=folds(y['train'],groups,5,42)
    membership=[dict(fold=i,fit_names=names['train'][a].tolist(),heldout_names=names['train'][b].tolist(),
                     fit_groups=len(set(groups[a])),heldout_groups=len(set(groups[b])),
                     fit_class_counts=np.bincount(y['train'][a],minlength=2).tolist(),
                     heldout_class_counts=np.bincount(y['train'][b],minlength=2).tolist())
                for i,(a,b) in enumerate(outer)]
    result=dict(packages={k:importlib.metadata.version(k) for k in ['numpy','scikit-learn']},
                dataset_sha256={s:hashlib.sha256((root/(s+'.csv')).read_bytes()).hexdigest() for s in rows},
                outer_folds=membership,models={})
    for name,x in candidates.items():
        scores=[]; models=[]
        for c in CS:
            model=fit(x['train'],y['train'],c)
            scores.append(dict(C=c,validation=metrics(y['validation'],model.predict_proba(x['validation'])[:,1])))
            models.append(model)
        best=min(range(len(CS)),key=lambda i:(scores[i]['validation']['log_loss'],CS[i]))
        model=models[best]; probs=model.predict_proba(x['validation'])[:,1]
        scaler,readout=model[0],model[1]
        np.savez_compressed(out/(name+'_model.npz'),mean=scaler.mean_,scale=scaler.scale_,
                            coefficients=readout.coef_,intercept=readout.intercept_)
        z=((x['validation']-scaler.mean_)/scaler.scale_)@readout.coef_.T+readout.intercept_
        replay=(1/(1+np.exp(-z))).ravel()
        delta=float(np.max(np.abs(replay-probs)))
        assert delta<1e-10,delta
        np.savez_compressed(out/(name+'_validation.npz'),names=names['validation'],labels=y['validation'],probabilities=probs)
        oof=np.full(len(groups),np.nan); fold_results=[]
        for i,(a,b) in enumerate(outer):
            inner=folds(y['train'][a],groups[a],3,43)
            losses=[]
            for c in CS:
                inner_probs=np.full(len(a),np.nan)
                for ia,ib in inner:
                    m=fit(x['train'][a[ia]],y['train'][a[ia]],c)
                    inner_probs[ib]=m.predict_proba(x['train'][a[ib]])[:,1]
                assert np.isfinite(inner_probs).all()
                losses.append(metrics(y['train'][a],inner_probs)['log_loss'])
            chosen=min(range(len(CS)),key=lambda j:(losses[j],CS[j]))
            m=fit(x['train'][a],y['train'][a],CS[chosen])
            oof[b]=m.predict_proba(x['train'][b])[:,1]
            fold_results.append(dict(fold=i,C=CS[chosen],inner_log_losses=losses,metrics=metrics(y['train'][b],oof[b])))
        assert np.isfinite(oof).all()
        np.savez_compressed(out/(name+'_outer_oof.npz'),names=names['train'],labels=y['train'],probabilities=oof)
        result['models'][name]=dict(candidates=scores,selected_C=CS[best],validation=scores[best]['validation'],
                                   replay_max_abs=delta,outer_folds=fold_results,outer_pooled=metrics(y['train'],oof))
        (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
        print(name, result['models'][name]['validation'],flush=True)
    (out/'COMPLETE').write_text('All five model families completed; no test predictions.\n')

if __name__=='__main__':
    main()

