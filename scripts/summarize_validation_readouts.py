"""Verify local exports and write compact report tables/figure; no test data."""
import csv, hashlib, json, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_sequence_baselines import features, metrics

out=Path('documents/experiments-2026-10-20')
cache=Path('runs/2026-10-20/collected')
r=json.loads((out/'readout-metrics.json').read_text())
rows=list(csv.DictReader(Path('data/processed/clean_v2/validation.csv').open()))
y=np.array([int(row['label']) for row in rows])
labels=['k-mer','NTv1 CLS','NTv1 mean','NTv2 CLS','NTv2 mean']
keys=list(r['models'])
verification={}
flat=[]
for name,m in r['models'].items():
    if name=='kmer': x=features([row['sequence'] for row in rows])
    else:
        family,pooling=name.split('_')
        with np.load(cache/'embeddings'/family/'validation.npz') as data:
            assert np.array_equal(data['names'],[row['name'] for row in rows])
            x=data[pooling].astype(np.float64)
    with np.load(cache/'readouts'/(name+'_model.npz')) as model:
        z=((x-model['mean'])/model['scale'])@model['coefficients'].T+model['intercept']
        probs=(1/(1+np.exp(-z))).ravel()
    with np.load(cache/'readouts'/(name+'_validation.npz')) as original:
        assert np.array_equal(original['labels'],y)
        delta=float(np.max(np.abs(probs-original['probabilities'])))
        assert delta<1e-10
        recalculated=metrics(y,original['probabilities'])
    assert all(abs(recalculated[k]-m['validation'][k])<1e-12 for k in recalculated)
    verification[name]=dict(export_replay_max_abs=delta,metrics_recomputed=True)
    for stage,score in [('validation',m['validation']),('nested_outer_pooled',m['outer_pooled'])]:
        flat.append(dict(model=name,stage=stage,**score))
    for suffix in ['validation','outer_oof']:
        with np.load(cache/'readouts'/(name+'_'+suffix+'.npz')) as pred:
            with (out/(name+'_'+suffix+'.csv')).open('w',newline='') as handle:
                writer=csv.writer(handle); writer.writerow(['name','label','probability'])
                writer.writerows(zip(pred['names'],pred['labels'],pred['probabilities']))
with (out/'comparison.csv').open('w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
colors=['#64748b','#2563eb','#60a5fa','#c2410c','#fb923c']
for ax,stage,title in zip(axes,['validation','outer_pooled'],['Fixed validation (77 windows)','Nested grouped CV (233 training windows)']):
    values=[r['models'][key][stage]['log_loss'] for key in keys]
    ax.barh(labels,values,color=colors)
    ax.invert_yaxis(); ax.set_xlim(0,1.05);ax.set_xlabel('Log loss (lower is better)');ax.set_title(title)
    for i,value in enumerate(values): ax.text(1.025,i,f'{value:.3f}',va='center',ha='right',fontsize=10)
    if stage=='outer_pooled':
        for i,key in enumerate(keys):
            ax.scatter([f['metrics']['log_loss'] for f in r['models'][key]['outer_folds']],[i]*5,
                       s=18,facecolors='white',edgecolors='black',zorder=3)
fig.suptitle('Frozen representations with regularized linear readouts',fontsize=14)
fig.savefig(out/'validation-comparison.png',dpi=180);plt.close(fig)
artifacts={}
for path in sorted(cache.rglob('*')):
    if path.is_file():artifacts[str(path.relative_to(cache))]=hashlib.sha256(path.read_bytes()).hexdigest()
scripts={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
    Path('scripts/extract_validation_embeddings.py'),Path('scripts/analyze_validation_readouts.py'),
    Path('scripts/extract_validation_embeddings.slurm'),Path('scripts/analyze_validation_readouts.slurm'),
    Path('src/baculo_tss_probe/models.py'),Path('scripts/run_sequence_baselines.py')]}
provenance=dict(base_commit='3a0e592f1c293128e057c1ca75945505ca6ed4fd',source_sha256=scripts,
                collected_artifact_sha256=artifacts,export_verification=verification,
                extraction={family:json.loads((cache/'embeddings'/family/'metadata.json').read_text()) for family in ['ntv1','ntv2']})
(out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(verification,indent=2))

