"""Supplementary fixed-model k-mer comparisons and execution/data integrity."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from evaluate_controlled_adaptation import basic_scores

out=Path('documents/adaptation-2026-10-23');dev=Path('runs/2026-10-23/development')
rows=list(csv.DictReader(Path('data/processed/clean_v2/validation.csv').open()))
y=np.array([int(r['label']) for r in rows]);groups=np.array([r['group'] for r in rows]);blocks=[np.where(groups==g)[0] for g in np.unique(groups)]
def load(name):
    with np.load(dev/(name+'_validation.npz')) as a:
        assert np.array_equal(a['names'],[r['name'] for r in rows])
        return a['probabilities'].copy()
baseline=load('kmer');candidates={}
for size in ['100m','500m']:
    candidates[size+'_before']=[load(size+'_before')]
    candidates[size+'_baculo']=[load(size+'_baculo_seed'+str(seed)) for seed in [42,43,44]]
with np.load('runs/2026-10-23/capacity-2.5b/readout/validation.npz') as a:candidates['capacity_2.5b']=[a['probabilities'].copy()]
rng=np.random.default_rng(42);samples=[np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))]) for _ in range(2000)]
results={}
for name,predictions in candidates.items():
    def diff(ix):return np.mean([basic_scores(y[ix],p[ix]) for p in predictions],axis=0)-basic_scores(y[ix],baseline[ix])
    point=diff(np.arange(len(y)));boot=np.stack([diff(ix) for ix in samples])
    results[name+'_minus_kmer']=dict(log_loss_delta=float(point[0]),mcc_delta=float(point[1]),log_loss_interval=np.quantile(boot[:,0],[.025,.975]).tolist(),mcc_interval=np.quantile(boot[:,1],[.025,.975]).tolist())
(out/'kmer-reference-comparisons.json').write_text(json.dumps(dict(scope='Supplementary fixed-model comparisons, no additional selection or fitting.',resamples=2000,groups=len(blocks),comparisons=results),indent=2)+'\n')
accounting={r['JobID']:r for r in csv.DictReader((out/'jobs.psv').open(),delimiter='|')}
cost={}
for size,original,control in [('100m','24896408','24897074'),('500m','24896596','24897075')]:
    cost[size]={}
    for condition,base,offset in [('baculo',original,0),('general',control,1)]:
        jobids=[base+'_'+str(2*i+offset) for i in range(3)]
        records=[accounting[job] for job in jobids];assert all(r['State']=='COMPLETED' for r in records)
        metas=[json.loads(Path('runs/2026-10-23/collected',size+'-'+condition+'-seed'+str(seed),'metadata.json').read_text()) for seed in [42,43,44]]
        cost[size][condition]=dict(jobids=jobids,slurm_allocated_seconds=sum(int(r['ElapsedRaw']) for r in records),training_seconds=sum(m['training_seconds'] for m in metas),selected_steps=[m['selected_step'] for m in metas],gpu=metas[0]['gpu'],max_gpu_allocated_bytes=max(m['max_gpu_allocated_bytes'] for m in metas))
(out/'compute-cost.json').write_text(json.dumps(cost,indent=2)+'\n')
prior=json.loads(Path('documents/adaptation-2026-10-21/preparation-provenance.json').read_text())['downstream_dataset_sha256']
current={s:hashlib.sha256(Path('data/processed/clean_v2',s+'.csv').read_bytes()).hexdigest() for s in ['train','validation','test']}
assert current==prior
(out/'downstream-integrity.json').write_text(json.dumps(dict(dataset_sha256=current,unchanged_from_preparation=True,validation_prediction_rows=len(y),real_test_predictions_generated=False,note='Test file hashed for identity only; real test inference was excluded from this window.'),indent=2)+'\n')
print(json.dumps(dict(comparisons=results,cost=cost),indent=2))
