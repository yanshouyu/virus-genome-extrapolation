"""CPU-only analysis of saved October 24 results; no model/test inference or fitting."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_sequence_baselines import features
from sklearn.metrics import (log_loss, matthews_corrcoef, roc_auc_score, accuracy_score,
                             precision_score, recall_score, average_precision_score)
def metrics(y,p):
 q=p>.5
 return dict(log_loss=float(log_loss(y,p,labels=[0,1])),mcc=float(matthews_corrcoef(y,q)),
             roc_auc=float(roc_auc_score(y,p)),accuracy=float(accuracy_score(y,q)),
             precision=float(precision_score(y,q,zero_division=0)),recall=float(recall_score(y,q,zero_division=0)),
             average_precision=float(average_precision_score(y,p)))
R=Path('documents/adaptation-2026-10-24/results'); O=R.parent/'analysis';O.mkdir(exist_ok=True)
D=Path('runs/2026-10-24/development-with-pilot');C=D.parent/'collected'
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=load(R/'development-metrics.json');assert r==load(D/'metrics.json')
assert len(r['runs'])==14 and len(r['models'])==59
rows={s:list(csv.DictReader(Path('data/processed/clean_v2',s+'.csv').open())) for s in ['train','validation']}
assert [len(v) for v in rows.values()]==[233,77]
assert r['dataset_sha256']==load(Path('documents/adaptation-2026-10-23/development-metrics.json'))['dataset_sha256']
for s,h in r['dataset_sha256'].items():assert s in rows and sha(Path('data/processed/clean_v2',s+'.csv'))==h
assert sha(Path('runs/2026-10-21/frozen-readout-selection.json'))==r['frozen_readout_sha256']
H=load(R/'histories.json'); sources=dict(r['source_sha256']);panels=[];replays={};original={}
for size,method in [('100m','full'),('500m','full'),('100m','lora')]:
 a=load(R/f'fresh-replay-{size}-{method}.json');assert a==load(D.parent/f'fresh-replay-{size}-{method}.json')
 v=a.pop('_verification');sources.update(v['source_sha256']);replays.update(a)
 assert sha(R.parent/'verification-amendment.md')==v['amendment_sha256']
 panels.append({k:v[k] for k in ['audit_panel','audit_panel_sha256','audit_mask_sha256','audit_groups']});original[size]=v['original_audit_mlm_loss']
assert panels[0]==panels[1]==panels[2] and len(panels[0]['audit_panel'])==256
B=[512,1024,2048,4096]; colors={'baculo':'#2563eb','general':'#d97706'}
for name,m in r['runs'].items():
 assert load(C/name/'metadata.json')==m and load(C/name/'history.json')==H[name] and (C/name/'COMPLETE').exists()
 sources.update(m['source_sha256']);assert sha(R.parent/'protocol.md')==m['protocol_sha256']
 assert m['pretraining_scope']=='multispecies' and m['input_tokens']==4096*32*257
 assert replays[name]['selected_states_exact']
 if '-lora-' in name:assert m['frozen_parameters_exact']
 for b in B:
  best=min((h for h in H[name] if h['step']<=b),key=lambda h:(h['mlm_loss'],h['step']));v=m['budgets'][str(b)]
  assert (best['step'],best['mlm_loss'])==(v['selected_step'],v['selected_mlm_loss'])
  assert (C/name/f'budget-{b}'/'COMPLETE').exists()
 prev=min(h['mlm_loss'] for h in H[name] if h['step']<=3072)
 assert abs((prev-m['selected_mlm_loss'])/prev-m['plateau']['best_loss_relative_improvement'])<1e-12
 assert m['plateau']['plateau_by_rule']==((prev-m['selected_mlm_loss'])/prev<.005)
 assert m['budgets']['4096']['endpoint_mlm_loss']==H[name][-1]['mlm_loss']
for p,h in sources.items():assert sha(Path(p.split('virus-genome-extrapolation/')[-1]))==h,p
for size in ['100m','500m']:
 assert r['selection'][size]['pooling']=='mean' and r['selection'][size]['C']==.01
 for seed in [42,43,44]:
  a,b=[r['runs'][f'{size}-full-{c}-seed{seed}'] for c in colors]
  assert a['input_tokens']==b['input_tokens'] and a['masked_tokens']==b['masked_tokens']
  for budget in B:assert a['budgets'][str(budget)]['masked_tokens']==b['budgets'][str(budget)]['masked_tokens']
A=list(csv.DictReader((R.parent/'jobs-completed.psv').open(),delimiter='|'))
assert all(a['State']=='COMPLETED' and a['ExitCode']=='0:0' for a in A)
for array,n in [('24919261',6),('24919263',6),('24919267',2)]:assert all(any(a['JobID']==f'{array}_{i}' for a in A) for i in range(n))
y=np.array([int(v['label']) for v in rows['validation']]);g=np.array([v['group'] for v in rows['validation']])
blocks=[np.flatnonzero(g==v) for v in np.unique(g)];assert len(blocks)==64
rng=np.random.default_rng(42);ixs=[np.concatenate([blocks[i] for i in rng.integers(64,size=64)]) for _ in range(2000)]
W=np.vstack([np.ones(len(y)),np.stack([np.bincount(ix,minlength=len(y)) for ix in ixs])]);W/=W.sum(1,keepdims=True)
S={};checks={}
for name,expected in r['models'].items():
 if name=='kmer':x={s:features([v['sequence'] for v in rows[s]]) for s in rows}
 else:
  if name.endswith('_before'):path=C/(name[:4]+'-full-baculo-seed42');prefix='before'
  else:
   run,b=name.rsplit('-budget',1);path=C/run/('budget-'+b);prefix='after'
  x={}
  for s in rows:
   with np.load(path/f'{prefix}_{s}.npz') as a:
    for k,values in [('names',[v['name'] for v in rows[s]]),('labels',[int(v['label']) for v in rows[s]]),('groups',[v['group'] for v in rows[s]])]:assert np.array_equal(a[k],values)
    x[s]=a['mean'].astype(float)
 saved=list(csv.DictReader((R/(name+'_validation.csv')).open()))
 assert [v['name'] for v in saved]==[v['name'] for v in rows['validation']] and np.array_equal([int(v['label']) for v in saved],y)
 p=np.array([float(v['probability']) for v in saved]);q=p>.5
 with np.load(D/(name+'_readout.npz')) as a:
  np.testing.assert_allclose(a['mean'],x['train'].mean(0),rtol=1e-12,atol=1e-12)
  scale=x['train'].std(0);scale[scale==0]=1;np.testing.assert_allclose(a['scale'],scale,rtol=1e-10,atol=1e-12)
  z=((x['validation']-a['mean'])/a['scale'])@a['coefficients'].T+a['intercept']
  delta=float(abs((1/(1+np.exp(-z))).ravel()-p).max());assert delta<1e-10
 actual=metrics(y,p);assert all(abs(actual[k]-expected[k])<1e-12 for k in actual)
 ll=W@(-y*np.log(p)-(1-y)*np.log1p(-p));tp,tn,fp,fn=[W@v for v in [q*(y==1),(~q)*(y==0),q*(y==0),(~q)*(y==1)]]
 den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));mcc=np.divide(tp*tn-fp*fn,den,out=np.zeros_like(tp),where=den>0)
 cmp=(p[y==1,None]>p[None,y==0])+.5*(p[y==1,None]==p[None,y==0]);wp,wn=W[:,y==1],W[:,y==0]
 auc=((wp@cmp)*wn).sum(1)/(wp.sum(1)*wn.sum(1));S[name]=np.stack([ll,mcc,auc],axis=1)
 np.testing.assert_allclose(S[name][0],[actual[k] for k in ['log_loss','mcc','roc_auc']],atol=1e-12)
 checks[name]=dict(training_only_scaling=True,replay_max_abs=delta,seven_metrics_recomputed=True)
E={}
def compare(key,left,right):
 e=np.mean([S[k] for k in left],axis=0)-np.mean([S[k] for k in right],axis=0)
 E[key]={k:dict(delta=float(e[0,i]),interval=np.quantile(e[1:,i],[.025,.975]).tolist()) for i,k in enumerate(['log_loss','mcc','roc_auc'])}
for size in ['100m','500m']:
 for b in B:
  arms={c:[f'{size}-full-{c}-seed{s}-budget{b}' for s in [42,43,44]] for c in colors}
  compare(f'{size}-budget{b}-baculo-minus-general',arms['baculo'],arms['general']);compare(f'{size}-budget{b}-baculo-minus-before',arms['baculo'],[size+'_before'])
 for c in colors:compare(f'{size}-{c}-4096-minus-512',[f'{size}-full-{c}-seed{s}-budget4096' for s in [42,43,44]],[f'{size}-full-{c}-seed{s}-budget512' for s in [42,43,44]])
 compare(f'{size}-budget4096-baculo-minus-kmer',[f'{size}-full-baculo-seed{s}-budget4096' for s in [42,43,44]],['kmer'])
for c in colors:compare(f'pilot-100m-{c}-lora-minus-full-seed42',[f'100m-lora-{c}-seed42-budget4096'],[f'100m-full-{c}-seed42-budget4096'])
for key,v in r['comparison'].items():
 for k in ['log_loss','mcc']:
  assert abs(E[key][k]['delta']-v[k+'_delta'])<1e-12
  np.testing.assert_allclose(E[key][k]['interval'],v[k+'_interval'],atol=1e-12,rtol=0)
V=dict(scientific_runs=14,readouts=59,accounting_rows=len(A),all_accounting_completed=True,source_hashes_checked=len(sources),selections_and_plateaus_recomputed=True,train_validation_hashes_match=True,probability_replay_max_abs=max(v['replay_max_abs'] for v in checks.values()),existing_intervals_reproduced=True,resamples=2000,validation_groups=64,test_opened=False,training_or_embedding_inference_launched=False,per_readout=checks)
for name,v in [('verification',V),('paired-effects',E),('audit-panel',panels[0])]: (O/(name+'.json')).write_text(json.dumps(v,indent=2)+'\n')
with (O/'readout-scores.csv').open('w',newline='') as f:
 w=csv.writer(f,lineterminator="\n");w.writerow(['readout','metric','score','conditional_95_low','conditional_95_high'])
 for name,a in S.items():
  for i,k in enumerate(['log_loss','mcc','roc_auc']):w.writerow([name,k,a[0,i],*np.quantile(a[1:,i],[.025,.975])])
lines=['| Comparison (left minus right) | Log loss [95% CI] | MCC [95% CI] | ROC AUC [95% CI] |','|---|---:|---:|---:|']
for key,e in E.items():
 if 'budget4096' in key or '4096-minus-512' in key or key.startswith('pilot'):lines.append('| '+key+' | '+' | '.join(f"{v['delta']:+.4f} [{v['interval'][0]:+.4f}, {v['interval'][1]:+.4f}]" for v in e.values())+' |')
(O/'paired-effects.md').write_text('\n'.join(lines)+'\n')
audit={(v['run'],int(v['budget'])):float(v['selected_audit_mlm_loss']) for v in csv.DictReader((R/'independent-mlm-audit.csv').open())}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
 for ext in ['png','pdf']:fig.savefig(O/(name+'.'+ext),dpi=180)
 plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for col,size in enumerate(['100m','500m']):
 for c in colors:
  for s in [42,43,44]:
   name=f'{size}-full-{c}-seed{s}';m=r['runs'][name];h=H[name]
   axes[0,col].plot([0]+[v['step'] for v in h],[m['initial_mlm_loss']]+[v['mlm_loss'] for v in h],color=colors[c],alpha=.6,label=c if s==42 else None)
   axes[1,col].plot([0]+B,[original[size]]+[audit[name,b] for b in B],color=colors[c],alpha=.6,label=c if s==42 else None,marker='.')
 for row in [0,1]:
  ax=axes[row,col];ax.set_title(f'NTv2 {size.upper()} (multispecies)');ax.set_xlabel('Optimizer steps' if row==0 else 'Maximum budget (selected state)');ax.grid(alpha=.15);ax.legend()
 axes[0,col].set_ylabel('Selection MLM loss (all evaluations)');axes[1,col].set_ylabel('Separate-locus audit MLM loss')
save(fig,'mlm-selection-audit')
fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
for row,size in enumerate(['100m','500m']):
 for col,k in enumerate(['log_loss','mcc','roc_auc']):
  ax=axes[row,col]
  for c in colors:
   vals=np.array([[r['models'][f'{size}-full-{c}-seed{s}-budget{b}'][k] for s in [42,43,44]] for b in B]);ax.errorbar(B,vals.mean(1),yerr=vals.std(1,ddof=1),color=colors[c],marker='o',capsize=3,label=c+' (mean +/- seed SD)')
   for i in range(3):ax.plot(B,vals[:,i],color=colors[c],alpha=.25,lw=.8)
  for key,color,ls,label in [(size+'_before','#64748b',':','Unadapted'),('kmer','#15803d','--','Oriented 1-4-mer')]:ax.axhline(r['models'][key][k],color=color,ls=ls,label=label)
  ax.set_title(f'NTv2 {size.upper()} (multispecies)');ax.set_ylabel(k.replace('_',' '));ax.set_xlabel('Maximum adaptation steps');ax.set_xticks(B,labels=[str(b) for b in B],rotation=30);ax.grid(alpha=.15)
  if row==0 and col==0:ax.legend(fontsize=7)
save(fig,'development-duration')
fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
for row,size in enumerate(['100m','500m']):
 keys=[f'{size}-budget4096-baculo-minus-general',f'{size}-budget4096-baculo-minus-before',f'{size}-baculo-4096-minus-512',f'{size}-budget4096-baculo-minus-kmer']
 for col,k in enumerate(['log_loss','mcc','roc_auc']):
  ax=axes[row,col]
  for i,key in enumerate(keys):
   v=E[key][k];p=v['delta'];lo,hi=v['interval'];ax.errorbar(p,i,xerr=[[p-lo],[hi-p]],fmt='o',color='#2563eb',capsize=3)
  ax.axvline(0,color='#64748b');ax.set_yticks(range(4),labels=['Baculo - general','Baculo - unadapted','Baculo 4096 - 512','Baculo - k-mer']);ax.invert_yaxis();ax.set_title(f'NTv2 {size.upper()} (multispecies)');ax.set_xlabel(k.replace('_',' ')+' change\n'+('Left favors baculo' if col==0 else 'Right favors baculo'));ax.grid(axis='x',alpha=.15)
save(fig,'endpoint-effects')
costs=[]
for c in colors:
 for method,array in [('full','24919261'),('lora','24919267')]:
  name=f'100m-{method}-{c}-seed42';m=r['runs'][name];job=f'{array}_{int(c=="general")}'
  costs.append(dict(run=name,trainable_parameters=m['trainable_parameters'],trainable_fraction=m['trainable_parameters']/m['parameters'],peak_allocated_gb=m['max_gpu_allocated_bytes']/1e9,training_minutes=m['training_seconds']/60,program_minutes=m['total_seconds']/60,allocated_gpu_minutes=int(next(v['ElapsedRaw'] for v in A if v['JobID']==job))/60,selected_step=m['selected_step'],audit_loss=audit[name,4096],**{k:r['models'][name+'-budget4096'][k] for k in ['log_loss','mcc','roc_auc']}))
(O/'pilot-resources.json').write_text(json.dumps(costs,indent=2)+'\n')
fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
for ax,k,label in zip(axes.flat,['trainable_fraction','peak_allocated_gb','allocated_gpu_minutes','log_loss','mcc','roc_auc'],['Trainable fraction','Peak allocated GPU GB','Allocated H100 minutes','Validation log loss','Validation MCC','Validation ROC AUC']):
 for i,v in enumerate(costs):
  ax.bar(i,v[k],color=colors['baculo' if 'baculo' in v['run'] else 'general'],alpha=1 if '-full-' in v['run'] else .5)
  ax.annotate(f"{v[k]*100:.3f}%" if k=='trainable_fraction' else f"{v[k]:.3f}",(i,v[k]),xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
 ax.margins(y=.12)
 ax.set_xticks(range(4),labels=['Full\nbaculo','LoRA\nbaculo','Full\ngeneral','LoRA\ngeneral']);ax.set_ylabel(label);ax.grid(axis='y',alpha=.15)
fig.suptitle('Exploratory NTv2 100M (multispecies), paired seed 42; unequal learning rates');save(fig,'pilot-resource-performance')

# Additional audit identity checks and complete per-seed adequacy table.
import random
from collections import Counter
corpus=Path('runs/2026-10-21/corpus')
viral=[json.loads(v) for v in (corpus/'baculo_validation.jsonl').read_text().splitlines()]
selection=[json.loads(v) for v in (corpus/'mlm_validation.jsonl').read_text().splitlines()]
locus=lambda v:(v['accession'],v['start'],v['end'])
excluded={locus(v) for v in selection}
panel=random.Random(2718).sample([v for v in viral if locus(v) not in excluded],256)
assert [{k:v for k,v in row.items() if k!='sequence'} for row in panel]==panels[0]['audit_panel']
assert hashlib.sha256(json.dumps(panel,sort_keys=True).encode()).hexdigest()==panels[0]['audit_panel_sha256']
assert dict(Counter(v['group'] for v in panel))==panels[0]['audit_groups']
assert not ({locus(v) for v in panel}&excluded)
for (name,b),loss in audit.items():assert loss==replays[name]['independent_audit_mlm_losses'][str(b)]
V['audit_panel_regenerated_and_selection_loci_excluded']=True
V['audit_csv_matches_replay']=True
(O/'verification.json').write_text(json.dumps(V,indent=2)+'\n')
lines=['| Run (NTv2 multispecies) | Budget | Selected step | Selection loss | Endpoint loss | Audit selected loss |','|---|---:|---:|---:|---:|---:|']
for name,m in r['runs'].items():
 for b in B:
  v=m['budgets'][str(b)]
  lines.append(f"| {name} | {b} | {v['selected_step']} | {v['selected_mlm_loss']:.6f} | {v['endpoint_mlm_loss']:.6f} | {audit[name,b]:.6f} |")
(O/'mlm-all-seeds.md').write_text('\n'.join(lines)+'\n')

# Record all immutable server evidence and compact replay inputs for reproducibility.
import importlib.metadata
inputs=list(R.glob('*'))+[R.parent/'jobs-completed.psv',Path(__file__)]
inputs+=list(C.rglob('*.npz'))+list(D.glob('*_readout.npz'))
inputs+=[corpus/'baculo_validation.jsonl',corpus/'mlm_validation.jsonl']
manifest=dict(script_sha256=sha(Path(__file__)),packages={k:importlib.metadata.version(k) for k in ['numpy','scikit-learn','matplotlib']},
              input_sha256={str(p):sha(p) for p in sorted(inputs) if p.is_file()},
              derived_sha256={str(p):sha(p) for p in sorted(O.iterdir()) if p.is_file() and p.name!='manifest.json'})
(O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in V.items() if k!='per_readout'},indent=2))
