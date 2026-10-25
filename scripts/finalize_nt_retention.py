"""Finalize all18 frozen-backbone retention tasks; no fitting or inference."""
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit,softmax
from baculo_adapt.retention import point_scores
R=Path('runs/2026-10-25-retention');O=Path('documents/retention-2026-10-25/results')
NAMES=['original']+[f'500m-full-{c}-seed{s}' for c in ['baculo','general'] for s in [42,43,44]]
METRICS=['mcc','log_loss','roc_auc']
def load(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def family(t):
 if t.startswith('promoter'):return 'Promoter'
 if t.startswith('enhancer'):return 'Enhancer'
 if t.startswith('splice'):return 'Splice'
 return 'Histone'
def effect(values,margin):
 finite=np.isfinite(values);boot=values[1:][finite[1:]];point=float(values[0]);lo,hi=np.quantile(boot,[.025,.975]);lower,upper=np.quantile(boot,[.05,.95])
 return dict(delta=point,interval=[float(lo),float(hi)],one_sided95_lower=float(lower),one_sided95_upper=float(upper),margin=margin,conclusion='preservation supported' if lower>-margin else ('material degradation supported' if upper<-margin else 'unresolved'))
def main():
 assert not (O/'COMPLETE').exists();O.mkdir(parents=True,exist_ok=True)
 audit=load(R/'dataset/audit.json');tasks=sorted(audit['tasks']);assert len(tasks)==18
 assert (R/'REPLAY_COMPLETE').exists()
 fresh=load(R/'fresh-replay.json');assert fresh['max_abs']==0 and len(fresh['backbones'])==7
 for path,digest in fresh['source_sha256'].items():assert sha(Path(path.split('virus-genome-extrapolation/')[-1]))==digest
 frozen=load(R/'selection/frozen.json');assert set(frozen['tasks'])==set(tasks) and (R/'selection/COMPLETE').exists()
 download=load(R/'dataset/download.json')
 assert audit['source_sha256']==sha('scripts/prepare_nt_retention.py')
 assert audit['protocol_sha256']==sha('documents/retention-2026-10-25/protocol.md')
 for task,info in frozen['tasks'].items():
  assert sha(R/'selection'/(task+'.json'))==frozen['selection_sha256'][task]
  assert info['source_sha256']==sha('scripts/select_nt_retention.py')
  assert info['train_index_sha256']==sha(R/'dataset'/task/'train.npz')
  assert len(info['candidates'])==20
  assert min(info['candidates'],key=lambda v:(v['oof_log_loss'],v['C'],v['layer']))==info['selected']
  assert all(not set(f['fit_chromosomes'])&set(f['validation_chromosomes']) for f in info['folds'])
 for path,digest in download['files'].items():assert sha(R/'dataset/raw'/path)==digest,path
 sources={};embedding_checks={}
 for name in NAMES:
  embedding_checks[name]={}
  for split in ['train','test']:
   root=R/'embeddings'/name;assert (root/(split+'-COMPLETE')).exists();m=load(root/(split+'-metadata.json'))
   assert m['repeat_max_abs']==0 and m['source']['revision']=='06615c1660c892fc199840c18123f8385b3542a8'
   assert sha(root/(split+'.npy'))==m['output_sha256'];assert sha(R/'dataset'/(split+'-sequences.jsonl'))==m['dataset_sha256']
   sources.update(m['source_sha256']);embedding_checks[name][split]=m
 scores=[];models={};csvrows=[];max_replay=0
 for task in tasks:
  root=R/'readouts'/task;assert (root/'COMPLETE').exists();m=load(root/'metrics.json');sources.update(m['source_sha256'])
  assert m['frozen_selection_sha256']==sha(R/'selection/frozen.json');assert m['bootstrap_sha256']==sha(R/'bootstrap/weights.npy')
  assert m['setting']==frozen['tasks'][task]['selected'];models[task]=m
  a=np.load(R/'dataset'/task/'test.npz');train=np.load(R/'dataset'/task/'train.npz');pred=np.load(root/'predictions.npz');bs=np.load(root/'bootstrap-scores.npz')['scores'];assert bs.shape==(20001,7,3)
  np.testing.assert_array_equal(pred['labels'],a['labels']);np.testing.assert_array_equal(pred['names'],a['names']);assert list(pred['backbones'])==NAMES
  li=m['setting']['layer_index']
  for j,name in enumerate(NAMES):
   # Independently replay full test predictions and verify full train scaling.
   xt=np.load(R/'embeddings'/name/'train.npy',mmap_mode='r')[train['ids'],li,:].astype(float)
   xv=np.load(R/'embeddings'/name/'test.npy',mmap_mode='r')[a['ids'],li,:].astype(float)
   co=np.load(root/(name+'-readout.npz'));np.testing.assert_allclose(co['mean'],xt.mean(0),rtol=1e-12,atol=1e-12)
   sd=xt.std(0);sd[sd==0]=1;np.testing.assert_allclose(co['scale'],sd,rtol=1e-10,atol=1e-12)
   z=((xv-co['mean'])/co['scale'])@co['coefficients'].T+co['intercept']
   actual=np.column_stack([1-expit(z[:,0]),expit(z[:,0])]) if z.shape[1]==1 else softmax(z,axis=1)
   delta=float(abs(actual-pred['probabilities'][j]).max());assert delta<1e-12;max_replay=max(max_replay,delta)
   point=point_scores(a['labels'],pred['probabilities'][j]);np.testing.assert_allclose(point,bs[0,j],atol=1e-12)
   csvrows.append(dict(task=task,family=family(task),backbone=name,layer=m['setting']['layer'],C=m['setting']['C'],**dict(zip(METRICS,map(float,point)))))
  scores.append(bs)
 for path,digest in sources.items():assert sha(Path(path.split('virus-genome-extrapolation/')[-1]))==digest,path
 scores=np.stack(scores,axis=0);valid=np.isfinite(scores[:,1:]).all((0,2,3));assert valid.sum()>=19000
 scores=scores[:,np.r_[True,valid]]
 comparisons={};taskrows=[]
 for i,task in enumerate(tasks):
  comparisons[task]={}
  for key,left,right in [('baculo-minus-original',slice(1,4),slice(0,1)),('general-minus-original',slice(4,7),slice(0,1)),('baculo-minus-general',slice(1,4),slice(4,7))]:
   delta=scores[i,:,left,:].mean(1)-scores[i,:,right,:].mean(1)
   e={metric:effect(delta[:,k],.05 if k==0 else 0) for k,metric in enumerate(METRICS)}
   for metric in METRICS[1:]:e[metric]={k:e[metric][k] for k in ['delta','interval']}
   e['mcc']['simultaneous_one_sided95_lower']=float(np.quantile(delta[1:,0],.05/18));comparisons[task][key]=e
  taskrows.append(dict(task=task,family=family(task),**{f'{k}_mcc':float(scores[i,0,sl,0].mean()) for k,sl in [('original',slice(0,1)),('baculo',slice(1,4)),('general',slice(4,7))]},**comparisons[task]['baculo-minus-original']['mcc']))
 families=['Promoter','Enhancer','Splice','Histone'];famscores=np.stack([scores[[i for i,t in enumerate(tasks) if family(t)==f]].mean(0) for f in families]);overall=famscores.mean(0)
 summary={}
 for label,sc in list(zip(families,famscores))+[('Category-balanced overall',overall)]:
  summary[label]={}
  for key,left,right in [('baculo-minus-original',slice(1,4),slice(0,1)),('general-minus-original',slice(4,7),slice(0,1)),('baculo-minus-general',slice(1,4),slice(4,7))]:
   delta=sc[:,left,:].mean(1)-sc[:,right,:].mean(1);summary[label][key]=effect(delta[:,0],.02)
  summary[label]['mcc']={name:float(sc[0,j,0]) for j,name in enumerate(NAMES)}
 all_tasks=all(comparisons[t]['baculo-minus-original']['mcc']['simultaneous_one_sided95_lower']>-.05 for t in tasks)
 verification=dict(tasks=18,backbones=7,readouts=126,all_embedding_hashes_match=True,all_raw_dataset_hashes_match=True,source_hashes_checked=len(sources),full_training_scalers_verified=True,fresh_process_embedding_max_abs=fresh['max_abs'],full_test_probability_replay_max_abs=max_replay,all_metrics_recomputed=True,paired_common_bootstrap_resamples=int(valid.sum()),frozen_settings_sha256=sha(R/'selection/frozen.json'),acmnpv_test_opened=False,new_backbone_training=False,all18_task_preservation_supported=all_tasks)
 write(O/'verification.json',verification);write(O/'summary.json',summary);write(O/'paired-effects.json',comparisons);write(O/'readout-evidence.json',models);write(O/'embedding-evidence.json',embedding_checks)
 for filename,records in [('scores.csv',csvrows),('task-retention.csv',taskrows)]:
  with (O/filename).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
 import shutil
 (O/'predictions').mkdir(exist_ok=True)
 for task in tasks:shutil.copy2(R/'readouts'/task/'predictions.npz',O/'predictions'/(task+'.npz'))
 shutil.copy2(R/'bootstrap/weights.npy',O/'bootstrap-weights.npy')
 for filename in ['audit.json','download.json']:shutil.copy2(R/'dataset'/filename,O/('dataset-'+filename))
 shutil.copy2(R/'selection/frozen.json',O/'frozen-readout-settings.json');shutil.copy2(R/'bootstrap/metadata.json',O/'bootstrap-metadata.json');shutil.copy2(R/'fresh-replay.json',O/'fresh-replay.json')
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
 def save(fig,name):
  for ext in ['png','pdf']:fig.savefig(O/(name+'.'+ext),dpi=180)
  plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,8),layout='constrained')
 for i,t in enumerate(tasks):
  for key,color,offset in [('baculo-minus-original','#2563eb',-.12),('general-minus-original','#d97706',.12)]:
   v=comparisons[t][key]['mcc'];lo,hi=v['interval'];point=v['delta'];ax.errorbar(point,i+offset,xerr=[[point-lo],[hi-point]],fmt='o',color=color,capsize=2,label=key if i==0 else None,ms=4)
 ax.axvline(0,color='#64748b');ax.axvline(-.05,color='#b91c1c',ls='--',label='Proposed task loss margin (-.05)');ax.set_yticks(range(18),labels=tasks);ax.invert_yaxis();ax.set_xlabel('MCC change vs original (mean seed-specific scores), paired95% intervals');ax.set_title('NTv2 500M (multispecies): fixed frozen-probe retention');ax.legend(fontsize=8);ax.grid(axis='x',alpha=.15);save(fig,'task-retention')
 fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
 labels=families+['Overall'];keys=families+['Category-balanced overall']
 for j,(condition,color) in enumerate([('baculo-minus-original','#2563eb'),('general-minus-original','#d97706')]):
  for i,key in enumerate(keys):
   v=summary[key][condition];lo,hi=v['interval'];point=v['delta'];ax.errorbar(i+(j-.5)*.15,point,yerr=[[point-lo],[hi-point]],fmt='o',color=color,capsize=3,label=condition if i==0 else None)
 ax.axhline(0,color='#64748b');ax.axhline(-.02,color='#b91c1c',ls='--',label='Overall loss margin (-.02)');ax.set_xticks(range(5),labels=labels);ax.set_ylabel('MCC change, paired95% intervals');ax.set_title('NTv2 500M (multispecies): family means and category-balanced overall');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15);save(fig,'family-retention')
 old=load(Path('documents/adaptation-2026-10-24/results/development-metrics.json'))['models'];base=old['500m_before']['mcc']
 fig,ax=plt.subplots(figsize=(7,5),layout='constrained');ax.scatter([0],[0],color='#64748b',label='Original')
 for c,color,sl in [('baculo','#2563eb',slice(1,4)),('general','#d97706',slice(4,7))]:
  x=[old[f'500m-full-{c}-seed{s}-budget4096']['mcc']-base for s in [42,43,44]];yy=overall[0,sl,0]-overall[0,0,0]
  ax.scatter(x,yy,color=color,alpha=.5,label=c+' seeds');ax.scatter([np.mean(x)],[np.mean(yy)],color=color,marker='D',s=70,label=c+' mean')
 ax.axhline(0,color='#64748b');ax.axvline(0,color='#64748b');ax.axhline(-.02,color='#b91c1c',ls='--');ax.set_xlabel('AcMNPV validation MCC change vs original');ax.set_ylabel('NT benchmark category-balanced MCC change vs original');ax.set_title('NTv2 500M (multispecies): adaptation benefit / retention');ax.legend(fontsize=8);ax.grid(alpha=.15);save(fig,'benefit-retention')
 lines=['# NTv2 500M multispecies: general benchmark retention','',
 'All18 revised NT benchmark tasks evaluated with frozen backbones, train-only selected logistic probes and the existing4096-budget selected adaptation states. No backbone training or AcMNPV test inference. These are fixed-probe retention results, not a replication of the paper\'s fine-tuned leaderboard.','',
 '| Task | Original MCC | Baculo MCC (mean seeds) | General MCC (mean seeds) | Baculo-original [95% CI] | Preservation at .05 margin |','|---|---:|---:|---:|---:|---|']
 for v in taskrows:
  lo,hi=v['interval'];lines.append(f"| {v['task']} | {v['original_mcc']:.4f} | {v['baculo_mcc']:.4f} | {v['general_mcc']:.4f} | {v['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}] | {v['conclusion']} |")
 e=summary['Category-balanced overall']['baculo-minus-original'];lo,hi=e['interval']
 lines+=['',f"Category-balanced mean MCC change, baculovirus-original: {e['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}]. One-sided95% lower bound {e['one_sided95_lower']:+.4f}, proposed margin -.02: **{e['conclusion']}**.",f'All18 task preservation at -.05 using multiplicity-adjusted one-sided bounds: **{all_tasks}**. Marginal task intervals cannot alone establish an all-task guarantee.','',
 '![Task retention](task-retention.png)','','![Family retention](family-retention.png)','','![Benefit and retention](benefit-retention.png)','',
 'Readouts use mean-pooled embeddings; layer/C were selected only on original training features with chromosome-grouped folds and then fixed across conditions. Classifiers/scalers were independently fitted on each training representation. Scores are means of seed-specific metrics, not ensemble predictions. Binary threshold>.5; multiclass argmax. MCC is primary; scores.csv also retains log loss and macro OvR ROC AUC.','',
 f"Uncertainty uses {int(valid.sum())} valid paired resamples of common chromosome/1Mb genomic blocks (20000 requested), with overlapping windows joined across bins and common weights across related tasks. Intervals condition on these models, readouts and supplied human benchmark. They do not supply independent species replication or guarantee general retention beyond this benchmark. Task-family means are weighted equally in the overall score; the ten histone tasks do not dominate.",'',
 'The official chromosome-held-out splits are preserved. No train/test coordinate overlap or exact/RC matches were found. Human genome pretraining exposure is expected; benchmark test chromosomes are held out from supervised probe fitting, not necessarily from original pretraining. The small promoter-TATA test and correlated task definitions can limit precision. AUC/log loss disagreements must be examined alongside MCC before a practical conclusion.','',
 'verification.json records independent full-test coefficient replay, full-training scaler checks, metric recomputation, source/dataset/embedding hashes and frozen settings. Large vectors and model states stay on Nibi. The parent protocol/execution files retain design, jobs, failures and timing. Figures are exported as PNG/PDF.','']
 (O/'README.md').write_text('\n'.join(lines));write(O/'manifest.json',dict(source_sha256=sha(__file__),files={str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file() and p.name!='manifest.json'}));(O/'COMPLETE').write_text('All18 tasks and126 readouts independently checked.\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
