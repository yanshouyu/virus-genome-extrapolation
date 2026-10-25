"""Frozen readout fitting and paired block bootstrap on the official NT test sets."""
import argparse,hashlib,json,os,re,time,warnings
from pathlib import Path
import numpy as np
from scipy.special import expit,softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from baculo_adapt.retention import point_scores,paired_bootstrap_scores
R=Path('runs/2026-10-25-retention')
NAMES=['original']+[f'500m-full-{c}-seed{s}' for c in ['baculo','general'] for s in [42,43,44]]
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def tasks():return sorted(load(R/'dataset/audit.json')['tasks'])
def make_blocks():
 out=R/'bootstrap';out.mkdir(exist_ok=True);parent={};arrays={}
 def find(x):
  if x not in parent:parent[x]=x
  if parent[x]!=x:parent[x]=find(parent[x])
  return parent[x]
 def union(x,y):
  a,b=find(x),find(y)
  if a!=b:parent[max(a,b)]=min(a,b)
 for task in tasks():
  a=np.load(R/'dataset'/task/'test.npz');arrays[task]={k:a[k] for k in ['chrom','start','end']}
  for c,s,e in zip(a['chrom'],a['start'],a['end']):
   left=(str(c),int(s)//1000000);find(left)
   for b in range(int(s)//1000000,int(e-1)//1000000+1):union(left,(str(c),b))
 labels=sorted(set(find(k) for k in parent));index={k:i for i,k in enumerate(labels)}
 for task,a in arrays.items():
  g=np.array([index[find((str(c),int(s)//1000000))] for c,s in zip(a['chrom'],a['start'])]);np.save(out/(task+'-groups.npy'),g)
 rng=np.random.default_rng(2718);w=np.vstack([np.ones(len(labels),dtype=np.uint16),rng.multinomial(len(labels),np.ones(len(labels))/len(labels),size=20000).astype(np.uint16)])
 np.save(out/'weights.npy',w)
 write(out/'metadata.json',dict(blocks=[list(v) for v in labels],count=len(labels),resamples=20000,seed=2718,method='Common global chromosome/1Mb genomic-block multiplicities; bins crossed by windows are joined',source_sha256=sha(__file__),weights_sha256=sha(out/'weights.npy')))
 (out/'COMPLETE').write_text('Paired blocks fixed from coordinates, no predictions.\n')
 print('BLOCKS',len(labels),flush=True)
def readout(task):
 started=time.time();out=R/'readouts'/task;out.mkdir(parents=True,exist_ok=False)
 frozen=load(R/'selection/frozen.json');setting=frozen['tasks'][task]['selected'];assert (R/'selection/COMPLETE').exists() and (R/'bootstrap/COMPLETE').exists()
 a={s:np.load(R/'dataset'/task/(s+'.npz')) for s in ['train','test']};y={s:a[s]['labels'] for s in a};li=setting['layer_index'];classes=np.unique(y['train']);assert np.array_equal(classes,np.arange(len(classes)))
 weights=np.load(R/'bootstrap/weights.npy',mmap_mode='r');groups=np.load(R/'bootstrap'/(task+'-groups.npy'))
 scores=[];preds=[];evidence={}
 for name in NAMES:
  x={};meta={}
  for s in a:
   root=R/'embeddings'/name;assert (root/(s+'-COMPLETE')).exists()
   meta[s]=load(root/(s+'-metadata.json'));assert meta[s]['layers'][li]==setting['layer']
   emb=np.load(root/(s+'.npy'),mmap_mode='r');x[s]=np.asarray(emb[a[s]['ids'],li,:],dtype=np.float64)
  scaler=StandardScaler().fit(x['train']);xt=scaler.transform(x['train']);xv=scaler.transform(x['test'])
  model=LogisticRegression(C=setting['C'],solver='lbfgs',max_iter=5000,random_state=42)
  with warnings.catch_warnings():
   warnings.simplefilter('error',ConvergenceWarning);model.fit(xt,y['train'])
  prob=model.predict_proba(xv);point=point_scores(y['test'],prob)
  z=xv@model.coef_.T+model.intercept_;replay=np.column_stack([1-expit(z[:,0]),expit(z[:,0])]) if len(classes)==2 else softmax(z,axis=1)
  delta=float(abs(replay-prob).max());assert delta<1e-12
  np.testing.assert_allclose(scaler.mean_,x['train'].mean(0),rtol=1e-12,atol=1e-12)
  np.savez_compressed(out/(name+'-readout.npz'),mean=scaler.mean_,scale=scaler.scale_,coefficients=model.coef_,intercept=model.intercept_,classes=model.classes_)
  boot=paired_bootstrap_scores(y['test'],prob,groups,weights);np.testing.assert_allclose(boot[0],point,atol=1e-12)
  scores.append(boot);preds.append(prob);evidence[name]=dict(mcc=float(point[0]),log_loss=float(point[1]),roc_auc=float(point[2]),iterations=int(model.n_iter_.max()),probability_replay_max_abs=delta,train_only_scaler=True,embeddings={s:dict(output_sha256=m['output_sha256'],state_sha256=m['state_sha256'],repeat_max_abs=m['repeat_max_abs']) for s,m in meta.items()})
  print(task,name,point.tolist(),time.time()-started,flush=True)
 np.savez_compressed(out/'predictions.npz',probabilities=np.stack(preds),labels=y['test'],names=a['test']['names'],backbones=np.array(NAMES),groups=groups)
 allscores=np.stack(scores,axis=1);np.savez_compressed(out/'bootstrap-scores.npz',scores=allscores)
 write(out/'metrics.json',dict(task=task,setting=setting,backbones=NAMES,models=evidence,train_rows=len(y['train']),test_rows=len(y['test']),blocks=len(np.unique(groups)),valid_common_resamples=int(np.isfinite(allscores[1:]).all((1,2)).sum()),seconds=time.time()-started,source_sha256={str(q):sha(q) for q in [Path(__file__),Path('src/baculo_adapt/retention.py')]},frozen_selection_sha256=sha(R/'selection/frozen.json'),bootstrap_sha256=sha(R/'bootstrap/weights.npy'),dataset_train_sha256=sha(R/'dataset'/task/'train.npz'),dataset_test_sha256=sha(R/'dataset'/task/'test.npz')))
 (out/'COMPLETE').write_text('Fixed frozen-backbone test readouts, coefficient replay and paired bootstrap complete.\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['blocks','readout']);p.add_argument('--task');a=p.parse_args()
 if a.mode=='blocks':make_blocks()
 else:readout(a.task or tasks()[int(os.environ['SLURM_ARRAY_TASK_ID'])])
