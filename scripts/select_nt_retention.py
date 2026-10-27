"""Training-only frozen probing selection for the revised NT benchmark."""
import argparse,hashlib,json,os,time,warnings
from pathlib import Path
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
R=Path('runs/2026-10-25-retention');CS=[.001,.01,.1,1,10]
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def tasks():return sorted(load(R/'dataset/audit.json')['tasks'])
def select(task):
 begin=time.time();out=R/'selection';out.mkdir(exist_ok=True);assert not (out/(task+'.json')).exists()
 a=np.load(R/'dataset'/task/'train.npz');y=a['labels'];chrom=a['chrom']
 assert (R/'embeddings/original/train-COMPLETE').exists()
 embeddings=np.load(R/'embeddings/original/train.npy',mmap_mode='r');meta=load(R/'embeddings/original/train-metadata.json')
 folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=42).split(np.zeros(len(y)),y,chrom));classes=np.unique(y)
 for tr,va in folds:
  assert not set(chrom[tr])&set(chrom[va]);assert np.array_equal(np.unique(y[tr]),classes) and np.array_equal(np.unique(y[va]),classes)
 candidates=[]
 for li,layer in enumerate(meta['layers']):
  x=np.asarray(embeddings[a['ids'],li,:],dtype=np.float64);probs={c:np.zeros((len(y),len(classes))) for c in CS};iters={str(c):[] for c in CS}
  for tr,va in folds:
   scaler=StandardScaler().fit(x[tr]);xt=scaler.transform(x[tr]);xv=scaler.transform(x[va])
   model=LogisticRegression(C=CS[0],solver='lbfgs',max_iter=5000,random_state=42,warm_start=True)
   for c in CS:
    model.set_params(C=c)
    with warnings.catch_warnings():
     warnings.simplefilter('error',ConvergenceWarning);model.fit(xt,y[tr])
    probs[c][va]=model.predict_proba(xv);iters[str(c)].append(int(model.n_iter_.max()))
  for c in CS:candidates.append(dict(layer=layer,layer_index=li,C=c,oof_log_loss=float(log_loss(y,probs[c],labels=classes)),iterations=iters[str(c)]))
  print(task,layer,json.dumps(candidates[-5:]),flush=True)
 best=min(candidates,key=lambda v:(v['oof_log_loss'],v['C'],v['layer']))
 result=dict(task=task,selected=best,candidates=candidates,folds=[dict(fit_chromosomes=sorted(set(chrom[tr])),validation_chromosomes=sorted(set(chrom[va])),fit_rows=len(tr),validation_rows=len(va)) for tr,va in folds],train_rows=len(y),source_sha256=sha(__file__),protocol_sha256=sha('documents/retention-2026-10-25/protocol.md'),train_index_sha256=sha(R/'dataset'/task/'train.npz'),embedding_sha256=meta['output_sha256'],seconds=time.time()-begin)
 write(out/(task+'.json'),result);print('SELECTED',json.dumps(result),flush=True)
def freeze():
 ts=tasks();settings={t:load(R/'selection'/(t+'.json')) for t in ts};assert len(settings)==18
 write(R/'selection/frozen.json',dict(tasks=settings,selection_sha256={t:sha(R/'selection'/(t+'.json')) for t in ts},test_predictions_used=False))
 (R/'selection/COMPLETE').write_text('All18 settings selected using only original-backbone train data.\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['select','freeze']);p.add_argument('--task');a=p.parse_args()
 if a.mode=='freeze':freeze()
 else:select(a.task or tasks()[int(os.environ['SLURM_ARRAY_TASK_ID'])])
