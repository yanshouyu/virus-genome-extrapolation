"""Pure scoring helpers for frozen-backbone benchmark retention."""
import numpy as np
from sklearn.metrics import log_loss,matthews_corrcoef,roc_auc_score

def decisions(prob):
 return (prob[:,1]>.5).astype(int) if prob.shape[1]==2 else prob.argmax(1)

def point_scores(y,prob):
 n=prob.shape[1]
 return np.array([matthews_corrcoef(y,decisions(prob)),log_loss(y,prob,labels=np.arange(n)),
                  roc_auc_score(y,prob[:,1]) if n==2 else roc_auc_score(y,prob,multi_class='ovr',average='macro',labels=np.arange(n))])

def paired_bootstrap_scores(y,prob,groups,weights,chunk=256):
 """Return MCC/log loss/AUC for common global block multiplicities; no fitting.

 weights[0] is the all-one point estimate. Resampling weights are shared across
 all models and tasks to preserve pairing and cross-task genomic dependence.
 """
 y=np.asarray(y);groups=np.asarray(groups);n=prob.shape[1];pred=decisions(prob)
 unique,inverse=np.unique(groups,return_inverse=True)
 w=weights[:,unique].astype(float);conf=np.zeros((len(unique),n*n))
 np.add.at(conf,(inverse,y*n+pred),1)
 c=(w@conf).reshape(-1,n,n);total=c.sum((1,2));tr=c.sum(2);pr=c.sum(1)
 cov=np.trace(c,axis1=1,axis2=2)*total-(tr*pr).sum(1)
 den=np.sqrt((total**2-(tr**2).sum(1))*(total**2-(pr**2).sum(1)))
 mcc=np.divide(cov,den,out=np.zeros_like(cov),where=den>0)
 losses=-np.log(np.clip(prob[np.arange(len(y)),y],np.finfo(float).eps,1-np.finfo(float).eps))
 loss=w@np.bincount(inverse,weights=losses,minlength=len(unique));loss=np.divide(loss,total,out=np.full_like(loss,np.nan),where=total>0)
 auc=np.zeros(len(weights));valid=np.ones(len(weights),dtype=bool)
 for label in ([1] if n==2 else range(n)):
  order=np.argsort(prob[:,label],kind='stable');pp=prob[order,label];positive=(y[order]==label)
  _,begin,count=np.unique(pp,return_index=True,return_counts=True);ends=begin+count-1
  values=np.empty(len(weights))
  for start in range(0,len(weights),chunk):
   rw=weights[start:start+chunk,groups[order]].astype(float)
   pos=rw*positive;neg=rw*~positive
   ptotal=pos.sum(1);ntotal=neg.sum(1)
   cumulative=np.cumsum(neg,axis=1);before=np.concatenate([np.zeros((len(rw),1)),cumulative[:,ends[:-1]]],axis=1)
   within=cumulative[:,ends]-before;possum=np.add.reduceat(pos,begin,axis=1)
   numerator=(possum*(before+.5*within)).sum(1);denom=ptotal*ntotal
   values[start:start+len(rw)]=np.divide(numerator,denom,out=np.full(len(rw),np.nan),where=denom>0)
  valid&=np.isfinite(values);auc+=values/(1 if n==2 else n)
 result=np.stack([mcc,loss,auc],axis=1);result[~valid]=np.nan
 return result
