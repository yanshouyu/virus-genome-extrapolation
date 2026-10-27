import numpy as np
from baculo_adapt.retention import paired_bootstrap_scores,point_scores

def test_block_bootstrap_matches_explicit_replicates_binary_and_multiclass():
 for n in [2,3]:
  rng=np.random.default_rng(42+n);y=np.arange(24)%n;p=rng.uniform(.1,1,(24,n));p/=p.sum(1,keepdims=True)
  p[1]=p[0]  # Explicit ties, including potentially different labels.
  groups=np.arange(24)//3
  weights=np.vstack([np.ones(8,dtype=int),rng.multinomial(8,np.ones(8)/8,size=17)])
  actual=paired_bootstrap_scores(y,p,groups,weights,chunk=4)
  for i,w in enumerate(weights):
   ix=np.repeat(np.arange(24),w[groups])
   np.testing.assert_allclose(actual[i],point_scores(y[ix],p[ix]),atol=1e-12)

def test_fixed_binary_threshold_and_missing_class_bootstrap():
 y=np.array([0,0,1,1]);p=np.array([[.5,.5],[.3,.7],[.5,.5],[.1,.9]])
 scores=paired_bootstrap_scores(y,p,np.array([0,0,1,1]),np.array([[1,1],[2,0]]))
 np.testing.assert_allclose(scores[0],point_scores(y,p),atol=1e-12)
 assert np.isnan(scores[1]).all()
