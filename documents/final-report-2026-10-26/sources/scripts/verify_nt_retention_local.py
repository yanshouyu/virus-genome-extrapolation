"""Independently recompute benchmark metrics and paired intervals from compact probabilities."""
import csv,hashlib,json,re
from collections import defaultdict
from pathlib import Path
import numpy as np
from baculo_adapt.retention import point_scores,paired_bootstrap_scores
O=Path('documents/retention-2026-10-25/results')
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert (O/'COMPLETE').exists()
 manifest=load(O/'manifest.json')
 pretest=load(O.parent/'pre-test-freeze-verification.json')
 assert sha(O/'frozen-readout-settings.json')==pretest['frozen_settings_sha256']
 assert sha(O.parent/'protocol.md')==pretest['protocol_sha256']
 for path,h in manifest['files'].items():assert sha(O/path)==h,path
 effects=load(O/'paired-effects.json');summary=load(O/'summary.json');tasks=sorted(effects);weights=np.load(O/'bootstrap-weights.npy');all_scores=[]
 # Independently rebuild connected genomic blocks with graph traversal rather
 # than the server's union-find implementation.
 graph=defaultdict(set);coordinate_rows={};saved_groups={}
 for task in tasks:
  a=np.load(O/'predictions'/(task+'.npz'));coordinate_rows[task]=[];saved_groups[task]=a['groups']
  for name in a['names']:
   m=re.match(r'^(chr[^:]+):(\d+)-(\d+)',str(name));assert m,name
   chrom,start,end=m.groups();start=int(start);end=int(end)
   bins=[(chrom,b) for b in range(start//1000000,(end-1)//1000000+1)]
   coordinate_rows[task].append(bins[0])
   for b in bins:graph[b]
   for x,y in zip(bins,bins[1:]):graph[x].add(y);graph[y].add(x)
 components=[];seen=set()
 for node in sorted(graph):
  if node in seen:continue
  todo=[node];component=set()
  while todo:
   x=todo.pop()
   if x in component:continue
   component.add(x);seen.add(x);todo.extend(graph[x]-component)
  components.append(component)
 components.sort(key=min);group_index={node:i for i,component in enumerate(components) for node in component}
 assert [list(min(c)) for c in components]==load(O/'bootstrap-metadata.json')['blocks']
 for task in tasks:np.testing.assert_array_equal([group_index[v] for v in coordinate_rows[task]],saved_groups[task])
 assert np.all(weights[0]==1) and np.all(weights.sum(1)==len(components))
 regenerated=np.vstack([np.ones(len(components),dtype=np.uint16),np.random.default_rng(2718).multinomial(len(components),np.ones(len(components))/len(components),size=20000).astype(np.uint16)])
 np.testing.assert_array_equal(weights,regenerated)
 table={(v['task'],v['backbone']):v for v in csv.DictReader((O/'scores.csv').open())};names=None
 for task in tasks:
  a=np.load(O/'predictions'/(task+'.npz'));names=list(a['backbones']);assert len(names)==7
  scores=[]
  for j,name in enumerate(names):
   point=point_scores(a['labels'],a['probabilities'][j]);np.testing.assert_allclose(point,[float(table[task,name][m]) for m in ['mcc','log_loss','roc_auc']],atol=1e-12,rtol=0)
   b=paired_bootstrap_scores(a['labels'],a['probabilities'][j],a['groups'],weights);np.testing.assert_allclose(b[0],point,atol=1e-12,rtol=0);scores.append(b)
  all_scores.append(np.stack(scores,axis=1));print('Recomputed',task,flush=True)
 scores=np.stack(all_scores);valid=np.isfinite(scores[:,1:]).all((0,2,3));scores=scores[:,np.r_[True,valid]]
 for i,task in enumerate(tasks):
  for key,left,right in [('baculo-minus-original',slice(1,4),slice(0,1)),('general-minus-original',slice(4,7),slice(0,1)),('baculo-minus-general',slice(1,4),slice(4,7))]:
   delta=scores[i,:,left,:].mean(1)-scores[i,:,right,:].mean(1)
   for k,m in enumerate(['mcc','log_loss','roc_auc']):
    e=effects[task][key][m];np.testing.assert_allclose(delta[0,k],e['delta'],atol=1e-10,rtol=0);np.testing.assert_allclose(np.quantile(delta[1:,k],[.025,.975]),e['interval'],atol=1e-10,rtol=0)
   np.testing.assert_allclose(np.quantile(delta[1:,0],.05/18),effects[task][key]['mcc']['simultaneous_one_sided95_lower'],atol=1e-10,rtol=0)
 def family(t):return 'Promoter' if t.startswith('promoter') else ('Enhancer' if t.startswith('enhancer') else ('Splice' if t.startswith('splice') else 'Histone'))
 families=['Promoter','Enhancer','Splice','Histone'];fs=np.stack([scores[[i for i,t in enumerate(tasks) if family(t)==f]].mean(0) for f in families]);overall=fs.mean(0)
 for label,sc in list(zip(families,fs))+[('Category-balanced overall',overall)]:
  for key,left,right in [('baculo-minus-original',slice(1,4),slice(0,1)),('general-minus-original',slice(4,7),slice(0,1)),('baculo-minus-general',slice(1,4),slice(4,7))]:
   delta=sc[:,left,0].mean(1)-sc[:,right,0].mean(1);e=summary[label][key]
   np.testing.assert_allclose(delta[0],e['delta'],atol=1e-10,rtol=0);np.testing.assert_allclose(np.quantile(delta[1:],[.025,.975]),e['interval'],atol=1e-10,rtol=0)
 result=dict(tasks=18,readouts=126,all_saved_metrics_recomputed=True,all_paired_intervals_and_simultaneous_task_bounds_reproduced=True,all_family_and_overall_intervals_reproduced=True,common_valid_resamples=int(valid.sum()),all_server_manifest_hashes_match=True,pre_test_frozen_settings_match=True,genomic_blocks_independently_reconstructed=True,bootstrap_weights_regenerated=True,source_sha256=sha(__file__),new_inference=False)
 (O.parent/'local-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
