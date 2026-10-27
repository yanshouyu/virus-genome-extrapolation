"""Render verified retention artifacts locally with the existing plotting environment."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path('documents/retention-2026-10-25/results')
def load(p):return json.loads(Path(p).read_text())
def main():
 assert (O/'COMPLETE').exists()
 comparisons=load(O/'paired-effects.json');summary=load(O/'summary.json');tasks=sorted(comparisons)
 families=['Promoter','Enhancer','Splice','Histone']
 pretty={'baculo-minus-original':'Baculovirus - original','general-minus-original':'General control - original'}
 display={'promoter_all':'Promoter (all)','promoter_no_tata':'Promoter (no TATA)','promoter_tata':'Promoter (TATA)','enhancers':'Enhancers','enhancers_types':'Enhancer types','splice_sites_all':'Splice sites (all)','splice_sites_acceptors':'Splice acceptors','splice_sites_donors':'Splice donors'}
 points={(v['task'],v['backbone']):float(v['mcc']) for v in csv.DictReader((O/'scores.csv').open())}
 names=['original']+[f'500m-full-{c}-seed{s}' for c in ['baculo','general'] for s in [42,43,44]]
 overall=np.zeros((1,7,3));overall[0,:,0]=[summary['Category-balanced overall']['mcc'][n] for n in names]
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
 def save(fig,name):
  for ext in ['png','pdf']:fig.savefig(O/(name+'.'+ext),dpi=180)
  plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,8),layout='constrained')
 for i,t in enumerate(tasks):
  for key,color,offset in [('baculo-minus-original','#2563eb',-.12),('general-minus-original','#d97706',.12)]:
   v=comparisons[t][key]['mcc'];lo,hi=v['interval'];point=v['delta']
   ax.plot([lo,hi],[i+offset,i+offset],color=color,marker='|',ms=5)
   corpus='baculo' if key.startswith('baculo') else 'general'
   seedpoints=[points[t,f'500m-full-{corpus}-seed{s}']-points[t,'original'] for s in [42,43,44]]
   ax.scatter(seedpoints,np.full(3,i+offset),facecolors='none',edgecolors=color,alpha=.4,s=18)
   ax.scatter([point],[i+offset],color=color,label=pretty[key] if i==0 else None,s=22)
 ax.axvline(0,color='#64748b');ax.axvline(-.05,color='#b91c1c',ls='--',label='Task loss margin (-0.05 MCC)');ax.set_yticks(range(18),labels=[display.get(t,t) for t in tasks]);ax.invert_yaxis();ax.set_xlabel('MCC change vs original (mean seed-specific scores), paired 95% intervals');ax.set_title('NTv2 500M (multispecies): fixed frozen-probe retention');ax.legend(fontsize=8);ax.grid(axis='x',alpha=.15);save(fig,'task-retention')
 fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
 labels=families+['Overall'];keys=families+['Category-balanced overall']
 for j,(condition,color) in enumerate([('baculo-minus-original','#2563eb'),('general-minus-original','#d97706')]):
  for i,key in enumerate(keys):
   v=summary[key][condition];lo,hi=v['interval'];point=v['delta'];x=i+(j-.5)*.15
   ax.plot([x,x],[lo,hi],color=color,marker='_',ms=6)
   ax.scatter([x],[point],color=color,label=pretty[condition] if i==0 else None,s=25)
 ax.axhline(0,color='#64748b');ax.axhline(-.02,color='#b91c1c',ls='--',label='Overall loss margin (-0.02 MCC)');ax.set_xticks(range(5),labels=labels);ax.set_ylabel('MCC change, paired 95% intervals');ax.set_title('NTv2 500M (multispecies): family means and category-balanced overall');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15);save(fig,'family-retention')
 old=load(Path('documents/adaptation-2026-10-24/results/development-metrics.json'))['models'];base=old['500m_before']['mcc']
 fig,ax=plt.subplots(figsize=(7,5),layout='constrained');ax.scatter([0],[0],color='#64748b',label='Original')
 for c,color,sl in [('baculo','#2563eb',slice(1,4)),('general','#d97706',slice(4,7))]:
  x=[old[f'500m-full-{c}-seed{s}-budget4096']['mcc']-base for s in [42,43,44]];yy=overall[0,sl,0]-overall[0,0,0]
  ax.scatter(x,yy,color=color,alpha=.5,label=('Baculovirus' if c=='baculo' else 'General control')+' seeds');ax.scatter([np.mean(x)],[np.mean(yy)],color=color,marker='D',s=70,label=('Baculovirus' if c=='baculo' else 'General control')+' mean')
 ax.axhline(0,color='#64748b');ax.axvline(0,color='#64748b');ax.axhline(-.02,color='#b91c1c',ls='--');ax.set_xlabel('AcMNPV validation MCC change vs original');ax.set_ylabel('NT benchmark category-balanced MCC change vs original');ax.set_title('NTv2 500M (multispecies): adaptation benefit / retention');ax.legend(fontsize=8);ax.grid(alpha=.15);save(fig,'benefit-retention')

 (O/'figure-manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),figures={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.iterdir() if p.suffix in ['.png','.pdf']}),indent=2)+'\n')
if __name__=='__main__':main()
