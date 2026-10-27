"""Fetch and audit the authors' frozen benchmark. No model or AcMNPV data access."""
import argparse, csv, hashlib, json, re, time
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path('runs/2026-10-25-retention'); REV='851f9946252e90c665cdb3cc3eedb78f1f26197c'
REPO='InstaDeepAI/nucleotide_transformer_downstream_tasks_revised'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['fetch','prepare']);a=p.parse_args()
 data=ROOT/'dataset';data.mkdir(parents=True,exist_ok=True)
 if a.mode=='fetch':
  from huggingface_hub import snapshot_download
  snapshot_download(REPO,repo_type='dataset',revision=REV,allow_patterns=['*/*.parquet','README.md'],local_dir=data/'raw',max_workers=4)
  write(data/'download.json',dict(repo=REPO,revision=REV,files={str(q.relative_to(data/'raw')):sha(q) for q in (data/'raw').rglob('*') if q.is_file() and '.cache' not in q.parts}));return
 import pyarrow.parquet as pq
 assert not (data/'COMPLETE').exists()
 seqs={'train':[],'test':[]};indices={'train':{},'test':{}};audit={};global_canonical={'train':set(),'test':set()}
 raw=data/'raw';tasks=sorted(q.name for q in raw.iterdir() if q.is_dir() and (q/'train.parquet').exists());assert len(tasks)==18
 trans=str.maketrans('ACGTN','TGCAN')
 arrays={}
 for task in tasks:
  audit[task]={};canonical={};chrs={};coords={}
  for split in ['train','test']:
   table=pq.read_table(raw/task/(split+'.parquet')).to_pylist();ids=[];labels=[];names=[];chrom=[];starts=[];ends=[];canonical[split]=set()
   for row in table:
    s=row['sequence'];assert s==s.upper() and set(s)<=set('ACGTN'),(task,set(s))
    name=row['name'];match=re.match(r'^(chr[^:]+):(\d+)-(\d+)',name);assert match,(task,name)
    c,start,end=match.groups();start=int(start);end=int(end);assert start<end
    if s not in indices[split]:indices[split][s]=len(seqs[split]);seqs[split].append(s)
    ids.append(indices[split][s]);labels.append(int(row['label']));names.append(name);chrom.append(c);starts.append(start);ends.append(end)
    canonical[split].add(min(s,s.translate(trans)[::-1]))
   global_canonical[split].update(canonical[split]);chrs[split]=set(chrom);coords[split]=list(zip(chrom,starts,ends))
   dest=data/task;dest.mkdir(exist_ok=True)
   np.savez_compressed(dest/(split+'.npz'),ids=np.array(ids),labels=np.array(labels),names=np.array(names),chrom=np.array(chrom),start=np.array(starts),end=np.array(ends))
   audit[task][split]=dict(rows=len(table),unique_sequences=len(set(ids)),class_counts=dict(Counter(labels)),chromosome_counts=dict(Counter(chrom)),length_counts=dict(Counter(map(len,[v['sequence'] for v in table]))))
  common=chrs['train']&chrs['test'];overlap=0
  for c in common:
   tr=sorted((x,y) for cc,x,y in coords['train'] if cc==c);te=sorted((x,y) for cc,x,y in coords['test'] if cc==c);j=0
   for x,y in te:
    while j<len(tr) and tr[j][1]<=x:j+=1
    if j<len(tr) and tr[j][0]<y:overlap+=1
  audit[task]['shared_chromosomes']=sorted(common);audit[task]['test_windows_overlapping_train']=overlap
  audit[task]['shared_exact_or_rc_sequences']=len(canonical['train']&canonical['test'])
  print(task,json.dumps(audit[task]),flush=True)
 for split in seqs:
  with (data/(split+'-sequences.jsonl')).open('w') as f:
   for s in seqs[split]:f.write(json.dumps(s)+'\n')
 report=dict(repo=REPO,revision=REV,tasks=audit,unique_sequences={k:len(v) for k,v in seqs.items()},global_shared_exact_or_rc_sequences=len(global_canonical['train']&global_canonical['test']),source_sha256=sha(__file__),protocol_sha256=sha('documents/retention-2026-10-25/protocol.md'))
 write(data/'audit.json',report);(data/'COMPLETE').write_text('Official benchmark downloaded, indexed and audited; no predictions.\n')
if __name__=='__main__':main()
