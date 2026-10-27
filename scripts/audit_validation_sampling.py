"""Describe sampling context on training/validation only; no model fitting."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
from Bio import SeqIO
from baculo_tss_probe.splitting import coordinates, read_known_tss, label_violations, cross_split_violations

root=Path('data/processed/clean_v2')
rows={s:list(csv.DictReader((root/(s+'.csv')).open())) for s in ['train','validation']}
gb=Path('data/raw/baculovirus/Baculoviridae/NC_001623.1.gb')
record=SeqIO.read(gb,'genbank')
sites=read_known_tss('data/raw/baculovirus/Baculovirus_TSS.csv')
cds={st:[] for st in ['+','-']}
for f in record.features:
    if f.type=='CDS':
        strand='-' if f.location.strand==-1 else '+'
        cds[strand].extend((int(part.start),int(part.end)) for part in f.location.parts)
result={'source_sha256':hashlib.sha256(gb.read_bytes()).hexdigest(),'splits':{},
        'cross_split_violations':cross_split_violations(rows,'data/interim/tss_seq_range.csv')}
for split,records in rows.items():
    detail=[]
    assert not label_violations(records,sites)
    for row in records:
        start,end,strand=coordinates(row)
        expected=record.seq[start:end]
        if strand=='-': expected=expected.reverse_complement()
        assert str(expected).upper()==row['sequence'].upper(),row['name']
        covered=set()
        for a,b in cds[strand]: covered.update(range(max(a,start),min(b,end)))
        tss_count=sum(start<=site<end and st==strand for site,st in sites)
        detail.append(dict(name=row['name'],label=int(row['label']),strand=strand,
                           cds_fraction=len(covered)/(end-start),recorded_tss_count=tss_count,
                           gc_fraction=sum(row['sequence'].count(base) for base in 'GC')/len(row['sequence'])))
    result['splits'][split]={}
    for label in [0,1]:
        subset=[r for r in detail if r['label']==label]
        result['splits'][split][str(label)]=dict(n=len(subset),
            overlapping_same_strand_cds=sum(r['cds_fraction']>0 for r in subset),
            median_cds_fraction=float(np.median([r['cds_fraction'] for r in subset])),
            mean_gc_fraction=float(np.mean([r['gc_fraction'] for r in subset])),
            strands={st:sum(r['strand']==st for r in subset) for st in ['+','-']},
            multiple_recorded_tss=sum(r['recorded_tss_count']>1 for r in subset))
    out=Path('documents/experiments-2026-10-20')
    with (out/(split+'_sampling.csv')).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(detail[0])); writer.writeheader(); writer.writerows(detail)
assert not result['cross_split_violations']
(out/'sampling-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
