"""Audit positive-window overlap with its assigned ORF versus other GenBank CDS."""
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter
from Bio import SeqIO


def intersection(a,b):
    return max(0,min(a[1],b[1])-max(a[0],b[0]))


def main():
    out=Path('documents/cds-audit-2026-10-21');out.mkdir(parents=True,exist_ok=True)
    table=Path('data/interim/tss_seq_range.csv')
    annotations={(int(r['TSS'])-1,r['strand']):r for r in csv.DictReader(table.open())}
    gb=Path('data/raw/baculovirus/Baculoviridae/NC_001623.1.gb')
    record=SeqIO.read(gb,'genbank')
    cds=[]
    for index,f in enumerate(record.features):
        if f.type!='CDS':continue
        cds.append(dict(id=index,name=';'.join(f.qualifiers.get('gene',f.qualifiers.get('locus_tag',f.qualifiers.get('protein_id',['unnamed'])))),
                        product=';'.join(f.qualifiers.get('product',[])),strand='-' if f.location.strand==-1 else '+',
                        start=int(f.location.start),end=int(f.location.end),
                        parts=[(int(p.start),int(p.end)) for p in f.location.parts]))
    details=[]; overlaps=[]; summary={}
    for split in ['train','validation','test']:
        for r in csv.DictReader(Path('data/processed/clean_v2',split+'.csv').open()):
            if int(r['label'])!=1:continue
            anchor,bounds,strand,_=r['name'].split('|');start,end=map(int,bounds.split(':'))
            annotation=annotations[(int(anchor),strand)]
            own=(int(annotation['orf_st'])-1,int(annotation['orf_ed']))
            matching=[f for f in cds if f['strand']==strand and own in f['parts']]
            own_bp=intersection((start,end),own)
            hits=[]
            for f in cds:
                bp=sum(intersection((start,end),part) for part in f['parts'])
                if not bp:continue
                relation=('assigned_downstream' if own_bp else 'assigned_other_cds_part') if f in matching else 'other_same_strand' if f['strand']==strand else 'other_opposite_strand'
                hit=dict(split=split,window=r['name'],cds_name=f['name'],product=f['product'],cds_strand=f['strand'],
                         cds_start_1based=f['start']+1,cds_end_1based=f['end'],overlap_bp=bp,relationship=relation)
                overlaps.append(hit);hits.append(hit)
            expected=record.seq[start:end]
            if strand=='-':expected=expected.reverse_complement()
            assert str(expected).upper()==r['sequence'].upper()
            details.append(dict(split=split,name=r['name'],strand=strand,window_start_1based=start+1,window_end_1based=end,
                tss_1based=int(anchor)+1,assigned_orf_start_1based=own[0]+1,assigned_orf_end_1based=own[1],
                assigned_genbank_matches=len(matching),assigned_genbank_names=';'.join(f['name'] for f in matching),
                assigned_orf_overlap_bp=own_bp, assigned_other_part_overlap_bp=sum(h['overlap_bp'] for h in hits if h['relationship']=='assigned_other_cds_part'),
                upstream_of_assigned_cds=(end<=own[0] if strand=='+' else start>=own[1]),
                other_same_strand_cds=';'.join(f"{h['cds_name']}:{h['overlap_bp']}bp" for h in hits if h['relationship']=='other_same_strand'),
                other_opposite_strand_cds=';'.join(f"{h['cds_name']}:{h['overlap_bp']}bp" for h in hits if h['relationship']=='other_opposite_strand')))
        selected=[r for r in details if r['split']==split]
        summary[split]=dict(positives=len(selected),assigned_orf_overlap_windows=sum(r['assigned_orf_overlap_bp']>0 for r in selected),
            assigned_orf_overlap_bp_histogram=dict(Counter(r['assigned_orf_overlap_bp'] for r in selected)),
            assigned_other_part_overlap_windows=sum(r['assigned_other_part_overlap_bp']>0 for r in selected),
            fully_upstream_windows=sum(r['upstream_of_assigned_cds'] for r in selected),
            other_same_strand_cds_windows=sum(bool(r['other_same_strand_cds']) for r in selected),
            other_opposite_strand_cds_windows=sum(bool(r['other_opposite_strand_cds']) for r in selected),
            unmatched_assigned_orfs=[r['name'] for r in selected if r['assigned_genbank_matches']!=1])
    for name,rows in [('positive-windows.csv',details),('cds-overlaps.csv',overlaps)]:
        with (out/name).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    result=dict(coordinates='All calculations zero-based half-open; CSV coordinate fields explicitly one-based inclusive.',
                scope='All three positive splits inspected for annotation/coordinates only; no model fitting or predictions.',
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [table,gb]},splits=summary,
                own_overlap_details=[r for r in details if r['assigned_orf_overlap_bp'] or r['assigned_other_part_overlap_bp']])
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
