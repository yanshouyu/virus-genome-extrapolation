"""Repair the general control to sample loci then include both orientations."""
import gzip,json,random
from pathlib import Path
from Bio import SeqIO
from prepare_adaptation_corpus import tiles,paired_sample,sha
root=Path('runs/2026-10-21/corpus')
report=json.loads((root/'corpus-report.json').read_text());target=report['counts']['train'];general=[]
assert target%2==0
for i,source in enumerate(report['general_sources']):
    organism=source['organism'];archive=root/(organism+'.fna.gz')
    assert sha(archive)==source['archive_sha256']
    candidates=[]
    with gzip.open(archive,'rt') as f:
        for record in SeqIO.parse(f,'fasta'):candidates.extend(w for w in tiles(str(record.seq),record.id,organism) if w['strand']=='+')
    count=(target//2)//3+(i<(target//2)%3)
    general.extend(paired_sample(candidates,count,42+i))
    source['selected_windows']=2*count
assert len(general)==target
old=root/'general_train.jsonl';backup=root/'general_train-unpaired-v1.jsonl'
assert not backup.exists();old.rename(backup)
with old.open('w') as f:
    for row in general:f.write(json.dumps(row)+'\n')
report['general_strand_pairing']='Every selected locus included in both orientations; equal unique-locus count to baculovirus.'
report['excluded_general_v1_sha256']=sha(backup)
report['files']['general_train.jsonl']=sha(old)
(root/'corpus-report-v1.json').write_text((root/'corpus-report.json').read_text())
(root/'corpus-report.json').write_text(json.dumps(report,indent=2)+'\n')
print('Repaired general windows',len(general),'sha256',sha(old))
