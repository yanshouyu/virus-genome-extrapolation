"""Verify corpus coordinates, orientations, grouping, and transfer-ready hashes."""
import gzip,hashlib,json
from pathlib import Path
from Bio import SeqIO
from prepare_adaptation_corpus import rc
root=Path('runs/2026-10-21/corpus')
report=json.loads((root/'corpus-report.json').read_text())
sequences={r.id:str(r.seq).upper() for p in Path('data/raw/baculovirus/Baculoviridae').glob('*.gb') for r in SeqIO.parse(p,'genbank')}
for p in root.glob('*.fna.gz'):
    with gzip.open(p,'rt') as f:sequences.update({r.id:str(r.seq).upper() for r in SeqIO.parse(f,'fasta')})
rows={name:[json.loads(line) for line in (root/name).read_text().splitlines()] for name in report['files']}
for name,records in rows.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==report['files'][name]
    for r in records:
        expected=sequences[r['accession']][r['start']:r['end']]
        if r['strand']=='-':expected=rc(expected)
        assert expected==r['sequence']
        assert len(expected)==1536 and not set(expected)-set('ACGT')
        assert r['accession'].split('.')[0]!='NC_001623'
train=rows['baculo_train.jsonl'];val=rows['baculo_validation.jsonl']
assert not {r['group'] for r in train}&{r['group'] for r in val}
assert not {r['accession'] for r in train}&{r['accession'] for r in val}
assert not {r['sequence'] for r in train}&{r['sequence'] for r in val}
assert len(rows['general_train.jsonl'])==len(train)
valkeys={(r['accession'],r['start'],r['strand']) for r in val}
assert all((r['accession'],r['start'],r['strand']) in valkeys for r in rows['mlm_validation.jsonl'])
locus_counts={}
for domain in ['baculo','general']:
    pairing={}
    for row in rows[domain+'_train.jsonl']:
        pairing.setdefault((row['accession'],row['start'],row['end']),set()).add(row['strand'])
    assert all(strands=={'+','-'} for strands in pairing.values())
    locus_counts[domain]=len(pairing)
assert locus_counts['baculo']==locus_counts['general']
result=dict(exact_strand_pairing_and_unique_locus_counts_match=True,verified_windows=sum(map(len,rows.values())),coordinates_and_orientation_match=True,
            accession_group_and_exact_sequence_separation=True,target_accession_absent=True,
            equal_training_window_counts=True,heldout_mlm_subset_verified=True,all_file_hashes_match=True)
Path('documents/adaptation-2026-10-23/corpus-verification-v2.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
