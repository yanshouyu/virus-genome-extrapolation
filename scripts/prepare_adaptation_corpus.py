"""Prepare accession/species-partitioned corpus and matched general control."""
import argparse,csv,gzip,hashlib,json,random,urllib.request
from pathlib import Path
from Bio import SeqIO


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rc(s):return s.translate(str.maketrans('ACGT','TGCA'))[::-1]
def digest(s):return hashlib.sha256(min(s,rc(s)).encode()).hexdigest()

def tiles(sequence,accession,group):
    for start in range(0,len(sequence)-1536+1,1536):
        seq=sequence[start:start+1536].upper()
        if set(seq)-set('ACGT'):continue
        for strand,text in [('+',seq),('-',rc(seq))]:
            yield dict(sequence=text,accession=accession,group=group,start=start,end=start+1536,strand=strand)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--general-manifest',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    records=[];manifest=[]
    target=SeqIO.read('data/raw/baculovirus/Baculoviridae/NC_001623.1.gb','genbank');target_hash=digest(str(target.seq).upper())
    for path in sorted(Path('data/raw/baculovirus/Baculoviridae').glob('*.gb')):
        r=SeqIO.read(path,'genbank');sequence=str(r.seq).upper();organism=r.annotations.get('organism',r.description)
        taxonomy=r.annotations.get('taxonomy',[]);group=taxonomy[-1] if taxonomy and taxonomy[-1] not in ['Alphabaculovirus','Betabaculovirus','Gammabaculovirus','Deltabaculovirus'] else organism
        excluded=r.id.split('.')[0]=='NC_001623' or 'autographa californica' in (organism+' '+r.description).lower() or 'aucalifornicae' in ' '.join(taxonomy).lower() or digest(sequence)==target_hash
        manifest.append(dict(accession=r.id,organism=organism,taxonomy=taxonomy,group=group,length=len(sequence),sha256=sha(path),excluded=excluded))
        if not excluded:records.append((r.id,sequence,group))
    # Union equal-sequence genomes before split, in addition to same-species grouping.
    parents={g:g for _,_,g in records}
    def find(g):
        while parents[g]!=g:g=parents[g]
        return g
    seen={}
    for accession,sequence,group in records:
        key=digest(sequence)
        if key in seen:parents[find(group)]=find(seen[key])
        else:seen[key]=group
    groups=sorted({find(g) for _,_,g in records});random.Random(42).shuffle(groups)
    val_groups=set(groups[:max(1,round(.1*len(groups)))])
    split={'train':[],'validation':[]};dedup=set()
    for accession,sequence,group in records:
        key=digest(sequence)
        if key in dedup:continue
        dedup.add(key);group=find(group)
        split['validation' if group in val_groups else 'train'].extend(tiles(sequence,accession,group))
    assert not {r['group'] for r in split['train']} & {r['group'] for r in split['validation']}
    assert not {r['sequence'] for r in split['train']} & {r['sequence'] for r in split['validation']}
    def write(name,rows):
        with (out/(name+'.jsonl')).open('w') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
    for name,rows in split.items():write('baculo_'+name,rows)
    eval_rows=random.Random(1729).sample(split['validation'],min(128,len(split['validation'])))
    write('mlm_validation',eval_rows)
    urls=Path(args.general_manifest).read_text().splitlines();sources=[];general=[]
    organisms=['Saccharomyces_cerevisiae','Caenorhabditis_elegans','Drosophila_melanogaster']
    for i,organism in enumerate(organisms):
        url=next(u for u in urls if '/'+organism+'/' in u)
        archive=out/(organism+'.fna.gz')
        urllib.request.urlretrieve(url,archive)
        candidates=[]
        with gzip.open(archive,'rt') as f:
            for record in SeqIO.parse(f,'fasta'):candidates.extend(tiles(str(record.seq),record.id,organism))
        n=len(split['train'])//3+(i<len(split['train'])%3)
        assert len(candidates)>=n,(organism,len(candidates),n)
        general.extend(random.Random(42+i).sample(candidates,n))
        sources.append(dict(organism=organism,url=url,archive_sha256=sha(archive),candidate_windows=len(candidates),selected_windows=n))
    assert len(general)==len(split['train'])
    write('general_train',general)
    report=dict(window_nt=1536,stride_nt=1536,manifest=manifest,general_sources=sources,
        groups=groups,validation_groups=sorted(val_groups),counts={k:len(v) for k,v in split.items()},general_windows=len(general),
        files={p.name:sha(p) for p in out.glob('*.jsonl')},manifest_source_sha256=sha(args.general_manifest))
    (out/'corpus-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['manifest','groups']},indent=2))
if __name__=='__main__':main()
