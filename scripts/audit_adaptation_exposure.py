"""Describe training-corpus exposure without fitting or changing any model."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
result={}
for domain in ['baculo','general']:
    rows=[json.loads(line) for line in Path('runs/2026-10-21/corpus',domain+'_train.jsonl').read_text().splitlines()]
    loci={}
    for r in rows:loci.setdefault((r['accession'],r['start'],r['end']),set()).add(r['strand'])
    counts=Counter(r['strand'] for r in rows)
    gc=np.array([(r['sequence'].count('G')+r['sequence'].count('C'))/len(r['sequence']) for r in rows])
    result[domain]=dict(windows=len(rows),orientations=dict(counts),unique_genomic_windows=len(loci),
        loci_with_both_orientations=sum(len(v)==2 for v in loci.values()),
        source_accessions=len({r['accession'] for r in rows}),groups=len({r['group'] for r in rows}),
        unique_oriented_sequences=len({r['sequence'] for r in rows}),gc_mean=float(gc.mean()),gc_sd=float(gc.std(ddof=1)),
        total_window_nt=len(rows)*1536,unique_locus_nt=len(loci)*1536)
Path('documents/adaptation-2026-10-23/corpus-exposure.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
