"""Freeze pooling/C using only unadapted training embeddings before outcome analysis."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from evaluate_controlled_adaptation import screen_pooling,select_c
from run_sequence_baselines import features

out=Path('runs/2026-10-21/frozen-readout-selection.json')
if out.exists():raise FileExistsError(out)
rows=list(csv.DictReader(Path('data/processed/clean_v2/train.csv').open()))
y=np.array([int(r['label']) for r in rows]);groups=np.array([r['group'] for r in rows])
selection={};hashes={}
for size in ['100m','500m']:
    path=Path('runs/2026-10-21/scientific')/(size+'-baculo-seed42')/'before_train.npz'
    with np.load(path) as a:
        assert np.array_equal(a['names'],[r['name'] for r in rows]);assert np.array_equal(a['labels'],y)
        x={p:a[p].astype(np.float64) for p in ['cls','mean']}
    pooling,screen=screen_pooling(x,y,groups);c,losses=select_c(x[pooling],y,groups)
    selection[size]=dict(pooling=pooling,C=c,C_inner_losses=losses,unadapted_nested_pooling_screen=screen)
    hashes[size]={p:hashlib.sha256(x[p].tobytes()).hexdigest() for p in x}
x=features([r['sequence'] for r in rows]);c,losses=select_c(x,y,groups);selection['kmer']=dict(C=c,C_inner_losses=losses)
out.write_text(json.dumps(dict(selection=selection,unadapted_training_embedding_sha256=hashes,
    train_csv_sha256=hashlib.sha256(Path('data/processed/clean_v2/train.csv').read_bytes()).hexdigest()),indent=2)+'\n')
print(json.dumps({size:{k:v for k,v in value.items() if k in ['pooling','C']} for size,value in selection.items()},indent=2))
