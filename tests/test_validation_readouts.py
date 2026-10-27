"""Exercise the validation-only workflow with no test dataset present."""
import csv
import json
import subprocess
import sys
from pathlib import Path
import numpy as np


def test_validation_workflow_without_test_file(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts/analyze_validation_readouts.py'
    root=tmp_path/'data/processed/clean_v2'; root.mkdir(parents=True)
    rng=np.random.default_rng(42)
    for split,n in [('train',60),('validation',16)]:
        labels=np.tile([0,1],n//2)
        names=np.array([f'{split}-{i}' for i in range(n)])
        groups=np.array([f'{split}-g{i//2}' for i in range(n)])
        with (root/(split+'.csv')).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['sequence','name','label','group']); writer.writeheader()
            for i in range(n):
                writer.writerow(dict(sequence=''.join(rng.choice(list('ACGT'),size=60)),
                                     name=names[i],label=labels[i],group=groups[i]))
        for family in ['ntv1','ntv2']:
            directory=tmp_path/'embeddings'/family; directory.mkdir(parents=True,exist_ok=True)
            x=rng.normal(size=(n,4)).astype(np.float32); x[:,0]+=labels
            np.savez_compressed(directory/(split+'.npz'),cls=x,mean=x+0.1,
                                names=names,groups=groups,labels=labels)
    subprocess.run([sys.executable,str(script),'--embedding-root',str(tmp_path/'embeddings'),
                    '--output',str(tmp_path/'results')],cwd=tmp_path,check=True,capture_output=True)
    assert (tmp_path/'results/COMPLETE').exists()
    result=json.loads((tmp_path/'results/metrics.json').read_text())
    for fold in result['outer_folds']:
        a={name.rsplit('-',1)[0]+'-'+str(int(name.rsplit('-',1)[1])//2) for name in fold['fit_names']}
        b={name.rsplit('-',1)[0]+'-'+str(int(name.rsplit('-',1)[1])//2) for name in fold['heldout_names']}
        assert not a & b
    assert set(result['models'])=={'kmer','ntv1_cls','ntv1_mean','ntv2_cls','ntv2_mean'}
    for name,record in result['models'].items():
        assert record['replay_max_abs']<1e-10
        with np.load(tmp_path/'results'/(name+'_outer_oof.npz')) as data:
            assert len(data['probabilities'])==60
            assert np.isfinite(data['probabilities']).all()

