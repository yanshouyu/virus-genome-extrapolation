"""Exercise the actual analysis entrypoint with a clearly synthetic complete panel."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np


def test_fixed_duration_readouts_and_paired_null_effects(tmp_path, monkeypatch):
    script=Path(__file__).resolve().parents[1]/'scripts/evaluate_adaptation_duration.py'
    sys.path.insert(0,str(script.parent))
    spec=importlib.util.spec_from_file_location('duration_analysis_test_target',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    Path('scripts').mkdir()
    for name in ['analyze_validation_readouts.py','evaluate_controlled_adaptation.py','run_sequence_baselines.py']:
        (Path('scripts')/name).write_bytes((script.parent/name).read_bytes())
    rng=np.random.default_rng(47); data={}
    for split,n in [('train',40),('validation',24)]:
        labels=np.arange(n)%2
        names=np.array([f'{split}-{i}' for i in range(n)])
        groups=np.array([f'{split}-group-{i//2}' for i in range(n)])
        x=rng.normal(size=(n,6));x[:,0]+=labels*.7
        data[split]=dict(names=names,groups=groups,labels=labels,cls=x,mean=x)
        folder=Path('data/processed/clean_v2');folder.mkdir(parents=True,exist_ok=True)
        import csv
        with (folder/(split+'.csv')).open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['name','group','label','sequence'])
            for i in range(n):
                sequence=''.join(rng.choice(list('ACGT'),size=60))
                writer.writerow([names[i],groups[i],labels[i],sequence])
    frozen=Path('runs/2026-10-21');frozen.mkdir(parents=True)
    (frozen/'frozen-readout-selection.json').write_text(json.dumps(dict(selection={
        '100m':dict(pooling='mean',C=.01),'500m':dict(pooling='mean',C=.01),'kmer':dict(C=.01)})))
    root=Path('fixture-scientific')
    for size in ['100m','500m']:
        for method in (['full','lora'] if size=='100m' else ['full']):
            for condition in ['baculo','general']:
                for seed in ([42,43,44] if method=='full' else [42]):
                    run=root/f'{size}-{method}-{condition}-seed{seed}';run.mkdir(parents=True)
                    (run/'COMPLETE').write_text('SYNTHETIC TEST FIXTURE')
                    meta=dict(all_source_parameters_exact=True,saved_state_exact_replay=True,frozen_parameters_exact=True,
                              input_tokens=4096*32*257,masked_tokens=123,
                              budgets={str(b):dict(masked_tokens=b) for b in [512,1024,2048,4096]})
                    (run/'metadata.json').write_text(json.dumps(meta))
                    for split in data:np.savez_compressed(run/f'before_{split}.npz',**data[split])
                    for budget in [512,1024,2048,4096]:
                        dest=run/f'budget-{budget}';dest.mkdir();(dest/'COMPLETE').write_text('SYNTHETIC TEST FIXTURE')
                        for split in data:np.savez_compressed(dest/f'after_{split}.npz',**data[split])
    monkeypatch.setattr(sys,'argv',[str(script),'--root',str(root),'--output','fixture-report','--include-pilot'])
    module.main()
    result=json.loads(Path('fixture-report/metrics.json').read_text())
    assert len(result['runs'])==14
    assert len(result['models'])==59
    for comparison in result['comparison'].values():
        assert abs(comparison['log_loss_delta'])<1e-12
        assert abs(comparison['mcc_delta'])<1e-12
    assert result['selection']['100m']['C']==.01
    assert max(m['readout_replay_max_abs'] for m in result['models'].values())<1e-12
