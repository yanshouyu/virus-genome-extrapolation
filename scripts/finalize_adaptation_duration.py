"""Finalize the predeclared study only after all scientific and replay checks pass."""
import argparse, csv, hashlib, json, shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/2026-10-24')
    p.add_argument('--output',default='documents/adaptation-2026-10-24/results')
    args=p.parse_args();root=Path(args.root);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    dev=root/'development-with-pilot';assert (dev/'COMPLETE').exists()
    r=json.loads((dev/'metrics.json').read_text());assert r['pilot_included'] and len(r['runs'])==14
    historical=json.loads(Path('documents/adaptation-2026-10-23/development-metrics.json').read_text())
    assert r['dataset_sha256']==historical['dataset_sha256']
    source_checks={};replay_checks={};max_delta=0.;audit_rows=[];audit_identity=set()
    for size,method,count in [('100m','full',6),('500m','full',6),('100m','lora',2)]:
        replay_path=root/f'fresh-replay-{size}-{method}.json';replay=json.loads(replay_path.read_text())
        runs={k:v for k,v in replay.items() if not k.startswith('_')};assert len(runs)==count
        verification=replay['_verification'];audit_identity.add((verification['audit_panel_sha256'],verification['audit_mask_sha256']))
        for path,value in verification['source_sha256'].items():assert sha(path)==value,path
        assert sha('documents/adaptation-2026-10-24/verification-amendment.md')==verification['amendment_sha256']
        original=verification['original_audit_mlm_loss']
        for name,check in runs.items():
            assert check['selected_states_exact']
            for budget,values in check['embedding_replay_max_abs'].items():
                max_delta=max(max_delta,max(values.values()))
                audit_rows.append(dict(run=name,budget=int(budget),original_audit_mlm_loss=original,selected_audit_mlm_loss=check['independent_audit_mlm_losses'][budget]))
            replay_checks[name]=check
        shutil.copy2(replay_path,out/replay_path.name)
    assert len(audit_identity)==1
    assert set(replay_checks)==set(r['runs'])
    parameters=[];all_metadata={};histories={}
    for name,meta in r['runs'].items():
        run=root/'scientific'/name
        assert (run/'COMPLETE').exists()
        for path,value in meta['source_sha256'].items():assert sha(path)==value,(name,path)
        assert sha('documents/adaptation-2026-10-24/protocol.md')==meta['protocol_sha256']
        source_checks[name]=True
        assert meta['args']['steps']==4096 and meta['input_tokens']==4096*32*257
        history=json.loads((run/'history.json').read_text());histories[name]=history
        for budget in [512,1024,2048,4096]:
            selection=meta['budgets'][str(budget)]
            eligible=[h for h in history if h['step']<=budget]
            chosen=min(eligible,key=lambda h:(h['mlm_loss'],h['step']))
            assert chosen['step']==selection['selected_step'] and abs(chosen['mlm_loss']-selection['selected_mlm_loss'])<1e-12
            assert sha(run/f'budget-{budget}'/'selected_state.pt')==selection['state_sha256']
        if meta['args']['method']=='lora':assert meta['frozen_parameters_exact']
        all_metadata[name]=meta
        parameters.append(dict(run=name,parameters=meta['parameters'],trainable_parameters=meta['trainable_parameters'],
            selected_step=meta['selected_step'],initial_mlm_loss=meta['initial_mlm_loss'],selected_mlm_loss=meta['selected_mlm_loss'],
            endpoint_mlm_loss=meta['budgets']['4096']['endpoint_mlm_loss'],training_seconds=meta['training_seconds'],
            total_seconds=meta['total_seconds'],peak_allocated_gb=meta['max_gpu_allocated_bytes']/1e9,
            peak_reserved_gb=meta['max_gpu_reserved_bytes']/1e9,plateau_by_rule=meta['plateau']['plateau_by_rule'],
            relative_last1024_improvement=meta['plateau']['best_loss_relative_improvement']))
    for condition in ['baculo','general']:
        a=r['runs'][f'100m-full-{condition}-seed42'];b=r['runs'][f'100m-lora-{condition}-seed42']
        for budget in [512,1024,2048,4096]:
            assert a['budgets'][str(budget)]['masked_tokens']==b['budgets'][str(budget)]['masked_tokens']
    for path,value in r['source_sha256'].items():assert sha(path)==value,path
    assert sha('runs/2026-10-21/frozen-readout-selection.json')==r['frozen_readout_sha256']
    readout_delta=max(m['readout_replay_max_abs'] for m in r['models'].values());assert readout_delta<1e-10
    verification=dict(completed_utc=datetime.now(timezone.utc).isoformat(),all_scientific_and_analysis_sources_match=True,
        scientific_runs=len(r['runs']),readouts=len(r['models']),all_budget_checkpoint_selections_match=True,
        dataset_sha256_matches_october23=True,fresh_process_embedding_max_abs=max_delta,
        readout_replay_max_abs=readout_delta,independent_audit_panel_and_masks_match=True,
        test_predictions_generated=False,protocol_sha256=sha('documents/adaptation-2026-10-24/protocol.md'),
        verification_amendment_sha256=sha('documents/adaptation-2026-10-24/verification-amendment.md'),
        finalizer_source_sha256=sha(__file__))
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    (out/'run-metadata.json').write_text(json.dumps(all_metadata,indent=2)+'\n')
    (out/'histories.json').write_text(json.dumps(histories,indent=2)+'\n')
    for filename,rows in [('training-summary.csv',parameters),('independent-mlm-audit.csv',audit_rows)]:
        with (out/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    shutil.copy2(dev/'metrics.json',out/'development-metrics.json')
    for path in dev.glob('*_validation.csv'):shutil.copy2(path,out/path.name)
    def summary(keys):
        return ' | '.join(f'{np.mean([r["models"][key][metric] for key in keys]):.3f}'+
            (f' +/- {np.std([r["models"][key][metric] for key in keys],ddof=1):.3f}' if len(keys)>1 else '') for metric in ['log_loss','mcc','roc_auc'])
    table=['| Model / original pretraining scope | Method | Corpus | Budget | Log loss | MCC | ROC AUC |','|---|---|---|---:|---:|---:|---:|']
    for size in ['100m','500m']:
        table.append(f'| NTv2 {size.upper()} (multispecies) | Original frozen | None | 0 | '+summary([size+'_before'])+' |')
        for budget in [512,1024,2048,4096]:
            for condition in ['general','baculo']:
                keys=[f'{size}-full-{condition}-seed{s}-budget{budget}' for s in [42,43,44]]
                table.append(f'| NTv2 {size.upper()} (multispecies) | Full | {condition} | {budget} | '+summary(keys)+' |')
    table.append('| Oriented 1-4-mer logistic | Logistic | None | 0 | '+summary(['kmer'])+' |')
    pilot=['| Model / original pretraining scope | Method | Corpus | Seed | Log loss | MCC | ROC AUC |','|---|---|---|---:|---:|---:|---:|']
    for condition in ['general','baculo']:
        for method in ['full','lora']:
            pilot.append(f'| NTv2 100M (multispecies) | {method}, 4096-step budget | {condition} | 42 | '+summary([f'100m-{method}-{condition}-seed42-budget4096'])+' |')
    effects=['| Planned endpoint comparison | Log-loss change [95% interval] | MCC change [95% interval] |','|---|---:|---:|']
    for key,c in r['comparison'].items():
        if '-budget4096-' in key or '-4096-minus-512' in key:
            vals=[]
            for metric in ['log_loss','mcc']:
                lo,hi=c[metric+'_interval'];vals.append(f'{c[metric+"_delta"]:+.4f} [{lo:+.4f}, {hi:+.4f}]')
            effects.append('| '+key+' | '+' | '.join(vals)+' |')
    plateau=sum(m['plateau']['plateau_by_rule'] for name,m in r['runs'].items() if '-full-baculo-' in name)
    text=['# October 24 adaptation-duration study and LoRA pilot','',
        'All 12 full-parameter trajectories and both single-seed LoRA corpus arms completed, with independent checkpoint replay and a separate-locus MLM audit. No downstream test predictions were generated. These remain development results on one AcMNPV genome.',
        '',f'Of the six full baculovirus runs, {plateau}/6 meet the predeclared practical loss-plateau rule (<0.5% cumulative-best loss improvement during the last 1,024 steps). This describes this schedule and held-out panel; it does not prove sufficient adaptation for all tasks. Inspect the independent MLM audit as well as the selection loss.',
        '', '## Fixed development scores','',*table,'',
        'Adapted full entries are mean +/- sample SD across seeds 42/43/44. All NT readouts use the previously frozen masked-mean pooling and C=.01, trained on 233 windows and evaluated on 77 validation windows. K-mers retain their frozen C. Threshold>.5; no retuning.',
        '', '## Planned endpoint effects','',*effects,'',
        'Negative log-loss changes and positive MCC changes favor the left condition. Intervals use 2,000 paired resamples of 64 validation groups and condition on these trained models/data. Intervals spanning zero leave the effect unresolved. Multiple duration comparisons are descriptive and unadjusted; do not choose a winning downstream budget.',
        '', '## Exploratory LoRA feasibility','',*pilot,'',
        'LoRA uses query/value rank8, alpha16, no dropout and frozen original weights including embeddings/MLM head. Its fixed lr 1e-4 differs from full lr 1e-5. One seed and no equal hyperparameter search prevent claims of superiority or noninferiority. Compare paired seed42 only and report actual compute/memory in training-summary.csv.',
        '', '## Interpretation limits','',
        'The new trajectories restart from original pinned pretrained weights with cosine decay over 4,096 steps. Budget 512 is a prefix of that schedule, not the October 23 separately decayed512-step run or an independently optimized short schedule. Selection uses only fixed viral MLM loss; the extra 256-window MLMaudit excludes both orientations of selection loci and never changes selection. These extra windows share held-out species groups and are not new biological replication.',
        '', 'Approximate MLM saturation plus improved downstream performance would support usefulness under the tested regime; it cannot establish universal necessity. Baculovirus-vs-general and comparisons with k-mers remain essential. The existing downstream test was inspected in historical project work and is not a pristine external cohort.',
        '', '## Evidence','',
        'verification.json records source/selection/replay/data identity checks. development-metrics.json contains every fixed readout and paired comparison. independent-mlm-audit.csv and fresh-replay JSON files retain audit losses, panel identities and embedding checks. run-metadata.json/histories.json retain all scientific provenance. Full state files remain in runs/2026-10-24/scientific/ on Nibi. Figures can be regenerated locally with scripts/summarize_adaptation_duration.py after collecting compact outputs.',
        '']
    (out/'README.md').write_text('\n'.join(text))
    (out/'performance-table.md').write_text('\n'.join(table)+'\n')
    (out/'lora-pilot-table.md').write_text('\n'.join(pilot)+'\n')
    (out/'COMPLETE').write_text('All 14 runs and independent checks complete; no downstream test inference.\n')
    print(json.dumps(verification,indent=2))

if __name__=='__main__':main()
