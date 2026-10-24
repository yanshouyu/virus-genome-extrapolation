"""Summarize completed adaptation development results and execution provenance."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--collected',required=True);p.add_argument('--development',required=True);p.add_argument('--output',required=True);p.add_argument('--capacity-root')
    args=p.parse_args();root=Path(args.collected);dev=Path(args.development);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    r=json.loads((dev/'metrics.json').read_text());sizes=r['sizes']
    capacity=json.loads((Path(args.capacity_root)/'readout/metrics.json').read_text()) if args.capacity_root else None
    shutil.copy2(dev/'metrics.json',out/'development-metrics.json');shutil.copy2(dev/'selection.json',out/'readout-selection.json')
    run_metadata={};rows=[]
    corpus_report=json.loads((out/'corpus-report-v2.json').read_text())
    for size in sizes:
        for condition in ['baculo','general']:
            for seed in [42,43,44]:
                name=f'{size}-{condition}-seed{seed}'
                meta=json.loads((root/name/'metadata.json').read_text());history=json.loads((root/name/'history.json').read_text())
                assert meta['corpus_sha256']==corpus_report['files'][condition+'_train.jsonl']
                assert meta['args']['seed']==seed and meta['args']['condition']==condition
                assert len(history)==4 and [h['step'] for h in history]==[128,256,384,512]
                assert meta['saved_state_exact_replay'] and meta['embedding_parameter_max_change']>0
                expected=min(history,key=lambda h:(h['mlm_loss'],h['step']))
                assert expected['step']==meta['selected_step']
                assert abs(expected['mlm_loss']-meta['selected_mlm_loss'])<1e-12
                assert meta['input_tokens']==512*32*257
                run_metadata[name]=meta
                rows.append(dict(model=size,condition=condition,seed=seed,selected_step=meta['selected_step'],
                    initial_mlm_loss=meta['initial_mlm_loss'],selected_mlm_loss=meta['selected_mlm_loss'],
                    training_seconds=meta['training_seconds'],total_seconds=meta['total_seconds'],
                    peak_allocated_gb=meta['max_gpu_allocated_bytes']/1e9,gpu=meta['gpu'],
                    input_tokens=meta['input_tokens'],masked_tokens=meta['masked_tokens']))
    for size in sizes:
        for seed in [42,43,44]:
            a=run_metadata[f'{size}-baculo-seed{seed}'];b=run_metadata[f'{size}-general-seed{seed}']
            assert a['input_tokens']==b['input_tokens'] and a['masked_tokens']==b['masked_tokens']
            assert a['source']==b['source']
    with (out/'training-summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    colors={'before':'#64748b','general':'#d97706','baculo':'#2563eb'}
    fig,axes=plt.subplots(len(sizes),2,figsize=(10,3.6*len(sizes)),squeeze=False,constrained_layout=True)
    for i,size in enumerate(sizes):
        for j,(metric,label) in enumerate([('log_loss','Validation log loss (lower is better)'),('mcc','Validation MCC (higher is better)')]):
            ax=axes[i,j]
            for k,condition in enumerate(['before','general','baculo']):
                vals=[r['models'][size+'_before'][metric]] if condition=='before' else [r['models'][f'{size}_{condition}_seed{seed}'][metric] for seed in [42,43,44]]
                ax.bar(k,np.mean(vals),color=colors[condition],alpha=.8)
                ax.scatter(np.full(len(vals),k),vals,c='black',s=20,zorder=3)
                ax.text(k,max(vals)+.012,f'{np.mean(vals):.3f}',ha='center',va='bottom',fontsize=10,bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=1))
            ax.axhline(r['models']['kmer'][metric],color='#15803d',linestyle='--',label='k-mer')
            if capacity is not None:ax.axhline(capacity['validation'][metric],color='#7c3aed',linestyle=':',label='NTv1 2.5B reference')
            ax.set_xticks(range(3),['Unadapted','General control','Baculovirus'])
            ax.set_ylim(ax.get_ylim()[0],ax.get_ylim()[1]*1.10)
            ax.set_ylabel(label);ax.set_title('NTv2 '+size.upper());ax.legend(loc='lower left',frameon=True,framealpha=.9,facecolor='white',edgecolor='none',fontsize=9)
    fig.suptitle('Paired multispecies adaptation: fixed development cohort',fontsize=14)
    fig.savefig(out/'adaptation-development.png',dpi=180);plt.close(fig)
    effect_rows=[(size,other) for size in sizes for other in ['before','general']]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    labels=[size.upper()+' baculovirus minus '+('unadapted' if other=='before' else 'general') for size,other in effect_rows]
    for ax,metric,title in zip(axes,['mcc','log_loss'],['MCC change (positive is better)','Log-loss change (negative is better)']):
        for i,(size,other) in enumerate(effect_rows):
            comparison=r['comparison'][size+'_baculo_minus_'+other]
            point=comparison[metric+'_delta'];lo,hi=comparison[metric+'_interval']
            color='#2563eb' if size=='100m' else '#c2410c'
            ax.plot([lo,hi],[i,i],color=color,lw=2)
            ax.scatter([point],[i],color=color,s=35,zorder=3)
            seed_deltas=[r['models'][f'{size}_baculo_seed{seed}'][metric]-(r['models'][size+'_before'][metric] if other=='before' else r['models'][f'{size}_general_seed{seed}'][metric]) for seed in [42,43,44]]
            ax.scatter(seed_deltas,[i]*3,facecolors='none',edgecolors='black',s=28,zorder=4)
        ax.axvline(0,color='#64748b',ls='--',lw=1);ax.set_yticks(range(len(labels)),labels if metric=='mcc' else [])
        ax.invert_yaxis();ax.set_title(title);ax.grid(axis='x',alpha=.15)
    fig.suptitle('Adaptation effects: paired group bootstrap 95% intervals',fontsize=14)
    fig.savefig(out/'paired-adaptation-effects.png',dpi=180);plt.close(fig)
    table=['| Model | Additional pretraining | Log loss | MCC | ROC AUC |','|---|---|---:|---:|---:|']
    for size in sizes:
        for condition in ['before','general','baculo']:
            records=[r['models'][size+'_before']] if condition=='before' else [r['models'][f'{size}_{condition}_seed{seed}'] for seed in [42,43,44]]
            text=[]
            for metric in ['log_loss','mcc','roc_auc']:
                values=[record[metric] for record in records]
                text.append(f'{np.mean(values):.3f}'+(f' +/- {np.std(values,ddof=1):.3f}' if len(values)>1 else ''))
            table.append('| '+size.upper()+' | '+{'before':'None','general':'General control','baculo':'Baculovirus'}[condition]+' | '+' | '.join(text)+' |')
    table.append('| k-mer | None | '+' | '.join(f"{r['models']['kmer'][metric]:.3f}" for metric in ['log_loss','mcc','roc_auc'])+' |')
    if capacity is not None:
        table.append('| NTv1 2.5B* | None | '+' | '.join(f"{capacity['validation'][metric]:.3f}" for metric in ['log_loss','mcc','roc_auc'])+' |')
        metadata=json.loads((Path(args.capacity_root)/'metadata.json').read_text())
        assert metadata['all_source_parameters_exact'] and metadata['embedding_replay_max_abs']==0
        (out/'capacity-reference.json').write_text(json.dumps(dict(extraction=metadata,readout=capacity),indent=2)+'\n')
        with np.load(Path(args.capacity_root)/'readout/validation.npz') as a:
            with (out/'capacity-2.5b_validation.csv').open('w',newline='') as f:
                writer=csv.writer(f,lineterminator='\n');writer.writerow(['name','label','probability']);writer.writerows(zip(a['names'],a['labels'],a['probabilities']))
    (out/'performance-table.md').write_text('\n'.join(table)+'\n')
    # Export compact predictions; source metadata retains hashes of full selected states on Nibi.
    for path in dev.glob('*_validation.npz'):
        with np.load(path) as a:
            with (out/(path.stem+'.csv')).open('w',newline='') as f:
                writer=csv.writer(f,lineterminator='\n');writer.writerow(['name','label','probability']);writer.writerows(zip(a['names'],a['labels'],a['probabilities']))
    provenance=dict(run_metadata=run_metadata,collected_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()},
                    development_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dev.iterdir() if p.is_file()})
    (out/'run-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(r['comparison'],indent=2))
if __name__=='__main__':main()
