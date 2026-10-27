"""Render the fixed duration panel, with an explicitly exploratory LoRA section."""
import argparse,csv,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/2026-10-24')
    p.add_argument('--development',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.root);dev=Path(args.development);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    r=json.loads((dev/'metrics.json').read_text());assert (dev/'COMPLETE').exists()
    for name in ['metrics.json','COMPLETE']:
        shutil.copy2(dev/name,out/('development-'+name if name=='metrics.json' else 'DEVELOPMENT_COMPLETE'))
    budgets=[512,1024,2048,4096];colors={'baculo':'#2563eb','general':'#d97706'}
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,size in zip(axes,['100m','500m']):
        for condition in ['baculo','general']:
            for seed in [42,43,44]:
                name=f'{size}-full-{condition}-seed{seed}';m=r['runs'][name]
                history=json.loads((root/'collected'/name/'history.json').read_text())
                ax.plot([0]+[h['step'] for h in history],[m['initial_mlm_loss']]+[h['mlm_loss'] for h in history],color=colors[condition],alpha=.65,label=condition if seed==42 else None)
        if r['pilot_included'] and size=='100m':
            for condition in ['baculo','general']:
                name=f'{size}-lora-{condition}-seed42';m=r['runs'][name]
                history=json.loads((root/'collected'/name/'history.json').read_text())
                ax.plot([0]+[h['step'] for h in history],[m['initial_mlm_loss']]+[h['mlm_loss'] for h in history],color=colors[condition],ls='--',label='LoRA '+condition+' (one seed)')
        ax.set_title('NTv2 '+size.upper()+' (multispecies)');ax.set_xlabel('Optimizer steps');ax.set_ylabel('Held-out viral MLM loss');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.savefig(out/'mlm-duration.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for i,size in enumerate(['100m','500m']):
        for ax,metric in zip(axes[i],['log_loss','mcc']):
            for condition in ['baculo','general']:
                values=np.array([[r['models'][f'{size}-full-{condition}-seed{s}-budget{b}'][metric] for s in [42,43,44]] for b in budgets])
                ax.errorbar(budgets,values.mean(1),yerr=values.std(1,ddof=1),marker='o',color=colors[condition],label=condition+' (mean +/- seed SD)',capsize=3)
                if r['pilot_included'] and size=='100m':
                    vals=[r['models'][f'{size}-lora-{condition}-seed42-budget{b}'][metric] for b in budgets]
                    ax.plot(budgets,vals,marker='x',color=colors[condition],ls='--',label='LoRA '+condition+' (one seed)')
            ax.axhline(r['models'][size+'_before'][metric],color='#64748b',ls=':',label='Unadapted NTv2 '+size.upper()+' (multispecies)')
            ax.axhline(r['models']['kmer'][metric],color='#15803d',ls='--',label='k-mer')
            ax.set_title('NTv2 '+size.upper()+' (multispecies)');ax.set_xlabel('Maximum adaptation steps');ax.set_ylabel('Validation '+('log loss (lower is better)' if metric=='log_loss' else 'MCC (higher is better)'));ax.set_xticks(budgets);ax.legend(fontsize=7);ax.grid(alpha=.15)
    fig.savefig(out/'development-duration.png',dpi=180);plt.close(fig)
    table=['| Model / original pretraining scope | Method | Corpus | Budget | Log loss | MCC | ROC AUC |','|---|---|---|---:|---:|---:|---:|']
    records=[]
    for size in ['100m','500m']:
        m=r['models'][size+'_before'];table.append(f"| NTv2 {size.upper()} (multispecies) | Frozen original | None | 0 | {m['log_loss']:.3f} | {m['mcc']:.3f} | {m['roc_auc']:.3f} |")
        for budget in budgets:
            for condition in ['general','baculo']:
                models=[r['models'][f'{size}-full-{condition}-seed{s}-budget{budget}'] for s in [42,43,44]]
                vals=['%.3f +/- %.3f'%(np.mean([m[k] for m in models]),np.std([m[k] for m in models],ddof=1)) for k in ['log_loss','mcc','roc_auc']]
                table.append(f'| NTv2 {size.upper()} (multispecies) | Full | {condition} | {budget} | '+' | '.join(vals)+' |')
    m=r['models']['kmer'];table.append(f"| Oriented 1–4-mer logistic | Logistic | None | 0 | {m['log_loss']:.3f} | {m['mcc']:.3f} | {m['roc_auc']:.3f} |")
    (out/'performance-table.md').write_text('\n'.join(table)+'\n')
    if r['pilot_included']:
        pilot=['| Model / original pretraining scope | Method | Corpus | Budget | Log loss | MCC | ROC AUC |','|---|---|---|---:|---:|---:|---:|']
        for condition in ['general','baculo']:
            for method in ['full','lora']:
                m=r['models'][f'100m-{method}-{condition}-seed42-budget4096']
                pilot.append(f"| NTv2 100M (multispecies) | {method}, seed42 | {condition} | 4096 | {m['log_loss']:.3f} | {m['mcc']:.3f} | {m['roc_auc']:.3f} |")
        (out/'lora-pilot-table.md').write_text('\n'.join(pilot)+'\n')
    for name,m in r['runs'].items():
        records.append(dict(run=name,trainable_parameters=m['trainable_parameters'],parameters=m['parameters'],selected_step=m['selected_step'],initial_mlm_loss=m['initial_mlm_loss'],selected_mlm_loss=m['selected_mlm_loss'],endpoint_mlm_loss=m['budgets']['4096']['endpoint_mlm_loss'],training_seconds=m['training_seconds'],total_seconds=m['total_seconds'],peak_allocated_gb=m['max_gpu_allocated_bytes']/1e9,peak_reserved_gb=m['max_gpu_reserved_bytes']/1e9,plateau_by_rule=m['plateau']['plateau_by_rule'],relative_last1024_improvement=m['plateau']['best_loss_relative_improvement']))
    with (out/'training-summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    shutil.copy2(dev/'metrics.json',out/'development-metrics.json')
    for path in dev.glob('*_validation.csv'):shutil.copy2(path,out/path.name)
    print(json.dumps(dict(primary_runs=sum('-full-' in n for n in r['runs']),pilot_included=r['pilot_included'],plateau_runs=[row['run'] for row in records if row['plateau_by_rule']],comparison=r['comparison']),indent=2))

if __name__=='__main__': main()
