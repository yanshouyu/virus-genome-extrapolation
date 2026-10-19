"""Summarize a fixed baseline experiment; never select models using test metrics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from run_sequence_baselines import metrics


def mcc(y, p):
    predicted = p > .5
    tp = np.sum((y == 1) & predicted)
    tn = np.sum((y == 0) & ~predicted)
    fp = np.sum((y == 0) & predicted)
    fn = np.sum((y == 1) & ~predicted)
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** .5
    return float((tp * tn - fp * fn) / denominator) if denominator else 0.


def plot_results(out, summaries, uncertainty, histories, baseline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(ntv1='#2864a5', ntv2='#c56a25', kmer_logistic='#31815b', prevalence='#777777')
    fig, (curves, scores) = plt.subplots(1, 2, figsize=(12, 5.3), layout='constrained')
    for name, history in histories.items():
        model, seed = name.split('-seed')
        evaluations = [h for h in history if 'eval_loss' in h]
        curves.plot([h['epoch'] for h in evaluations], [h['eval_loss'] for h in evaluations],
                    color=colors[model], alpha=.6, linewidth=1.3,
                    label=model.replace('ntv', 'NTv')+' (3 seeds)' if seed == '42' else None)
        selected = next(row for row in summaries if row['model'] == name)
        curves.scatter(selected['selected_epoch'], selected['validation_loss'],
                       color=colors[model], s=23, zorder=3)
    reference = json.loads((baseline/'metrics.json').read_text())
    curves.axhline(reference['models']['kmer_logistic']['validation']['log_loss'],
                   color=colors['kmer_logistic'], linestyle='--', label='k-mer logistic')
    curves.set(xlabel='Epoch', ylabel='Validation log loss', title='Validation during head training')
    curves.legend(frameon=False, fontsize=9)
    labels = []
    for position, row in enumerate(summaries):
        name = row['model']
        family = name.split('-seed')[0]
        low, high = uncertainty['mcc_ci95'][name]
        scores.plot([low, high], [position, position], color=colors[family], linewidth=2)
        scores.scatter(row['mcc'], position, color=colors[family], s=34, zorder=3)
        labels.append(name.replace('ntv', 'NTv').replace('-seed', ' / seed ')
                      .replace('kmer_logistic', 'k-mer logistic'))
    lower = max(-1., min(0., min(v[0] for v in uncertainty['mcc_ci95'].values()))-.05)
    upper = min(1., max(v[1] for v in uncertainty['mcc_ci95'].values())+.05)
    scores.set(yticks=range(len(labels)), yticklabels=labels, xlabel='MCC',
               title='Test MCC: 95% group bootstrap intervals', xlim=(lower, upper))
    scores.invert_yaxis()
    scores.axvline(0, color='#bbbbbb', linewidth=.8)
    for axis in (curves, scores):
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='x', alpha=.15)
    fig.suptitle('Clean AcMNPV baseline: 233 train / 77 validation / 78 test', fontsize=13)
    fig.savefig(out/'baseline-comparison.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--baseline-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--plot', action='store_true', help='Render the optional matplotlib figure.')
    args = parser.parse_args()
    run = Path(args.run_dir)
    baseline = Path(args.baseline_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    test_rows = list(csv.DictReader(open('data/processed/clean_v2/test.csv')))
    manifest = json.loads(Path('data/processed/clean_v2/manifest.json').read_text())
    reference = json.loads((baseline/'metrics.json').read_text())
    for split in ('train', 'validation', 'test'):
        digest = hashlib.sha256(Path(f'data/processed/clean_v2/{split}.csv').read_bytes()).hexdigest()
        assert digest == manifest['splits'][split]['sha256'] == reference['dataset_sha256'][split]
    labels = np.array([int(r['label']) for r in test_rows])
    probabilities = {}
    summaries = []
    histories = {}
    for name in ('prevalence', 'kmer_logistic'):
        data = np.load(baseline/f'{name}_test_predictions.npz')
        np.testing.assert_array_equal(data['labels'], labels)
        np.testing.assert_array_equal(data['names'], [r['name'] for r in test_rows])
        probabilities[name] = data['probabilities']
        summaries.append(dict(model=name, **metrics(labels, probabilities[name])))
    for model in ('ntv1', 'ntv2'):
        for seed in (42, 43, 44):
            name = f'{model}-seed{seed}'
            directory = run/name
            predictions = np.load(directory/'test_predictions.npz')
            np.testing.assert_array_equal(predictions['labels'], labels)
            logits = predictions['logits'].astype(np.float64)
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probabilities[name] = exp[:, 1] / exp.sum(axis=1)
            history = json.loads((directory/'trainer_history.json').read_text())
            evals = [h for h in history if 'eval_loss' in h]
            state = json.loads((directory/'trainer_state.json').read_text())
            # Trainer also saves a final checkpoint at step 250 in this run.
            eligible = [h for h in evals if h['step'] % 30 == 0 or h['step'] == state['global_step']]
            best = min(eligible, key=lambda h: (h['eval_loss'], h['step']))
            assert state['global_step'] == max(h['step'] for h in history)
            assert state['best_global_step'] == best['step']
            assert state['best_model_checkpoint'].endswith(f"checkpoint-{best['step']}")
            assert abs(state['best_metric'] - best['eval_loss']) < 1e-7
            metadata = json.loads((directory/'run_metadata.json').read_text())
            config = json.loads((directory/'config.json').read_text())
            assert config['eval_steps'] == config['save_steps'] == 30
            assert config['seed'] == seed and config['num_train_epochs'] == 50
            assert metadata['dataset_manifest'] == manifest and metadata['pooling'] == 'cls'
            calculated = metrics(labels, probabilities[name])
            recorded = json.loads((directory/'test_metrics.json').read_text())
            assert abs(calculated['accuracy'] - recorded['test_accuracy']) < 1e-12
            assert abs(calculated['mcc'] - recorded['test_matthews_correlation']) < 1e-12
            assert abs(calculated['log_loss'] - recorded['test_loss']) < 1e-5
            summaries.append(dict(model=name, **calculated,
                                  validation_loss=best['eval_loss'], selected_step=best['step'],
                                  selected_epoch=best['epoch'], git_commit=metadata['git_commit'],
                                  pooling=metadata['pooling'], revision=metadata['model_revision']))
            histories[name] = history
    with (out/'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in summaries for k in r)))
        writer.writeheader()
        writer.writerows(summaries)
    with (out/'test_predictions.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['name', 'group', 'label', *probabilities])
        writer.writeheader()
        writer.writerows(dict(name=r['name'], group=r['group'], label=int(r['label']),
                              **{key: float(value[i]) for key, value in probabilities.items()})
                         for i, r in enumerate(test_rows))
    groups = sorted(set(r['group'] for r in test_rows))
    members = {g: np.array([i for i, r in enumerate(test_rows) if r['group'] == g]) for g in groups}
    rng = np.random.default_rng(20261019)
    draws = {key: [] for key in probabilities}
    differences = {model: [] for model in ('ntv1', 'ntv2')}
    for _ in range(2000):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([members[g] for g in sampled])
        scores = {key: mcc(labels[indices], p[indices]) for key, p in probabilities.items()}
        for key, score in scores.items():
            draws[key].append(score)
        for model in differences:
            differences[model].append(np.mean([scores[f'{model}-seed{s}'] for s in (42, 43, 44)])
                                      - scores['kmer_logistic'])
    uncertainty = dict(method='2000 paired bootstrap draws of test groups; fixed trained models',
                       test_groups=len(groups),
                       mcc_ci95={k: np.quantile(v, [.025, .975]).tolist() for k, v in draws.items()},
                       mean_seed_mcc_minus_kmer_ci95={k: np.quantile(v, [.025, .975]).tolist()
                                                    for k, v in differences.items()})
    (out/'uncertainty.json').write_text(json.dumps(uncertainty, indent=2)+'\n')
    (out/'metrics.json').write_text(json.dumps(summaries, indent=2)+'\n')
    (out/'training_histories.json').write_text(json.dumps(histories, indent=2)+'\n')
    subgroups = {}
    for strand in ('+', '-'):
        indices = np.array([i for i, row in enumerate(test_rows) if row['name'].split('|')[2] == strand])
        subgroups[strand] = dict(samples=len(indices), positives=int(labels[indices].sum()),
                                 models={name: metrics(labels[indices], p[indices])
                                         for name, p in probabilities.items()})
    (out/'exploratory_strand_metrics.json').write_text(json.dumps(subgroups, indent=2)+'\n')
    errors = []
    for i, row in enumerate(test_rows):
        wrong = [name for name, p in probabilities.items() if (p[i] > .5) != bool(labels[i])]
        errors.append(dict(name=row['name'], group=row['group'], label=int(labels[i]),
                           strand=row['name'].split('|')[2],
                           gc_fraction=(row['sequence'].count('G') + row['sequence'].count('C'))/300,
                           wrong_models=wrong))
    (out/'errors.json').write_text(json.dumps(errors, indent=2)+'\n')
    if args.plot:
        plot_results(out, summaries, uncertainty, histories, baseline)
    print(json.dumps(dict(metrics=summaries, uncertainty=uncertainty), indent=2))


if __name__ == '__main__':
    main()
