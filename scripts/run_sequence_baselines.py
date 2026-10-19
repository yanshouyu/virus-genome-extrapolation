"""Simple sequence baselines; select on validation, then evaluate held-out test."""
import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, log_loss,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def features(sequences):
    words = [''.join(word) for k in range(1, 5) for word in itertools.product('ACGT', repeat=k)]
    index = {word: i for i, word in enumerate(words)}
    result = np.zeros((len(sequences), len(words)), dtype=np.float64)
    for row, sequence in enumerate(sequences):
        sequence = sequence.upper()
        for k in range(1, 5):
            for start in range(len(sequence) - k + 1):
                word = sequence[start:start + k]
                if word in index:
                    result[row, index[word]] += 1 / (len(sequence) - k + 1)
    return result


def metrics(labels, probabilities):
    predicted = (probabilities > 0.5).astype(int)
    return dict(log_loss=float(log_loss(labels, probabilities, labels=[0, 1])),
                accuracy=float(accuracy_score(labels, predicted)),
                precision=float(precision_score(labels, predicted, zero_division=0)),
                recall=float(recall_score(labels, predicted, zero_division=0)),
                mcc=float(matthews_corrcoef(labels, predicted)),
                roc_auc=float(roc_auc_score(labels, probabilities)),
                average_precision=float(average_precision_score(labels, probabilities)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/processed/clean_v2')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    directory = Path(args.data_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    rows = {s: list(csv.DictReader((directory / f'{s}.csv').open()))
            for s in ('train', 'validation', 'test')}
    x = {s: features([r['sequence'] for r in records]) for s, records in rows.items()}
    y = {s: np.array([int(r['label']) for r in records]) for s, records in rows.items()}
    candidates = []
    models = []
    for c in (0.01, 0.1, 1., 10., 100.):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000,
                                                                   solver='lbfgs', random_state=42))
        model.fit(x['train'], y['train'])
        score = metrics(y['validation'], model.predict_proba(x['validation'])[:, 1])
        candidates.append(dict(C=c, validation=score))
        models.append(model)
    selected = min(range(len(models)), key=lambda i: (candidates[i]['validation']['log_loss'],
                                                       candidates[i]['C']))
    selected_model = models[selected]
    np.savez_compressed(out/'kmer_model.npz',
                        mean=selected_model[0].mean_, scale=selected_model[0].scale_,
                        coefficients=selected_model[1].coef_, intercept=selected_model[1].intercept_)
    prevalence = float(y['train'].mean())
    results = dict(candidates=candidates, selected_C=candidates[selected]['C'],
                   train_prevalence=prevalence, packages={p: importlib.metadata.version(p)
                   for p in ('numpy', 'scikit-learn')},
                   dataset_sha256={s: hashlib.sha256((directory/f'{s}.csv').read_bytes()).hexdigest()
                                   for s in rows}, models={})
    for name, model in [('prevalence', None), ('kmer_logistic', models[selected])]:
        probabilities = {s: np.full(len(y[s]), prevalence) if model is None else
                         model.predict_proba(x[s])[:, 1] for s in ('validation', 'test')}
        results['models'][name] = {s: metrics(y[s], probabilities[s]) for s in probabilities}
        for s in probabilities:
            np.savez_compressed(out/f'{name}_{s}_predictions.npz', labels=y[s],
                                probabilities=probabilities[s],
                                names=np.array([r['name'] for r in rows[s]]))
    (out/'metrics.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
