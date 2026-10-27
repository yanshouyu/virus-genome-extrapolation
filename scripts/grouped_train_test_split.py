#!/usr/bin/env python3
"""Generate a versioned train/validation/test split of the fixed AcMNPV windows.

Same-strand overlaps and windows associated with the same ORF stay together.
Opposite-strand overlap is permitted. Existing course files are not overwritten.
Run from the repository root with PYTHONPATH=src.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from baculo_tss_probe.splitting import (
    build_groups, coordinates, cross_split_violations, split_rows, read_known_tss, label_violations,
)


def read_fasta(path):
    """Read the simple DNA FASTA emitted by the extraction scripts."""
    name, sequence = None, []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if name is not None:
                    yield name, ''.join(sequence).upper()
                name, sequence = line[1:].split()[0], []
            elif name is None:
                raise ValueError(f'Sequence before FASTA header in {path}')
            else:
                sequence.append(line)
    if name is not None:
        yield name, ''.join(sequence).upper()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pos-fasta', default='data/interim/tss_seq_range.fasta')
    parser.add_argument('--neg-fasta', default='data/interim/tss_neg_seq_range.fasta')
    parser.add_argument('--pos-clusters', default='data/interim/cluster_tss_pos_clusters.tsv')
    parser.add_argument('--neg-clusters', default='data/interim/cluster_tss_neg_clusters.tsv')
    parser.add_argument('--tss-table', default='data/interim/tss_seq_range.csv')
    parser.add_argument('--label-source', default='data/raw/baculovirus/Baculovirus_TSS.csv')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--validation-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', default='data/processed/clean_v2')
    args = parser.parse_args()
    rows = []
    for path in (args.pos_fasta, args.neg_fasta):
        for name, sequence in read_fasta(path):
            row = {'sequence': sequence, 'name': name,
                   'label': int(name.split('|')[-1])}
            coordinates(row)
            rows.append(row)
    excluded = label_violations(rows, read_known_tss(args.label_source))
    excluded_names = {row['name'] for row in excluded}
    source_count = len(rows)
    rows = [row for row in rows if row['name'] not in excluded_names]
    groups = build_groups(rows, (args.pos_clusters, args.neg_clusters), args.tss_table,
                          allowed_missing=excluded_names)
    splits = split_rows(rows, groups, args.test_size, args.validation_size, args.seed)
    violations = cross_split_violations(splits, args.tss_table)
    if violations:
        raise ValueError(f'Split failed leakage checks: {violations[:3]}')
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, records in splits.items():
        with (out / f'{split}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['sequence', 'name', 'label', 'group'],
                                    lineterminator='\n')
            writer.writeheader()
            writer.writerows(records)
        with (out / f'{split}.fasta').open('w') as handle:
            for row in records:
                handle.write(f">{row['name']}\n{row['sequence']}\n")
    inputs = (args.pos_fasta, args.neg_fasta, args.pos_clusters, args.neg_clusters,
              args.tss_table, args.label_source)
    report = {
        'dataset_version': 'clean_v2', 'genome': 'NC_001623.1', 'seed': args.seed,
        'policy': 'same-strand overlap, same associated ORF, exact/RC duplicate and known cluster groups; opposite-strand overlap allowed',
        'target_fractions': {'train': round(1 - args.test_size - args.validation_size, 6),
                             'validation': args.validation_size, 'test': args.test_size},
        'source_sha256': {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs},
        'source_rows': source_count, 'excluded_label_errors': excluded,
        'rows': len(rows), 'groups': len(set(groups)),
        'splits': {name: {'rows': len(records), 'labels': dict(Counter(r['label'] for r in records)),
                          'groups': len({r['group'] for r in records}),
                          'sha256': hashlib.sha256((out / f'{name}.csv').read_bytes()).hexdigest()}
                   for name, records in splits.items()},
        'cross_split_violations': violations,
        'limitations': 'New split of previously inspected windows; not an unseen biological cohort or proof of independence from pretraining data. No exhaustive remote homology search.',
    }
    (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
