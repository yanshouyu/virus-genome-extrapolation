"""Audit the strand-aware policy across every pair of dataset splits."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from baculo_tss_probe.splitting import coordinates, cross_split_violations, read_known_tss, label_violations


def audit(directory):
    files = {name: directory / f'{name}.csv' for name in ('train', 'validation', 'test')
             if (directory / f'{name}.csv').is_file()}
    if len(files) < 2:
        raise ValueError('At least two CSV splits are required.')
    rows = {name: list(csv.DictReader(path.open())) for name, path in files.items()}
    identifiers = [row['name'] for records in rows.values() for row in records]
    result = {
        'policy': 'same-strand overlap and duplicate/group leakage prohibited; opposite-strand overlap allowed',
        'splits': {name: {'rows': len(records), 'labels': dict(Counter(r['label'] for r in records)),
                          'lengths': dict(Counter(len(r['sequence']) for r in records)),
                          'sha256': hashlib.sha256(files[name].read_bytes()).hexdigest()}
                   for name, records in rows.items()},
        'duplicate_identifiers': len(identifiers) - len(set(identifiers)),
        'violations': cross_split_violations(rows,
            Path(__file__).resolve().parents[1] / 'data/interim/tss_seq_range.csv'),
        'label_errors': label_violations([row for records in rows.values() for row in records],
            read_known_tss(Path(__file__).resolve().parents[1] / 'data/raw/baculovirus/Baculovirus_TSS.csv')),
        'allowed_opposite_strand_overlap_pairs': 0,
    }
    names = list(rows)
    for i, first in enumerate(names):
        for second in names[i+1:]:
            for a in rows[first]:
                start, end, strand = coordinates(a)
                for b in rows[second]:
                    other_start, other_end, other_strand = coordinates(b)
                    if strand != other_strand and max(start, other_start) < min(end, other_end):
                        result['allowed_opposite_strand_overlap_pairs'] += 1
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--data-dir', type=Path, default=root / 'data/processed/clean_v2')
    parser.add_argument('--allow-leakage', action='store_true', help='Report historical violations without failing.')
    args = parser.parse_args()
    result = audit(args.data_dir)
    print(json.dumps(result, indent=2))
    if not args.allow_leakage and (result['violations'] or result['duplicate_identifiers'] or result['label_errors']):
        raise SystemExit(1)
