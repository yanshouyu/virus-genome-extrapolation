"""Read-only audit of the existing single-genome, non-wrapping CSV splits.

Run from the repository root: python scripts/audit_dataset.py
Coordinates in sequence names are zero-based, half-open. Opposite-strand
intersection means shared genomic territory, not necessarily identical input.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def interval(row):
    fields = row['name'].split('|')
    start, end = map(int, fields[1].split(':'))
    if start < 0 or end <= start:
        raise ValueError('This audit requires non-wrapping, nonnegative intervals.')
    if end - start != len(row['sequence']):
        raise ValueError('Coordinate length differs from sequence length.')
    return start, end, fields[2]


def audit(root):
    rows = {}
    result = {'scope': 'NC_001623.1; linear intervals; genomic overlap is not a homology search', 'splits': {}}
    for split in ('train', 'test'):
        path = root / 'data' / 'processed' / f'{split}.csv'
        rows[split] = list(csv.DictReader(path.open()))
        for row in rows[split]:
            interval(row)
        result['splits'][split] = {
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'rows': len(rows[split]),
            'labels': dict(Counter(row['label'] for row in rows[split])),
            'lengths': dict(Counter(len(row['sequence']) for row in rows[split])),
        }
    pairs = []
    for train in rows['train']:
        start, end, strand = interval(train)
        for test in rows['test']:
            other_start, other_end, other_strand = interval(test)
            overlap = max(0, min(end, other_end) - max(start, other_start))
            if overlap:
                pairs.append({'train': train['name'], 'test': test['name'],
                              'overlap_bp': overlap, 'same_strand': strand == other_strand})
    result['overlap_pairs'] = len(pairs)
    result['same_strand_pairs'] = sum(p['same_strand'] for p in pairs)
    result['test_rows_overlapping_train'] = len({p['test'] for p in pairs})
    result['test_rows_overlapping_train_same_strand'] = len({p['test'] for p in pairs if p['same_strand']})
    result['max_overlap_bp'] = max((p['overlap_bp'] for p in pairs), default=0)
    result['largest_overlaps'] = sorted(pairs, key=lambda p: -p['overlap_bp'])[:5]
    train_sequences = {row['sequence'].upper() for row in rows['train']}
    test_sequences = {row['sequence'].upper() for row in rows['test']}
    complement = str.maketrans('ACGTN', 'TGCAN')
    result['shared_exact_sequences'] = len(train_sequences & test_sequences)
    result['test_sequences_matching_train_reverse_complement'] = len(
        {s for s in test_sequences if s.translate(complement)[::-1] in train_sequences}
    )
    return result


if __name__ == '__main__':
    print(json.dumps(audit(Path(__file__).resolve().parents[1]), indent=2))
