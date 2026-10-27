"""Group the fixed AcMNPV windows before splitting; coordinates are half-open."""
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def coordinates(row):
    fields = row['name'].split('|')
    if len(fields) != 4 or fields[2] not in ('+', '-') or fields[3] not in ('0', '1'):
        raise ValueError(f"Invalid sequence identifier: {row['name']}")
    start, end = map(int, fields[1].split(':'))
    if start < 0 or end <= start or end - start != len(row['sequence']):
        raise ValueError(f"Invalid non-wrapping coordinates: {row['name']}")
    if str(row['label']) != fields[3]:
        raise ValueError(f"Label disagrees with identifier: {row['name']}")
    return start, end, fields[2]


def read_known_tss(path):
    """Read one-based starts from the curated source, forward-filling strand."""
    sites, strand = set(), None
    with Path(path).open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            strand = row['Direction'].strip() or strand
            if row['TSS'].strip():
                if strand not in ('+', '-'):
                    raise ValueError('Invalid TSS strand in source table.')
                sites.add((int(row['TSS']) - 1, strand))
    return sites


def label_violations(rows, sites):
    """Find positives missing their anchor and negatives containing recorded TSS."""
    violations = []
    for row in rows:
        start, end, strand = coordinates(row)
        contained = sorted(tss for tss, st in sites if st == strand and start <= tss < end)
        reasons = []
        if int(row['label']) == 1:
            anchor = int(row['name'].split('|')[0])
            if not start <= anchor < end:
                reasons.append('positive_anchor_outside_window')
            if (anchor, strand) not in sites:
                reasons.append('positive_anchor_not_recorded')
        elif contained:
            reasons.append('negative_contains_recorded_tss')
        if reasons:
            violations.append({'name': row['name'], 'label': int(row['label']),
                               'reasons': reasons, 'contained_tss_1based': [tss + 1 for tss in contained]})
    return violations


def build_groups(rows, cluster_tables=(), tss_table=None, allowed_missing=()):
    """Join same-strand overlaps, associated ORFs, duplicates and known clusters.

    Opposite-strand windows remain independent by explicit policy. Grouping is
    transitive and joint across labels. TSS CSV is one-based; FASTA is zero-based.
    """
    names = [row['name'] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate sequence identifiers.')
    intervals = [coordinates(row) for row in rows]
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(a, b):
        parent[find(b)] = find(a)

    for i, (start, end, strand) in enumerate(intervals):
        for j in range(i):
            other_start, other_end, other_strand = intervals[j]
            if strand == other_strand and max(start, other_start) < min(end, other_end):
                join(i, j)

    complement = str.maketrans('ACGTN', 'TGCAN')
    duplicates = {}
    for i, row in enumerate(rows):
        sequence = row['sequence'].upper()
        canonical = min(sequence, sequence.translate(complement)[::-1])
        key = (intervals[i][2], canonical)
        if key in duplicates:
            join(i, duplicates[key])
        else:
            duplicates[key] = i

    index = {name: i for i, name in enumerate(names)}
    for table in cluster_tables:
        clusters = {}
        with Path(table).open() as handle:
            for fields in csv.reader(handle, delimiter='\t'):
                if not fields:
                    continue
                if len(fields) != 2 or any(name not in index and name not in allowed_missing for name in fields):
                    raise ValueError(f'Invalid cluster entry in {table}: {fields}')
                representative, member = fields
                if member not in index:
                    continue
                i = index[member]
                key = (representative, intervals[i][2])
                if key in clusters:
                    join(i, clusters[key])
                else:
                    clusters[key] = i

    if tss_table is not None:
        with Path(tss_table).open() as handle:
            annotations = {(int(row['TSS']) - 1, row['strand']): row
                           for row in csv.DictReader(handle)}
        genes = {}
        for i, row in enumerate(rows):
            if str(row['label']) != '1':
                continue
            strand = intervals[i][2]
            annotation = annotations[(int(row['name'].split('|')[0]), strand)]
            key = (strand, annotation['orf_st'], annotation['orf_ed'])
            if key in genes:
                join(i, genes[key])
            else:
                genes[key] = i

    members = {}
    for i, name in enumerate(names):
        members.setdefault(find(i), []).append(name)
    stable_ids = {root: min(group) for root, group in members.items()}
    return [stable_ids[find(i)] for i in range(len(rows))]


def _holdout(rows, groups, fraction, seed):
    """Choose a grouped partition closest to per-class row targets, using no metrics."""
    labels = np.array([int(row['label']) for row in rows])
    counts = Counter(labels.tolist())
    if len(counts) != 2 or len(set(groups)) < 3:
        raise ValueError('Need both classes and at least three independent groups.')
    best = None
    splitter = GroupShuffleSplit(n_splits=512, test_size=fraction, random_state=seed)
    for train, test in splitter.split(np.arange(len(rows)), labels, groups):
        if len(set(labels[train])) != 2 or len(set(labels[test])) != 2:
            continue
        error = sum(abs(int(np.sum(labels[test] == label)) - count * fraction)
                    for label, count in counts.items())
        if best is None or error < best[0]:
            best = (error, train, test)
    if best is None:
        raise ValueError('Cannot split groups while retaining both classes.')
    return best[1], best[2]


def split_rows(rows, groups, test_size=0.2, validation_size=0.2, seed=42):
    """Return train/validation/test rows without separating any group."""
    if not 0 < test_size < 1 or not 0 < validation_size < 1 - test_size:
        raise ValueError('Require positive holdouts whose fractions sum to less than one.')
    if len(rows) != len(groups):
        raise ValueError('One group required per row.')
    remaining, test = _holdout(rows, groups, test_size, seed)
    remaining_rows = [rows[i] for i in remaining]
    remaining_groups = [groups[i] for i in remaining]
    train, validation = _holdout(remaining_rows, remaining_groups,
                                validation_size / (1 - test_size), seed + 1)
    indices = {'train': remaining[train], 'validation': remaining[validation], 'test': test}
    return {name: [dict(rows[i], group=groups[i]) for i in idx]
            for name, idx in indices.items()}


def cross_split_violations(splits, tss_table=None):
    """Report same-strand overlaps, group leakage and duplicate sequence leakage."""
    violations = []
    complement = str.maketrans('ACGTN', 'TGCAN')
    annotations = {}
    if tss_table is not None:
        with Path(tss_table).open() as handle:
            annotations = {(int(row['TSS']) - 1, row['strand']):
                           (row['orf_st'], row['orf_ed']) for row in csv.DictReader(handle)}
    names = list(splits)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            for a in splits[first]:
                start, end, strand = coordinates(a)
                sequence = a['sequence'].upper()
                for b in splits[second]:
                    other_start, other_end, other_strand = coordinates(b)
                    reasons = []
                    if a.get('group') is not None and a['group'] == b.get('group'):
                        reasons.append('shared_group')
                    if strand == other_strand:
                        if str(a['label']) == str(b['label']) == '1' and annotations:
                            gene_a = annotations[(int(a['name'].split('|')[0]), strand)]
                            gene_b = annotations[(int(b['name'].split('|')[0]), strand)]
                            if gene_a == gene_b:
                                reasons.append('same_associated_orf')
                        if max(start, other_start) < min(end, other_end):
                            reasons.append('same_strand_overlap')
                        if b['sequence'].upper() in (sequence, sequence.translate(complement)[::-1]):
                            reasons.append('same_strand_duplicate')
                    if reasons:
                        violations.append({'splits': [first, second], 'names': [a['name'], b['name']],
                                           'reasons': reasons})
    return violations
