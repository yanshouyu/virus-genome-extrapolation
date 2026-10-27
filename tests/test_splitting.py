import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest
from baculo_tss_probe.splitting import (
    build_groups, coordinates, split_rows, cross_split_violations, read_known_tss, label_violations,
)

ROOT = Path(__file__).resolve().parents[1]


def row(name, start, end, strand='+', label=1, sequence=None):
    return {'name': f'{name}|{start}:{end}|{strand}|{label}',
            'sequence': sequence or 'A' * (end-start), 'label': label}


def test_overlap_is_transitive_joint_across_labels_and_strand_specific():
    rows = [row('a', 0, 10), row('b', 8, 18, label=0, sequence='ACGTACGTAA'),
            row('c', 16, 26, sequence='TTTCAAAAGG'),
            row('opposite', 0, 10, strand='-'), row('touching', 26, 36, sequence='C'*10)]
    groups = build_groups(rows)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]
    assert groups[4] != groups[0]
    assert cross_split_violations({'train': [rows[0]], 'test': [rows[3]]}) == []
    assert cross_split_violations({'train': [rows[0]], 'test': [rows[1]]})


def test_duplicates_reverse_complements_and_conflicting_ids():
    rows = [row('a', 0, 10, sequence='AAAACCCCGG'),
            row('b', 30, 40, sequence='CCGGGGTTTT'),
            row('c', 60, 70, strand='-', sequence='AAAACCCCGG')]
    groups = build_groups(rows)
    assert groups[0] == groups[1] != groups[2]
    with pytest.raises(ValueError, match='Duplicate'):
        build_groups([rows[0], rows[0]])
    with pytest.raises(ValueError, match='coordinates'):
        coordinates(row('bad', -1, 9))


def test_associated_orf_grouping_uses_zero_based_tss(tmp_path):
    table = tmp_path/'tss.csv'
    table.write_text('TSS,strand,orf_st,orf_ed\n1,+,500,600\n31,+,500,600\n')
    rows = [row('0', 0, 10), row('30', 30, 40, sequence='C'*10)]
    assert len(set(build_groups(rows, tss_table=table))) == 1


def read_clean():
    return {name: list(csv.DictReader((ROOT/f'data/processed/clean_v2/{name}.csv').open()))
            for name in ('train', 'validation', 'test')}


def test_generated_dataset_has_no_leakage_and_preserves_original_rows():
    splits = read_clean()
    assert cross_split_violations(splits, ROOT/'data/interim/tss_seq_range.csv') == []
    old = [r for name in ('train', 'test')
           for r in csv.DictReader((ROOT/f'data/processed/{name}.csv').open())]
    new = [r for records in splits.values() for r in records]
    manifest = json.loads((ROOT/'data/processed/clean_v2/manifest.json').read_text())
    excluded = {r['name'] for r in manifest['excluded_label_errors']}
    assert len(new) == 388 and len(old) == 391 and len(excluded) == 3
    assert {(r['name'], r['sequence'], int(r['label'])) for r in new} == {
        (r['name'], r['sequence'], int(r['label'])) for r in old if r['name'] not in excluded}
    assert [len(splits[n]) for n in ('train', 'validation', 'test')] == [233, 77, 78]
    known = read_known_tss(ROOT/'data/raw/baculovirus/Baculovirus_TSS.csv')
    assert label_violations(new, known) == []
    regenerated_groups = build_groups(new, [ROOT/'data/interim/cluster_tss_pos_clusters.tsv',
                                            ROOT/'data/interim/cluster_tss_neg_clusters.tsv'],
                                     ROOT/'data/interim/tss_seq_range.csv', allowed_missing=excluded)
    assert [r['group'] for r in new] == regenerated_groups
    # Check sequence extraction independently against the downloaded reference.
    gb = (ROOT/'data/raw/baculovirus/Baculoviridae/NC_001623.1.gb').read_text()
    genome = ''.join(c for c in gb.split('ORIGIN', 1)[1].split('//', 1)[0].upper() if c in 'ACGTN')
    complement = str.maketrans('ACGTN', 'TGCAN')
    for r in new:
        start, end, strand = coordinates(r)
        expected = genome[start:end]
        if strand == '-':
            expected = expected.translate(complement)[::-1]
        assert expected == r['sequence']


def test_split_reproducible_and_group_preserving():
    rows = [r for split in read_clean().values() for r in split]
    rows.sort(key=lambda r: r['name'])
    groups = build_groups(rows, tss_table=ROOT/'data/interim/tss_seq_range.csv')
    first = split_rows(rows, groups, seed=42)
    assert first == split_rows(rows, groups, seed=42)
    assert cross_split_violations(first) == []
    with pytest.raises(ValueError, match='fractions'):
        split_rows(rows, groups, validation_size=0.9)


def test_label_checks_find_excluded_positive_and_negative_errors():
    rows = [row('20', 0, 10), row('neg', 30, 40, label=0)]
    errors = label_violations(rows, {(20, '+'), (35, '+')})
    assert errors[0]['reasons'] == ['positive_anchor_outside_window']
    assert errors[1]['reasons'] == ['negative_contains_recorded_tss']


def test_known_cluster_constraints_and_unknown_members(tmp_path):
    rows = [row('a', 0, 10), row('b', 30, 40, sequence='C'*10),
            row('c', 50, 60, strand='-', sequence='T'*10)]
    table = tmp_path/'clusters.tsv'
    table.write_text(f"{rows[0]['name']}\t{rows[0]['name']}\n"
                     f"{rows[0]['name']}\t{rows[1]['name']}\n"
                     f"{rows[0]['name']}\t{rows[2]['name']}\n")
    groups = build_groups(rows, [table])
    assert groups[0] == groups[1] != groups[2]
    table.write_text('missing\tmissing\n')
    with pytest.raises(ValueError, match='cluster entry'):
        build_groups(rows, [table])


def test_cli_regenerates_identical_export(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'))
    subprocess.run([sys.executable, str(ROOT/'scripts/grouped_train_test_split.py'),
                    '--out-dir', str(tmp_path)], cwd=ROOT, env=env, check=True, capture_output=True)
    for name in ('train.csv', 'validation.csv', 'test.csv', 'train.fasta',
                 'validation.fasta', 'test.fasta', 'manifest.json'):
        assert (tmp_path/name).read_bytes() == (ROOT/'data/processed/clean_v2'/name).read_bytes()
