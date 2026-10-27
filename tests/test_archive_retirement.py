"""Source retirement must require verified archives and off-host receipts."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from archive_project_artifacts import create, verify, save
from retire_archived_artifacts import retire


def prepared(tmp_path):
    root = tmp_path / 'repo'
    (root / 'runs' / 'run1').mkdir(parents=True)
    (root / 'runs/run1/state').write_bytes(b'model and optimizer')
    plan = dict(source_root=str(root), archive_root=str(tmp_path / 'nearline'),
                receipt_root=str(tmp_path / 'control'),
                packages=[dict(name='run1', paths=['runs/run1'])])
    receipt = create(plan, plan['packages'][0])
    verify(receipt)
    acks = {'run1': hashlib.sha256(receipt.read_bytes()).hexdigest()}
    return root, plan, acks, receipt


def test_retirement_dry_run_execute_and_idempotency(tmp_path):
    root, plan, acks, receipt = prepared(tmp_path)
    retire(plan, acks, root, tracked=set())
    assert (root / 'runs/run1/state').exists()
    results = retire(plan, acks, root, execute=True, tracked=set())
    assert results[0]['executed'] and not (root / 'runs/run1').exists()
    assert receipt.with_name('run1.cleanup.json').exists()
    retire(plan, acks, root, execute=True, tracked=set())


def test_retirement_requires_matching_off_host_receipt(tmp_path):
    root, plan, acks, receipt = prepared(tmp_path)
    with pytest.raises(RuntimeError, match='off-host'):
        retire(plan, {'run1': 'invalid'}, root, execute=True, tracked=set())
    assert (root / 'runs/run1/state').exists()
    assert retire(plan, {}, root, execute=True, tracked=set()) == []


def test_retirement_preserves_tracked_and_changed_files(tmp_path):
    root, plan, acks, receipt = prepared(tmp_path)
    with pytest.raises(RuntimeError, match='tracked content'):
        retire(plan, acks, root, execute=True, tracked={'runs/run1/state'})
    (root / 'runs/run1/state').write_bytes(b'new training')
    with pytest.raises(RuntimeError, match='Source changed'):
        retire(plan, acks, root, execute=True, tracked=set())
    assert (root / 'runs/run1/state').read_bytes() == b'new training'


def test_retirement_rejects_wrong_root_and_unverified_archive(tmp_path):
    root, plan, acks, receipt = prepared(tmp_path)
    with pytest.raises(ValueError, match='Source root'):
        retire(plan, acks, tmp_path, execute=True, tracked=set())
    result = json.loads(receipt.read_text())
    result['verified'] = False
    save(receipt, result)
    acks['run1'] = hashlib.sha256(receipt.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match='incomplete'):
        retire(plan, acks, root, execute=True, tracked=set())

