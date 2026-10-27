"""Exercise archive replay, corruption detection, and source-change protection."""
import importlib.util
import json
import os
import tarfile
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('archive_tool', Path(__file__).resolve().parents[1] / 'scripts/archive_project_artifacts.py')
archive_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive_tool)


def fixture_plan(tmp_path):
    source = tmp_path / 'source'
    (source / 'run' / 'empty').mkdir(parents=True)
    (source / 'run' / 'weights.bin').write_bytes(bytes(range(256)) * 200)
    (source / 'run' / 'state.json').write_text('{"step": 21}')
    (source / 'run' / 'link').symlink_to('state.json')
    os.link(source / 'run/state.json', source / 'run/state-hardlink.json')
    plan = dict(source_root=str(source), archive_root=str(tmp_path / 'archive'),
                receipt_root=str(tmp_path / 'receipts'),
                packages=[dict(name='run', paths=['run'])])
    return plan


def test_roundtrip_and_replay_without_source(tmp_path):
    plan = fixture_plan(tmp_path)
    receipt = archive_tool.create(plan, plan['packages'][0])
    result = archive_tool.verify(receipt)
    assert result['verified'] and len(result['entries']) == 6
    restored = tmp_path / 'restored'
    with tarfile.open(result['archive']) as archive:
        archive.extractall(restored, filter='data')
    assert (restored / 'run/weights.bin').read_bytes() == (Path(plan['source_root']) / 'run/weights.bin').read_bytes()
    assert (restored / 'run/empty').is_dir()
    assert (restored / 'run/link').is_symlink()
    (Path(plan['source_root']) / 'run/state.json').unlink()
    original_receipt = receipt.read_bytes()
    archive_tool.verify(receipt, check_source=False)
    assert receipt.read_bytes() == original_receipt
    assert receipt.with_name('run.reverification.json').is_file()
    with pytest.raises(RuntimeError, match='Source member set'):
        archive_tool.verify(receipt)


def test_corrupt_archive_is_rejected(tmp_path):
    plan = fixture_plan(tmp_path)
    receipt = archive_tool.create(plan, plan['packages'][0])
    target = Path(json.loads(receipt.read_text())['archive'])
    with tarfile.open(target) as archive:
        offset = archive.getmember('run/weights.bin').offset_data
    with target.open('r+b') as data:
        data.seek(offset)
        data.write(b'corrupt')
    with pytest.raises(RuntimeError, match='Content mismatch'):
        archive_tool.verify(receipt)


def test_changed_source_is_rejected(tmp_path):
    plan = fixture_plan(tmp_path)
    receipt = archive_tool.create(plan, plan['packages'][0])
    (Path(plan['source_root']) / 'run/state.json').write_text('{"step": 22}')
    with pytest.raises(RuntimeError, match='Source changed'):
        archive_tool.verify(receipt)


@pytest.mark.parametrize('unsafe', ['../outside', '/etc/passwd', '.'])
def test_unsafe_source_is_rejected(tmp_path, unsafe):
    with pytest.raises(ValueError, match='Unsafe source'):
        list(archive_tool.entries(tmp_path, [unsafe]))

