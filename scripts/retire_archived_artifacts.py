"""Retire verified source artifacts only after their receipts are copied off-host."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from archive_project_artifacts import entries, signature, save


def retire(plan, acknowledgements, expected_root, execute=False, tracked=None):
    root = Path(expected_root).resolve(strict=True)
    if Path(plan['source_root']).resolve() != root:
        raise ValueError('Source root does not match the explicitly selected repository')
    if tracked is None:
        tracked = set(subprocess.check_output(
            ['git', 'ls-files', '-z'], cwd=root).decode().rstrip('\0').split('\0'))
    results = []
    for package in plan['packages']:
        name = package['name']
        if name not in acknowledgements:
            continue
        receipt = Path(plan['receipt_root']) / (name + '.json')
        receipt_hash = hashlib.sha256(receipt.read_bytes()).hexdigest()
        if receipt_hash != acknowledgements[name]:
            raise RuntimeError('Receipt differs from the off-host copy: ' + name)
        result = json.loads(receipt.read_text())
        cleanup = receipt.with_name(name + '.cleanup.json')
        if cleanup.exists():
            previous = json.loads(cleanup.read_text())
            if previous['receipt_sha256'] != receipt_hash:
                raise RuntimeError('Cleanup receipt changed: ' + name)
            results.append(previous)
            continue
        if not result.get('verified') or not result.get('source_checked'):
            raise RuntimeError('Archive/source verification is incomplete: ' + name)
        if result['package'] != package or Path(result['source_root']).resolve() != root:
            raise RuntimeError('Archive package identity mismatch: ' + name)
        target = Path(plan['archive_root']) / (name + '.tar')
        if Path(result['archive']) != target or not target.is_file():
            raise RuntimeError('Archive destination mismatch: ' + name)
        if target.stat().st_size != result['archive_bytes']:
            raise RuntimeError('Archive size changed: ' + name)
        verification_time = datetime.fromisoformat(result['verified_utc']).timestamp()
        if target.stat().st_mtime > verification_time:
            raise RuntimeError('Archive was modified after verification: ' + name)
        expected = {r['path']: r for r in result['entries']}
        actual = {relative: path for path, relative in entries(root, package['paths'])}
        if set(actual) != set(expected):
            raise RuntimeError('Source membership changed: ' + name)
        for relative, path in actual.items():
            if signature(path) != expected[relative]['source_stat']:
                raise RuntimeError('Source changed: ' + relative)
        targets = []
        for relative in package['paths']:
            path = root / relative
            parent = path.parent.resolve()
            if path == root or (parent != root and root not in parent.parents):
                raise ValueError('Removal escapes the selected repository')
            if '.git' in Path(relative).parts or relative == '.':
                raise ValueError('Refusing to remove repository metadata')
            if any(t == relative or t.startswith(relative.rstrip('/') + '/') for t in tracked):
                raise RuntimeError('Refusing to remove tracked content: ' + relative)
            targets.append(path)
        summary = dict(package=name, source_root=str(root), removed_paths=package['paths'],
                       receipt_sha256=receipt_hash, archive_sha256=result['archive_sha256'],
                       source_bytes=result['source_bytes'], executed=execute)
        if execute:
            for path in targets:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            summary['completed_utc'] = datetime.now(timezone.utc).isoformat()
            save(cleanup, summary)
        results.append(summary)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan')
    parser.add_argument('acknowledgements', help='Package to SHA-256 of its off-host receipt copy')
    parser.add_argument('--expected-root', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    results = retire(json.loads(Path(args.plan).read_text()),
                     json.loads(Path(args.acknowledgements).read_text()),
                     args.expected_root, execute=args.execute)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

