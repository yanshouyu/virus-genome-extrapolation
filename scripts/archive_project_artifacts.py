"""Create and verify cold-storage tar archives; never deletes source files.

JSON plan: source_root, archive_root, receipt_root, packages [{name, paths}].
Paths remain source-relative; files are hashed on copy and independent replay.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

BLOCK = 8 * 1024 * 1024


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def signature(path):
    s = path.lstat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, mode=s.st_mode,
                inode=s.st_ino, device=s.st_dev)


def entries(root, paths):
    seen = set()
    for relative in paths:
        p = Path(relative)
        if p.is_absolute() or '..' in p.parts or str(p) == '.':
            raise ValueError(f'Unsafe source path: {relative}')
        source = root / p
        if source.parent.resolve() != root and root not in source.parent.resolve().parents:
            raise ValueError(f'Source escapes root: {source}')
        candidates = [source]
        if source.is_dir() and not source.is_symlink():
            for base, dirs, files in os.walk(source, followlinks=False):
                candidates.extend(Path(base) / n for n in sorted(dirs + files))
        for candidate in candidates:
            name = candidate.relative_to(root).as_posix()
            if name not in seen:
                seen.add(name)
                yield candidate, name


class HashedReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, length=-1):
        data = self.stream.read(length)
        self.digest.update(data)
        return data


class HashedWriter:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def write(self, data):
        written = self.stream.write(data)
        if written != len(data):
            raise IOError('Short archive write')
        self.digest.update(data)
        return written


def create(plan, package):
    root = Path(plan['source_root']).resolve(strict=True)
    target = Path(plan['archive_root']) / (package['name'] + '.tar')
    receipt = Path(plan['receipt_root']) / (package['name'] + '.json')
    if target.exists() or receipt.exists():
        raise FileExistsError(f'Refusing to overwrite {target} or {receipt}')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix('.tar.partial')
    records = []
    with partial.open('xb') as output:
        writer = HashedWriter(output)
        with tarfile.open(fileobj=writer, mode='w|', copybufsize=BLOCK,
                          bufsize=BLOCK, format=tarfile.PAX_FORMAT) as archive:
            for path, name in entries(root, package['paths']):
                before = signature(path)
                info = archive.gettarinfo(str(path), arcname=name)
                record = dict(path=name, source_stat=before)
                if info.isfile():
                    with path.open('rb') as source:
                        reader = HashedReader(source)
                        archive.addfile(info, reader)
                    record.update(kind='file', sha256=reader.digest.hexdigest(), size=info.size)
                elif info.isdir():
                    archive.addfile(info)
                    record.update(kind='directory')
                elif info.issym() or info.islnk():
                    archive.addfile(info)
                    record.update(kind='symlink' if info.issym() else 'hardlink', link=info.linkname)
                else:
                    raise ValueError(f'Unsupported source type: {path}')
                if signature(path) != before:
                    raise RuntimeError(f'Source changed during archive: {path}')
                records.append(record)
        output.flush()
        os.fsync(output.fileno())
    partial.replace(target)
    result = dict(package=package, source_root=str(root), archive=str(target),
                  archive_sha256=writer.digest.hexdigest(), archive_bytes=target.stat().st_size,
                  source_bytes=sum(r.get('size', 0) for r in records), entries=records,
                  created_utc=datetime.now(timezone.utc).isoformat(), verified=False)
    save(receipt, result)
    print(f'CREATED {target.name}: {result["source_bytes"]} source bytes', flush=True)
    return receipt


def verify(receipt, check_source=True):
    result = json.loads(Path(receipt).read_text())
    already_verified = result.get('verified', False)
    expected = {r['path']: r for r in result['entries']}
    found = set()
    target = Path(result['archive'])
    with target.open('rb') as source:
        reader = HashedReader(source)
        with tarfile.open(fileobj=reader, mode='r|', bufsize=BLOCK) as archive:
            for member in archive:
                if member.name in found or member.name not in expected:
                    raise RuntimeError(f'Unexpected or duplicate member: {member.name}')
                found.add(member.name)
                entry = expected[member.name]
                if entry['kind'] == 'file':
                    if not member.isfile() or member.size != entry['size']:
                        raise RuntimeError(f'Type/size mismatch: {member.name}')
                    digest = hashlib.sha256()
                    with archive.extractfile(member) as contents:
                        while data := contents.read(BLOCK):
                            digest.update(data)
                    if digest.hexdigest() != entry['sha256']:
                        raise RuntimeError(f'Content mismatch: {member.name}')
                elif entry['kind'] == 'directory':
                    if not member.isdir():
                        raise RuntimeError(f'Directory mismatch: {member.name}')
                else:
                    correct_type = member.issym() if entry['kind'] == 'symlink' else member.islnk()
                    if not correct_type or member.linkname != entry['link']:
                        raise RuntimeError(f'Link mismatch: {member.name}')
        while reader.read(BLOCK):
            pass
    if found != set(expected) or reader.digest.hexdigest() != result['archive_sha256']:
        raise RuntimeError('Archive member set or whole-file checksum mismatch')
    if check_source:
        root = Path(result['source_root'])
        actual = {name for _, name in entries(root, result['package']['paths'])}
        if actual != set(expected):
            raise RuntimeError('Source member set changed since archive')
        for entry in expected.values():
            if signature(root / entry['path']) != entry['source_stat']:
                raise RuntimeError(f'Source changed: {entry["path"]}')
    result.update(verified=True, verified_utc=datetime.now(timezone.utc).isoformat(),
                  source_checked=check_source)
    if already_verified:
        result['original_receipt_sha256'] = hashlib.sha256(Path(receipt).read_bytes()).hexdigest()
        save(Path(receipt).with_name(Path(receipt).stem + '.reverification.json'), result)
    else:
        save(receipt, result)
    print(f'VERIFIED {target.name}: {len(found)} entries', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan')
    parser.add_argument('--workers', type=int, choices=[1, 2], default=1)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--without-source', action='store_true')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    if args.without_source and not args.verify_only:
        parser.error('--without-source requires --verify-only')
    def process(package):
        receipt = Path(plan['receipt_root']) / (package['name'] + '.json')
        if not args.verify_only:
            if receipt.exists() and json.loads(receipt.read_text()).get('verified'):
                print(f'Already verified: {package["name"]}', flush=True)
                return
            if not receipt.exists():
                create(plan, package)
        verify(receipt, check_source=not args.without_source)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(process, plan['packages']))


if __name__ == '__main__':
    main()

