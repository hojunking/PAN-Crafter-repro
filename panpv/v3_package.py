"""Bounded, non-executing intake for the user-supplied PANPV v3 ZIP."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

from .common import atomic_json, file_sha, load_json


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or '\\' in name or path.is_absolute() or
            any(p in ('', '.', '..') for p in name.rstrip('/').split('/'))):
        raise ValueError('Unsafe package member: ' + name)
    return path


def unpack(archive, destination, receipt, *, expected_bytes=None):
    """New destination only; ZIP CRC + every manifest SHA verified before acceptance.

    Hashes certify transfer consistency, not authenticity of provider assertions.
    No Python/scripts shipped in the ZIP are imported or executed.
    A failed extraction is left visibly incomplete, never admitted/reused.
    """
    archive, destination, receipt = map(Path, (archive, destination, receipt))
    if archive.is_symlink() or not archive.is_file():
        raise ValueError('Archive must be a regular local file')
    if expected_bytes is not None and archive.stat().st_size != expected_bytes:
        raise ValueError('Archive length differs from Drive metadata')
    if (destination.exists() or destination.is_symlink() or
            receipt.exists() or receipt.is_symlink()):
        raise ValueError('Use new extraction and receipt paths; never overwrite old inputs')
    if receipt.resolve().is_relative_to(destination.resolve()):
        raise ValueError('Receipt must be outside the immutable input tree')
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        if not members or len(members) > 10000:
            raise ValueError('Empty or unexpectedly large package inventory')
        names, roots = set(), set()
        total = 0
        for member in members:
            path = safe_name(member.filename)
            normalized = str(path)
            if normalized in names:
                raise ValueError('Duplicate package member: ' + normalized)
            names.add(normalized)
            roots.add(path.parts[0])
            kind = stat.S_IFMT(member.external_attr >> 16)
            if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError('Links and special files are not permitted')
            if member.flag_bits & 1:
                raise ValueError('Encrypted ZIP is not supported')
            total += member.file_size
        if len(roots) != 1 or total > 32 * 1024**3:
            raise ValueError('Unexpected archive root/expanded size')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(destination.parent).free < total + 10 * 1024**3:
            raise ValueError('Insufficient space for extraction plus 10 GiB reserve')
        archive_sha = file_sha(archive)
        destination.mkdir()
        extracted = {}
        for member in members:
            target = destination.joinpath(*safe_name(member.filename).parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            digest, count = hashlib.sha256(), 0
            with source.open(member) as src, target.open('xb') as out:
                while chunk := src.read(4 << 20):
                    count += len(chunk)
                    digest.update(chunk)
                    out.write(chunk)
            if count != member.file_size:
                raise ValueError('Expanded length mismatch')
            extracted[member.filename] = dict(bytes=count, sha256=digest.hexdigest())
            if target.suffix == '.h5':
                print('Extracted and CRC checked: ' + target.name, flush=True)
    root = destination / next(iter(roots))
    manifest = load_json(root / 'manifest.json')
    if manifest.get('schema') != 'PANPV_PORTABLE_PACKAGE_v1':
        raise ValueError('Unsupported package manifest')
    declared = {}
    for entry in manifest['files']:
        name = str(safe_name(entry['path']))
        if name in declared:
            raise ValueError('Duplicate manifest member')
        declared[name] = dict(bytes=entry['bytes'], sha256=entry['sha256'])
    checksums = {}
    for line in (root / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        sha, name = line.split('  ', 1)
        name = str(safe_name(name))
        if name in checksums or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('Invalid checksum manifest')
        checksums[name] = sha
    if checksums != {k: v['sha256'] for k, v in declared.items()}:
        raise ValueError('SHA256SUMS and package manifest disagree')
    actual = {str(PurePosixPath(k).relative_to(root.name)): v for k, v in extracted.items()}
    if set(actual) != set(declared) | {'manifest.json', 'SHA256SUMS'}:
        raise ValueError('Unlisted or missing package members')
    for name, metadata in declared.items():
        if actual[name] != metadata:
            raise ValueError('Package size/SHA mismatch: ' + name)
    result = dict(schema='PANPV_V3_LOCAL_PACKAGE_INTAKE_v1',
                  timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  state='TRANSFER_VERIFIED_NOT_TRAINING_ADMISSION',
                  archive=str(archive.resolve()), archive_sha256=archive_sha,
                  archive_bytes=archive.stat().st_size, input_root=str(root.resolve()),
                  extracted_bytes=total, crc_verified=True, manifest_verified=True,
                  files=actual, executed_provider_code=False, training_started=False)
    atomic_json(receipt, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--expected-bytes', type=int)
    args = parser.parse_args()
    result = unpack(args.archive, args.destination, args.receipt,
                    expected_bytes=args.expected_bytes)
    print(result['state'], result['archive_sha256'], flush=True)


if __name__ == '__main__':
    main()
