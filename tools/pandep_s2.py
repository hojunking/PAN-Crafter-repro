#!/usr/bin/env python3
"""Deploy the Git-shipped PANDEP snapshot to its isolated sibling and launch it.

Pulling Git never starts training. Explicit ``start`` is required. This stdlib-only
wrapper never imports legacy training code, reads credential contents, modifies an
existing deployment, downloads an image, or terminates another experiment.
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile

CAMPAIGN = 'PANDEP_S2_SHARED_PLH_20261001_v4'
COMMANDS = ('start', 'status', 'pause', 'resume', 'stop', 'sync-sheet')
REPO = Path(__file__).resolve().parents[1]


class DeploymentError(RuntimeError):
    pass


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def safe_relative(value):
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value:
        raise DeploymentError('BLOCKED_DISTRIBUTION: invalid relative file path')
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or str(path) != value or any(part in ('.', '..') for part in path.parts):
        raise DeploymentError('BLOCKED_DISTRIBUTION: noncanonical or escaping file path: ' + value)
    return path


def regular_path(root, relative):
    path = Path(root)
    if path.is_symlink() or not path.is_dir():
        raise DeploymentError('BLOCKED_PATH: project root must be a real directory')
    parts = safe_relative(relative).parts
    for index, part in enumerate(parts):
        path = path / part
        if path.is_symlink():
            raise DeploymentError('BLOCKED_PATH: tracked file or ancestor is a symlink: ' + str(path))
        if index < len(parts) - 1 and not path.is_dir():
            raise DeploymentError('BLOCKED_PATH: missing tracked parent: ' + str(path))
    if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise DeploymentError('BLOCKED_PATH: missing/nonregular tracked file: ' + str(path))
    return path


def verify_files(project, manifest, *, exact=False):
    for relative, identity in manifest['files'].items():
        path = regular_path(project, relative)
        if path.stat().st_size != identity['size_bytes'] or file_sha(path) != identity['sha256']:
            raise DeploymentError('BLOCKED_DISTRIBUTION: tracked file differs; never overwritten: ' + str(path))
    if exact:
        present = set()
        for directory, dirs, files in os.walk(project, followlinks=False):
            for name in dirs + files:
                path = Path(directory) / name
                if path.is_symlink():
                    raise DeploymentError('BLOCKED_DISTRIBUTION: symlinks are forbidden in the Git payload')
            present.update(str((Path(directory) / name).relative_to(project)) for name in files)
        if present != set(manifest['files']):
            raise DeploymentError('BLOCKED_DISTRIBUTION: payload file inventory differs from manifest')
    else:
        # Preserve generated artifacts and notes, but do not execute an otherwise
        # matching deployment with unverified import hooks/packages or vendor code.
        # Existing __pycache__ is permitted; normal Python validates cached sources.
        ignored_roots = {'work_dir', 'dist', '.git', '.venv', '__pycache__'}
        for directory, dirs, files in os.walk(project, followlinks=False):
            relative_directory = Path(directory).relative_to(project)
            dirs[:] = [name for name in dirs if name != '__pycache__'
                       and not (relative_directory == Path('.') and name in ignored_roots)]
            for name in dirs:
                if (Path(directory) / name).is_symlink():
                    raise DeploymentError('BLOCKED_DISTRIBUTION: unverified import directory symlink')
            for name in files:
                path = Path(directory) / name
                relative = str(path.relative_to(project))
                if relative in manifest['files']:
                    continue
                vendor_source = relative.startswith('vendor_reference/') and path.suffix not in ('.pyc', '.pyo')
                import_code = path.suffix in ('.py', '.pyc', '.pyo')
                if vendor_source or import_code:
                    raise DeploymentError('BLOCKED_DISTRIBUTION: unverified executable/source file: ' + str(path))


def read_distribution(repo):
    directory = Path(repo) / 'deploy/pandep_shared'
    # Check every ancestor too: no redirect to another distribution tree.
    manifest_path = regular_path(repo, 'deploy/pandep_shared/manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('schema') != 'PANDEP_GIT_DISTRIBUTION_v1'
            or manifest.get('campaign_id') != CAMPAIGN or not isinstance(manifest.get('files'), dict)
            or not manifest['files']):
        raise DeploymentError('BLOCKED_DISTRIBUTION: unknown manifest/campaign')
    for relative, identity in manifest['files'].items():
        safe_relative(relative)
        if (not isinstance(identity, dict) or set(identity) != {'sha256', 'size_bytes'}
                or not re.fullmatch(r'[0-9a-f]{64}', str(identity['sha256']))
                or type(identity['size_bytes']) is not int or identity['size_bytes'] < 0):
            raise DeploymentError('BLOCKED_DISTRIBUTION: invalid file identity')
    project = directory / 'project'
    verify_files(project, manifest, exact=True)
    receipt_path = regular_path(project, 'verification/local_cpu_tests.json')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    tests = receipt.get('unit_tests', {})
    if (receipt.get('schema') != 'PANDEP_LOCAL_CPU_VERIFICATION_v1' or receipt.get('status') != 'PASS'
            or tests.get('success') is not True or tests.get('returncode') != 0
            or tests.get('test_count', 0) < 1 or any(tests.get(key) for key in ('errors', 'failures', 'skipped'))):
        raise DeploymentError('BLOCKED_DISTRIBUTION: passing local CPU verification receipt required')
    source = receipt.get('source', {})
    source_body = {key: value for key, value in source.items() if key != 'source_sha256'}
    if (source.get('schema') != 'PANDEP_SOURCE_v1' or not source.get('files')
            or canonical_sha(source_body) != source.get('source_sha256')
            or source.get('source_sha256') != manifest.get('source_sha256')):
        raise DeploymentError('BLOCKED_DISTRIBUTION: verified source identity differs')
    expected_sources = {relative for relative in manifest['files']
                        if relative in ('launch.py', 'pyproject.toml', 'requirements.lock')
                        or relative.startswith('vendor_reference/')
                        or (relative.startswith(('src/', 'tests/')) and relative.endswith('.py'))}
    if expected_sources != set(source['files']) or 'launch.py' not in expected_sources:
        raise DeploymentError('BLOCKED_DISTRIBUTION: source receipt file inventory differs')
    for relative, sha in source['files'].items():
        if manifest['files'][relative]['sha256'] != sha:
            raise DeploymentError('BLOCKED_DISTRIBUTION: source receipt hash differs: ' + relative)
    return project, manifest


def resolve_target(repo, target=None):
    repo = Path(repo).resolve(strict=True)
    target = Path(target) if target is not None else repo.parent / 'pan_deploy_shared'
    target = Path(os.path.abspath(target))
    if target.is_symlink() or target.parent.resolve(strict=True) != repo.parent or target.name in ('', '.', '..'):
        raise DeploymentError('BLOCKED_PATH: deployment must be a real sibling directory of the legacy repository')
    target = target.parent.resolve(strict=True) / target.name
    if target == repo or target == target.parent:
        raise DeploymentError('BLOCKED_PATH: deployment must not overlap the legacy repository')
    if target.exists() and not target.is_dir():
        raise DeploymentError('BLOCKED_PATH: deployment target is not a directory')
    return repo, target


def publish_noreplace(staged, target):
    """Linux atomic directory publication; never replace a concurrently created path."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    if rename is None:
        raise DeploymentError('BLOCKED_PLATFORM: atomic renameat2(NOREPLACE) is required')
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staged), -100, os.fsencode(target), 1):
        error = ctypes.get_errno()
        raise DeploymentError('BLOCKED_PUBLICATION: target was not replaced: ' + os.strerror(error))


def deploy(project, manifest, target):
    target = Path(target)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(target.parent / ('.' + target.name + '.pandep-deploy.lock'), flags, 0o600)
    staged = None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise DeploymentError('BLOCKED_LOCK: deployment lock is not regular')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DeploymentError('BLOCKED_LOCK: another PANDEP deployment is publishing') from error
        if target.exists() or target.is_symlink():
            verify_files(target, manifest)
            return False
        staged = Path(tempfile.mkdtemp(prefix='.' + target.name + '.stage-', dir=target.parent))
        for relative in sorted(manifest['files']):
            source = regular_path(project, relative)
            destination = staged / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Copy bytes, not symlinks, metadata, secrets, generated data, or old work_dir.
            with source.open('rb') as incoming, destination.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            destination.chmod(stat.S_IMODE(source.stat().st_mode) & 0o777)
        verify_files(staged, manifest, exact=True)
        publish_noreplace(staged, target)
        staged = None
        directory_fd = os.open(target.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return True
    finally:
        if staged is not None:
            # Only the exact private temporary directory created in this invocation.
            shutil.rmtree(staged)
        os.close(fd)


def launcher_argv(command, repo, target, credentials=None, microbatch=None):
    if command not in COMMANDS:
        raise DeploymentError('BLOCKED_COMMAND: unregistered command')
    args = [sys.executable, str(target / 'launch.py'), '--legacy-root', str(repo)]
    if command in ('start', 'resume', 'sync-sheet'):
        credential = Path(credentials) if credentials is not None else repo / 'gspread/account.json'
        args += ['--credentials', str(credential.resolve())]
    args.append(command)
    if microbatch is not None:
        if command not in ('start', 'resume') or microbatch not in (12, 24, 48):
            raise DeploymentError('BLOCKED_RECIPE: microbatch is only 12, 24, or 48 for start/resume')
        args += ['--microbatch', str(microbatch)]
    return args


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=COMMANDS)
    parser.add_argument('--legacy-root', type=Path, default=REPO)
    parser.add_argument('--project-root', type=Path, help='isolated sibling path (default: ../pan_deploy_shared)')
    parser.add_argument('--credentials', type=Path, help='credential path only; never copied or read by this wrapper')
    parser.add_argument('--microbatch', type=int, choices=(12, 24, 48))
    parser.add_argument('--dry-run', action='store_true', help='read-only plan; no file creation or Docker calls')
    args = parser.parse_args(argv)
    try:
        repo, target = resolve_target(args.legacy_root, args.project_root)
        project, manifest = read_distribution(repo)
        identity = repo / 'gspread/server.txt'
        server = identity.read_text(encoding='utf-8').strip() if identity.is_file() else None
        if not args.dry_run and server != 's2':
            raise DeploymentError('BLOCKED_SERVER: this command requires gspread/server.txt = s2')
        exists = target.exists()
        if exists:
            verify_files(target, manifest)
        elif args.command != 'start':
            raise DeploymentError('NOT_DEPLOYED: run start on s2 first; control commands never create deployments')
        command = launcher_argv(args.command, repo, target, args.credentials, args.microbatch)
        if args.dry_run:
            print(json.dumps({'status': 'DRY_RUN', 'campaign_id': CAMPAIGN,
                              'source_sha256': manifest['source_sha256'], 'payload_file_count': len(manifest['files']),
                              'deployment_target': str(target), 'would_deploy': not exists,
                              'server': server, 'server_would_block': server != 's2',
                              'launcher_argv': command, 'writes': False, 'docker_executed': False},
                             ensure_ascii=False, indent=2))
            return 0
        deployed = deploy(project, manifest, target)
        print(('Deployed verified snapshot to ' if deployed else 'Verified existing isolated snapshot at ')
              + str(target), flush=True)
        result = subprocess.run(command, cwd=target, check=False)
        return result.returncode
    except (DeploymentError, OSError, ValueError, TypeError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
