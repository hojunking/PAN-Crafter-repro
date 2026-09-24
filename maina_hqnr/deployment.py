"""Committed local release; existing user changes and historical runs are untouched."""
import os
from pathlib import Path
import subprocess

from maina_hqnr.common import ROOT, camp, read, locked, immutable_json, source_identity, verify_server
from maina_hqnr.plan import verify_sources


def _link(source, target):
    if not source.exists():
        return
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise ValueError('Frozen runtime asset link differs: ' + str(target))
        return
    if target.exists():
        if source.is_dir() and target.is_dir():
            for child in source.iterdir():
                _link(child, target / child.name)
        elif source.is_file() and target.is_file():
            from maina_hqnr.common import sha256
            if sha256(source) != sha256(target):
                raise ValueError('Existing runtime asset bytes differ')
        else:
            raise ValueError('Existing runtime asset type differs')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source.resolve(), target, target_is_directory=source.is_dir())


def frozen_checkout(root=ROOT, server='s4'):
    root = Path(root).resolve(); verify_server(server); verify_sources(root)
    with locked(camp(root, server) / 'deployment.lock'):
        receipt = read(camp(root, server) / 'runtime_release.json')
        if receipt:
            target = Path(receipt['path'])
            if target.parent != root.parent or not target.name.startswith(root.name + '-runtime-maina-hqnr-' + server + '-'):
                raise ValueError('Unexpected MAIN-A frozen release path')
            current = source_identity(target)
            if current['files'] != receipt['files'] or current['git_release'] != receipt['git_commit']:
                raise ValueError('Frozen source changed; a mixed-source block cannot resume')
            verify_sources(target)
            return target
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        files = source_identity(root)['files']
        names = [name for name in files if not name.startswith('external/')]
        names += [str(path.relative_to(root)) for path in (root / 'maina_hqnr').glob('test_*.py')]
        tracked = set(subprocess.check_output(['git', 'ls-files', '-z', '--', *names], cwd=root).decode().split('\0'))
        if set(names) - tracked:
            raise ValueError('Commit MAIN-A execution sources/tests before deployment: ' + str(sorted(set(names) - tracked)))
        dirty = subprocess.check_output(['git', 'diff', 'HEAD', '--name-only', '--', *names], cwd=root, text=True).splitlines()
        if dirty:
            raise ValueError('Uncommitted execution source: ' + ', '.join(dirty))
        target = root.parent / f'{root.name}-runtime-maina-hqnr-{server}-{commit[:12]}'
        if not target.exists():
            subprocess.run(['git', 'worktree', 'add', '--detach', str(target), commit], cwd=root, check=True)
        if source_identity(target)['files'] != files:
            raise ValueError('Frozen checkout differs from committed execution source')
        for name in ('data', 'work_dir', 'assets'):
            _link(root / name, target / name)
        (target / 'gspread').mkdir(exist_ok=True)
        for credential in (root / 'gspread').glob('*.json'):
            _link(credential, target / 'gspread' / credential.name)
        immutable_json(camp(root, server) / 'runtime_release.json',
                       dict(path=str(target), origin_root=str(root), git_commit=commit, files=files))
        return target
