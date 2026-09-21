"""Explicit-only background submission and HEAD-preserving portable source bundle."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile

from qg40.common import ROOT, camp, read


def _read_crontab():
    result = subprocess.run(['crontab', '-l'], capture_output=True, text=True,
                            timeout=15, env=dict(os.environ, LC_ALL='C'))
    if result.returncode == 0:
        return True, result.stdout
    if (result.returncode == 1 and not result.stdout.strip()
            and re.fullmatch(r'(?:crontab:\s*)?no crontab for [^\r\n]+', result.stderr.strip())):
        return False, ''
    raise RuntimeError('Cannot read existing crontab; no jobs changed')


def spawn(root, server):
    folder = camp(root, server)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'runner.log').open('a') as log:
        process = subprocess.Popen([sys.executable, str(Path(root) / 'tools/qg40_runner.py'),
                                    'run', '--server', server], cwd=root, stdout=log,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True, close_fds=True,
                                   env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    return dict(pid=process.pid, status='SUBMITTED', log=str(folder / 'runner.log'))


def install_recovery(root, server):
    """Manage only QG40's tagged cron line; never edit active WV3 scripts/jobs."""
    root = Path(root).resolve()
    for part in (str(root), sys.executable):
        if any(ch in part for ch in ('\n', '\r', '%')):
            raise ValueError('Unsupported cron path')
    existed, original = _read_crontab()
    script = root / 'tools/qg40_runner.py'
    command = ' '.join(shlex.quote(x) for x in (sys.executable, str(script), 'ensure', '--server', server))
    tag = f'# PANDA-QG40-{server}-{hashlib.sha256(str(root).encode()).hexdigest()[:12]}'
    entries = [f'*/5 * * * * {command} {tag}\n', f'@reboot sleep 120 && {command} {tag}\n']
    own = [line for line in original.splitlines(keepends=True) if line.rstrip().endswith(tag)]
    if own == entries:
        return dict(status='ALREADY_INSTALLED', readback_verified=True)
    # Unexpected jobs with this exact tag are preserved, not silently replaced.
    if own:
        raise ValueError('Existing QG40 recovery entry differs; preserved')
    updated = original + ('\n' if original and not original.endswith('\n') else '') + ''.join(entries)
    if _read_crontab() != (existed, original):
        raise ValueError('Crontab changed during registration; no overwrite')
    subprocess.run(['crontab', '-'], input=updated, text=True, capture_output=True,
                   check=True, timeout=15)
    if _read_crontab() != (True, updated):
        raise ValueError('Recovery readback mismatch; inspect crontab')
    return dict(status='INSTALLED', readback_verified=True)


def bundle(root=ROOT):
    from qg40.plan import SOURCE_SHAS, CASES
    root = Path(root).resolve()
    paths = sorted((root / 'qg40').glob('*.py')) + sorted((root / 'tools').glob('qg40_*.py'))
    paths += sorted((root / 'tools').glob('qg40_*.sh'))
    paths += [root / 'reporting_extra' / name for name in
              ('__init__.py', 'sensor_sheet.py', 'sensor_layout.py')]
    paths += [root / name for name in SOURCE_SHAS]
    paths += [root / 'qg40/README.md']
    review = root / 'qg40/IMPLEMENTATION.md'
    if review.exists() or review.is_symlink():
        paths.append(review)
    for name in ('qg40/sensor_sources.json', 'qg40/SENSOR_SOURCE_EVIDENCE.md'):
        source = root / name
        if source.exists() or source.is_symlink():
            paths.append(source)
    paths += [root / 'config/qg40' / (case.run_id + '.yaml') for case in CASES]
    paths += [root / 'config/qg40/QG40_Registry.json']
    for path in paths:
        if (not path.is_file() or any(p.is_symlink() for p in (path, *path.parents))
                or root not in path.resolve().parents):
            raise ValueError('Missing or unsafe QG40 bundle entry: ' + str(path))
    entries = {str(p.relative_to(root)): p.read_bytes() for p in paths}
    runtime_window = root / 'work_dir/_qg40/campaign_window.json'
    packaged_window = root / 'qg40/campaign_window.json'
    for window in (runtime_window, packaged_window):
        if any(p.is_symlink() for p in (window, *window.parents)):
            raise ValueError('Unsafe QG40 shared window path')
    if runtime_window.is_file() or packaged_window.is_file():
        from qg40.controller import shared_window
        shared_window(root)  # Both copies, when present, must name one common clock.
        window = runtime_window if runtime_window.is_file() else packaged_window
        entries['qg40/campaign_window.json'] = window.read_bytes()
    manifest = dict(schema='QG40_SOURCE_BUNDLE_v1', keep_existing_git_head=True,
                    files={key: hashlib.sha256(value).hexdigest() for key, value in entries.items()})
    entries['qg40/package_manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    destination = root / 'work_dir/_qg40/deployment/qg40-overlay.tar.gz'
    if any(p.is_symlink() for p in (destination, *destination.parents)):
        raise ValueError('Unsafe QG40 bundle destination')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, 'w:gz') as tf:
        for name, value in entries.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(value), 0o644
            tf.addfile(info, io.BytesIO(value))
    return dict(path=str(destination), sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                file_count=len(entries), activation_performed=False)
