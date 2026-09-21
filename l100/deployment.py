"""Release manifest/bundle excludes data, credentials and old runtime artifacts."""
from pathlib import Path
import subprocess
import os
import tarfile

from l100.common import ROOT, atomic_json, object_sha, sha256, locked
from l100.plan import CAMPAIGN_ID, registry_sha256, verify_sources


def frozen_checkout(root, server):
    """Called only by explicit start: protect live sources from subsequent pull."""
    from l100.plan import SERVERS
    if server not in SERVERS:
        raise ValueError('LOCAL-T server must be s3..s5')
    root = Path(root).resolve()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    # A clean committed numerical release is required, not a clean unrelated
    # figure/log worktree. Never stash/reset the user's other changes.
    from l100.common import source_identity
    release_files = source_identity(root)['files']
    files = [name for name in release_files if not name.startswith('external/')]
    files += ['tools/l100_start.sh']
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z', '--', *files], cwd=root).decode().split('\0'))
    if set(files) - tracked:
        raise ValueError('Commit LOCAL-T sources before starting the frozen release')
    changed = subprocess.check_output(['git', 'diff', 'HEAD', '--name-only', '--', *files], cwd=root, text=True).splitlines()
    if changed:
        raise ValueError('Uncommitted numerical release files: ' + ', '.join(changed))
    target = root.parent / f'{root.name}-runtime-l100-{server}-{head[:12]}'
    with locked(root / 'work_dir/_l100/frozen_checkout.lock'):
        if not target.exists():
            subprocess.run(['git', 'worktree', 'add', '--detach', str(target), head], cwd=root, check=True)
        actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=target, text=True).strip()
        if actual != head:
            raise ValueError('Existing runtime checkout belongs to another release')
        if source_identity(target)['files'] != release_files:
            raise ValueError('Frozen checkout numerical files differ from the committed release')
        (root / 'work_dir').mkdir(exist_ok=True)
        for name in ('data', 'work_dir'):
            source, link = root / name, target / name
            if not source.exists():
                continue
            if link.is_symlink() and link.resolve() == source.resolve():
                continue
            if link.exists() or link.is_symlink():
                raise ValueError('Runtime data binding already exists and differs: ' + str(link))
            os.symlink(source.resolve(), link, target_is_directory=True)
        # gspread contains tracked helper scripts/backups; preserve those and
        # bind only local top-level credential JSON, never include it in git.
        (target / 'gspread').mkdir(exist_ok=True)
        for source in (root / 'gspread').glob('*.json'):
            link = target / 'gspread' / source.name
            if link.is_symlink() and link.resolve() == source.resolve():
                continue
            if not link.exists() and not link.is_symlink():
                os.symlink(source.resolve(), link)
    return target


def build_bundle(root=ROOT, output=None):
    root = Path(root).resolve()
    verify_sources(root)
    output = Path(output) if output else root / 'outputs/l100_release'
    output.mkdir(parents=True, exist_ok=True)
    # Only tracked files are deployable. Dirty tracked contents are rejected,
    # avoiding an accidental transfer of another campaign's local exception.
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    prefixes = ('l100/', 'config/l100/', 'fh12/', 'qg40/', 'model/', 'pa/', 'tools/metrics/',
                'reporting_extra/sensor_')
    exact = {'g20/__init__.py', 'g20/common.py', 'g20/data.py', 'g20/model.py', 'g20/evaluation.py',
        'g20/plan.py', 'g20/losses.py', 'g20/postrun.py', 'tools/l100_runner.py', 'tools/l100_start.sh',
        'tools/repair_lpan.py', 'tools/eval_dlpan.py', 'tools/repair_qb_ms.py', 'requirements.txt',
        'utils.py', 'reporting_extra/__init__.py'}
    from l100.plan import SOURCE_SHAS
    exact.update(SOURCE_SHAS)
    files = sorted(p for p in names if p and (p.startswith(prefixes) or p in exact)
                   and Path(p).name != 'campaign_window.json')
    changed = subprocess.check_output(['git', 'diff', 'HEAD', '--name-only', '--', *files], cwd=root, text=True).splitlines()
    if changed:
        raise ValueError('Commit the release sources before bundling: ' + ', '.join(changed))
    if not any(p.startswith('l100/') for p in files):
        raise ValueError('LOCAL-T implementation is not committed yet')
    hashes = {p: sha256(root / p) for p in files}
    manifest = dict(campaign_id=CAMPAIGN_ID, registry_sha256=registry_sha256(), files=hashes,
        content_sha256=object_sha(hashes), data_included=False, credentials_included=False,
        clock_included=False, launch_performed=False, kind='SOURCE_OVERLAY_ON_EXISTING_PANCRAFTER_CHECKOUT',
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip())
    path = output / 'l100_source.tar.gz'
    with tarfile.open(path, 'w:gz') as archive:
        for name in files:
            archive.add(root / name, arcname=name)
    atomic_json(output / 'manifest.json', dict(manifest, bundle_sha256=sha256(path)))
    return dict(path=str(path), manifest=str(output / 'manifest.json'), **manifest)
