"""Local, explicit Docker launcher with immutable content-addressed source.

No git write, image pull, server discovery, cron install or process termination.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from panda_rb.common import ROOT, atomic_json, locked, read, sha256, server_dir
from panda_rb.plan import PLAN_DIR, binding_path, campaign_dir, validate_plan

DEFAULT_IMAGE = 'sha256:ebe266ad6514c1602b423518f77bf87e57e9f6e51cca9d2ac21581a64d106887'
SOURCE_DIRS = ('panda_rb', 'fh12', 'fh20r1', 'model', 'pa', 'tools', 'reporting_extra',
               'feeders', 'align', 'kd', 'kdv', 'sr', 'uvs')


def git_mounts(root=ROOT):
    """Read-only Git identity/objects for isolated original-reader parity.

    Worktrees have a .git *file* pointing to a separate gitdir and commondir;
    mounting that file alone is insufficient for `git archive <origin SHA>`.
    """
    root = Path(root).resolve()
    pointer = root / '.git'
    if not pointer.exists():
        raise ValueError('Frozen B01 runtime requires the original Git object database')
    paths = {pointer}
    for option in ('--absolute-git-dir', '--git-common-dir'):
        value = subprocess.check_output(['git', 'rev-parse', option], cwd=root, text=True).strip()
        path = Path(value)
        paths.add((root / path).resolve() if not path.is_absolute() else path.resolve())
    return sorted(paths)


def legacy_processes(root=ROOT, proc_root=Path('/proc')):
    """Detect known local PAN controllers even while they have no GPU context.

    Return PID/script only, not complete process arguments (which may contain
    unrelated private values). This never signals or terminates any process.
    """
    root = Path(root).resolve(); found = []
    for entry in Path(proc_root).iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = [part.decode(errors='replace') for part in (entry / 'cmdline').read_bytes().split(b'\0') if part]
            cwd = (entry / 'cwd').resolve()
        except (FileNotFoundError, PermissionError, OSError):
            continue
        if not args:
            continue
        # Other projects are not claimed merely because they use runner.py.
        local = cwd == root or cwd.is_relative_to(root) or (
            cwd.parent == root.parent and cwd.name.startswith('PAN-Crafter'))
        local = local or any(str(root) in token for token in args[1:] if '\n' not in token)
        if not local:
            continue
        for token in args[1:]:
            if any(c.isspace() for c in token):
                continue
            name = Path(token).name
            known = name in ('_run_cases.sh', '_watchdog.sh', 'main.py', 'run.sh') or (
                name.endswith('_runner.py') and name != 'panda_rb_runner.py')
            if known:
                found.append(dict(pid=int(entry.name), script=name)); break
    return found


def source_files(root=ROOT):
    root = Path(root)
    files = list(root.glob('*.py'))
    for name in SOURCE_DIRS:
        files.extend((root / name).rglob('*.py'))
    files.extend((root / 'config').glob('*.yaml'))
    files.extend(p for p in (root / PLAN_DIR).rglob('*') if p.is_file())
    return {str(p.relative_to(root)): sha256(p) for p in sorted(set(files))}


def snapshot(root=ROOT):
    root = Path(root).resolve(); files = source_files(root)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    parent = campaign_dir(root) / 'releases'; parent.mkdir(parents=True, exist_ok=True)
    target = parent / digest
    if target.exists():
        manifest = read(target / 'runtime_release.json')
        if manifest.get('files') != files: raise ValueError('Frozen source manifest changed')
        for name, expected in files.items():
            if sha256(target / name) != expected: raise ValueError('Frozen source bytes changed: ' + name)
        return target, digest
    temporary = Path(tempfile.mkdtemp(prefix='pending-', dir=parent))
    for name, expected in files.items():
        destination = temporary / name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, destination)
        if sha256(destination) != expected: raise ValueError('Source changed while freezing: ' + name)
    # Nested bind destinations must exist before the parent source is mounted
    # read-only; Docker must not mutate a frozen release to create mountpoints.
    for name in ('work_dir', 'data', 'pan_h5'):
        (temporary / name).mkdir(exist_ok=True)
    if (root / '.git').is_dir():
        (temporary / '.git').mkdir(exist_ok=True)
    else:
        shutil.copyfile(root / '.git', temporary / '.git')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    atomic_json(temporary / 'runtime_release.json', dict(files=files, release_sha256=digest,
                                                       git_origin=commit, allows_dirty_worktree=True))
    os.rename(temporary, target)
    return target, digest


def _absolute_paths(value):
    if isinstance(value, dict):
        for item in value.values(): yield from _absolute_paths(item)
    elif isinstance(value, list):
        for item in value: yield from _absolute_paths(item)
    elif isinstance(value, str) and value.startswith('/'):
        path = Path(value)
        if path.exists(): yield path


def asset_mounts(root=ROOT, binding=None, extra=()):
    """Expose only actual referenced existing assets; mounted read-only."""
    root = Path(root).resolve(); paths = set()
    # Existing symlinks still require their real absolute archive targets.
    candidates = [Path(binding or binding_path(root)), root / 'work_dir/_fh12/s1/dataset_manifest.json',
                  root / 'work_dir/_fh12/s1/references/FH12_S1_T_P0_W112_D123_WV3_S71001_FRESH50_v1/reference_manifest.json']
    for candidate in candidates:
        if candidate.is_file():
            paths.add(candidate.resolve())
            for path in _absolute_paths(read(candidate)):
                paths.add(path.resolve())
    for name in ('data', 'pan_h5'):
        path = root / name
        if path.exists():
            paths.add(path.resolve())
            for child in path.rglob('*'):
                if child.is_symlink() and child.exists(): paths.add(child.resolve())
    # The selected origin's init manifest can itself be an archived symlink.
    init = root / 'work_dir/FH12_S1_T_P0_W112_D123_WV3_S71001_FRESH50_v1/init_manifest.json'
    if init.exists(): paths.add(init.resolve())
    dlpan = Path(os.environ.get('PANCRAFTER_DLPAN', str(root.parent / 'DLPan-Toolbox')))
    if dlpan.is_dir(): paths.add(dlpan.resolve())
    paths.update(Path(p).resolve(strict=True) for p in extra)
    work = (root / 'work_dir').resolve()
    paths = {p for p in paths if not p.is_relative_to(work)}
    paths = {p for p in paths if not any(p != other and p.is_relative_to(other) for other in paths)}
    if any(p == Path('/') or p == Path.home() for p in paths):
        raise ValueError('Refusing broad asset mount; supply exact asset directories')
    return sorted(paths)


def start(server, root=ROOT, binding=None, image=DEFAULT_IMAGE, gpu='0', extra_assets=(),
          retry_technical=False, dry_run=False):
    root = Path(root).resolve(); control = server_dir(server, root); validate_plan(root)
    name = 'panda-rb-b01-' + server
    if dry_run:
        return dict(server=server, container=name, source_files=len(source_files(root)), image=image,
                    schedule_actions=25, training_runs=8, paired_curves=16,
                    creates_container=False, activates_legacy_guard=False,
                    binding=str(binding or binding_path(root)),
                    requires='Docker CUDA; original/imported common F1 and all-six-seed maps; free GPU')
    if image != DEFAULT_IMAGE:
        raise ValueError('Different image requires an explicitly revised common runtime policy, not a per-server exception')
    # One physical checkout cannot concurrently impersonate two server slots,
    # including the interval before the first container initializes CUDA.
    with locked(campaign_dir(root) / 'control/launch.lock'):
        owner_path = root / 'work_dir/_panda_rb/local_owner.json'
        if owner_path.exists() and read(owner_path).get('server') != server:
            raise RuntimeError('This checkout already belongs to a different B01 server slot')
        image_id = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image], text=True))[0]['Id']
        if image_id != DEFAULT_IMAGE: raise ValueError('Docker image bytes differ from pinned common runtime')
        containers = subprocess.check_output(['docker', 'ps', '--filter', 'label=panda_rb.server=' + server,
                                             '--filter', 'label=panda_rb.campaign=B01', '--format', '{{.ID}}'], text=True).strip()
        if containers: return dict(already_running=True, server=server, container=name, id=containers)
        if read(control / 'status.json').get('state') == 'STOP_FOR_REVIEW':
            return dict(state='STOP_FOR_REVIEW', server=server, admitted=False)
        if read(control / 'status.json').get('state') == 'TECHNICAL_FAILURE' and not retry_technical:
            raise RuntimeError('Inspect technical failure and explicitly use --retry-technical')
        busy = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        if busy: raise RuntimeError('GPU already occupied; no process is stopped or queued automatically: ' + busy)
        legacy = legacy_processes(root)
        if legacy:
            raise RuntimeError('Existing PAN queue/trainer owns admissions even without a CUDA context: ' + json.dumps(legacy))
        # Prevent an old host queue from restarting while the explicit B01 owner runs.
        if not (root / 'tools/_watchdog.sh').is_file() or 'PANDA_RB_LOCAL_OWNER' not in (root / 'tools/_watchdog.sh').read_text():
            raise RuntimeError('B01 legacy watchdog guard is not installed in the source checkout')
        release, digest = snapshot(root)
        previous = read(control / 'deployment.json')
        deployment = dict(server=server, image_id=image_id, release_sha256=digest, root=str(root), gpu=str(gpu),
                          binding=str(Path(binding or binding_path(root)).resolve()))
        if previous and previous != deployment:
            raise ValueError('Different source/image/binding cannot silently resume the same B01 cohort')
        atomic_json(control / 'deployment.json', deployment)
        command = ['docker', 'run', '--detach', '--name', name, '--network', 'none', '--pid', 'host',
                   '--label', 'panda_rb.server=' + server, '--label', 'panda_rb.campaign=B01',
                   '--gpus', 'device=' + str(gpu), '--user', f'{os.getuid()}:{os.getgid()}', '--shm-size', '8g',
                   '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'OMP_NUM_THREADS=2',
                   '-e', 'HF_HOME=/tmp/panda_rb_hf', '-e', 'MPLCONFIGDIR=/tmp/panda_rb_mpl',
                   '-e', 'PANDA_RB_IMAGE_ID=' + image_id, '-e', 'PYTHONUNBUFFERED=1',
                   '-v', f'{release}:{root}:ro', '-v', f'{root / "work_dir"}:{root / "work_dir"}:rw']
        for path in git_mounts(root): command += ['-v', f'{path}:{path}:ro']
        for path in asset_mounts(root, binding, extra_assets): command += ['-v', f'{path}:{path}:ro']
        if os.environ.get('PANCRAFTER_DLPAN'):
            command += ['-e', 'PANCRAFTER_DLPAN=' + str(Path(os.environ['PANCRAFTER_DLPAN']).resolve())]
        command += ['-w', str(root), '--entrypoint', 'python', image_id, 'tools/panda_rb_runner.py',
                    'run', '--server', server, '--binding', deployment['binding']]
        if retry_technical: command += ['--retry-technical']
        # Never remove an old stopped container and its logs. Use a new attempt name.
        names = subprocess.check_output(['docker', 'ps', '-a', '--format', '{{.Names}}'], text=True).splitlines()
        attempt = 1
        while name in names:
            attempt += 1; name = f'panda-rb-b01-{server}-attempt{attempt}'
        command[command.index('--name') + 1] = name
        atomic_json(root / 'work_dir/_panda_rb/local_owner.json', dict(server=server, campaign='B01',
                    container=name, no_legacy_fallback=True, permanent_until_explicit_release=True))
        # Recheck after publishing the opt-in guard to catch an old admission
        # that raced the first read-only process/GPU check; never kill it.
        legacy = legacy_processes(root)
        if legacy:
            raise RuntimeError('PAN admission raced the B01 owner guard; retained hold, no process stopped: ' + json.dumps(legacy))
        identifier = subprocess.check_output(command, text=True).strip()
        return dict(started=True, container=name, container_id=identifier, deployment=deployment,
                    stop_after='8 registered Students + 16 paired curves; STOP_FOR_REVIEW')
