"""One-command pinned local launch. No image pull, remote launch, or process kill."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from panda_rb.deployment import DEFAULT_IMAGE, SOURCE_DIRS, git_mounts, legacy_processes, asset_mounts
from panda_rb_m12.common import ROOT, atomic_json, locked, read, sha256, server_dir
from panda_rb_m12.plan import PLAN_DIR, binding_path, campaign_dir, validate_plan


def source_files(root=ROOT):
    root = Path(root)
    files = list(root.glob('*.py'))
    for folder in (*SOURCE_DIRS, 'panda_rb_m12', 'reporting_bridge'):
        files.extend((root / folder).rglob('*.py'))
    files.extend((root / 'config').glob('*.yaml'))
    for plan in (PLAN_DIR, Path('research_log/PANDA_REBUTTAL_STAGED_S135_2026-09-28')):
        files.extend(p for p in (root / plan).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    return {str(p.relative_to(root)): sha256(p) for p in sorted(set(files))}


def snapshot(root=ROOT):
    root = Path(root).resolve(); files = source_files(root)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    parent = campaign_dir(root) / 'releases'; parent.mkdir(parents=True, exist_ok=True)
    target = parent / digest
    if target.exists():
        manifest = read(target / 'runtime_release.json')
        if manifest.get('files') != files or any(sha256(target / name) != value for name, value in files.items()):
            raise ValueError('Frozen M12 source bytes changed')
        return target, digest
    temporary = Path(tempfile.mkdtemp(prefix='pending-', dir=parent))
    for name, value in files.items():
        destination = temporary / name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, destination)
        if sha256(destination) != value:
            raise ValueError('Source changed during snapshot')
    for name in ('work_dir', 'data', 'pan_h5'):
        (temporary / name).mkdir(exist_ok=True)
    if (root / '.git').is_dir():
        (temporary / '.git').mkdir()
    else:
        shutil.copyfile(root / '.git', temporary / '.git')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    atomic_json(temporary / 'runtime_release.json', dict(files=files, release_sha256=digest,
                                                       git_origin=commit, allows_dirty_worktree=True))
    os.rename(temporary, target)
    return target, digest


def docker_command(root, release, binding, server, gpu='0', extra_assets=(), command='run', activate=False):
    root = Path(root).resolve()
    args = ['docker', 'run', '--network', 'none', '--pid', 'host', '--gpus', 'device=' + str(gpu),
            '--user', f'{os.getuid()}:{os.getgid()}', '--shm-size', '8g',
            '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'OMP_NUM_THREADS=2', '-e', 'PYTHONUNBUFFERED=1',
            '-e', 'HF_HOME=/tmp/panda_rb_m12_hf', '-e', 'MPLCONFIGDIR=/tmp/panda_rb_m12_mpl',
            '-e', 'PANDA_RB_IMAGE_ID=' + DEFAULT_IMAGE,
            '-e', 'PANDA_RB_SERVER=' + server,
            '-v', f'{release}:{root}:ro', '-v', f'{root / "work_dir"}:{root / "work_dir"}:rw']
    for path in git_mounts(root):
        args += ['-v', f'{path}:{path}:ro']
    old_binding = root / 'work_dir/_panda_rb/20260928/B01/common/bindings.json'
    for path in asset_mounts(root, binding if Path(binding).exists() else old_binding, extra_assets):
        args += ['-v', f'{path}:{path}:ro']
    if os.environ.get('PANCRAFTER_DLPAN'):
        args += ['-e', 'PANCRAFTER_DLPAN=' + str(Path(os.environ['PANCRAFTER_DLPAN']).resolve())]
    args += ['-w', str(root), '--entrypoint', 'python', DEFAULT_IMAGE, 'tools/rb_m12_runner.py',
             command, '--server', server, '--binding', str(binding)]
    if activate:
        args.append('--activate')
    return args


def start(server, root=ROOT, binding=None, gpu='0', extra_assets=(), retry_technical=False,
          dry_run=False, activate=False):
    root = Path(root).resolve(); validate_plan(root)
    binding = Path(binding or binding_path(root)).resolve(); control = server_dir(server, root)
    if dry_run:
        return dict(server=server, training_runs=40, native_observations=80, stress_curves=96,
                    image=DEFAULT_IMAGE, binding=str(binding), changes_state=False,
                    requires='verified local B01 F1/data/LP/e-cache, pinned CUDA image, free GPU, --activate')
    if not activate:
        raise PermissionError('Starting the experiment requires --activate')
    configured = root / 'gspread/server.txt'
    if configured.is_file() and configured.read_text().strip() != server:
        raise ValueError('Requested server differs from local server identity')
    with locked(campaign_dir(root) / 'control/launch.lock'):
        image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', DEFAULT_IMAGE], text=True))[0]['Id']
        if image != DEFAULT_IMAGE:
            raise ValueError('Pinned image mismatch')
        running = subprocess.check_output(['docker', 'ps', '--filter', 'label=panda_rb.campaign=B02_M12',
            '--filter', 'label=panda_rb.server=' + server, '--format', '{{.ID}}'], text=True).strip()
        if running:
            return dict(already_running=True, server=server, container_id=running)
        state = read(control / 'status.json').get('state')
        if state == 'STOP_FOR_REVIEW':
            return dict(state=state, server=server, admitted=False)
        if state == 'TECHNICAL_FAILURE' and not retry_technical:
            raise RuntimeError('Inspect then retry same experiment with --retry-technical')
        busy = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                                       '--format=csv,noheader,nounits'], text=True).strip()
        legacy = legacy_processes(root)
        if busy or legacy:
            waiting = dict(state='WAITING_FOR_GPU_OWNER', server=server, gpu_pids=busy, controllers=legacy,
                           processes_stopped=False, training_admitted=False)
            atomic_json(control / 'status.json', waiting)
            return waiting
        if 'PANDA_RB_LOCAL_OWNER' not in (root / 'tools/_watchdog.sh').read_text():
            raise RuntimeError('Existing watchdog ownership guard is required')
        release, digest = snapshot(root)
        deployment = dict(server=server, image_id=image, release_sha256=digest, root=str(root),
                          gpu=str(gpu), binding=str(binding))
        previous = read(control / 'deployment.json')
        if previous and previous != deployment:
            raise ValueError('Frozen source/image changed; refusing silent resume')
        name = f'panda-rb-m12-{server}'
        names = subprocess.check_output(['docker', 'ps', '-a', '--format', '{{.Names}}'], text=True).splitlines()
        attempt = 1
        while name in names:
            attempt += 1; name = f'panda-rb-m12-{server}-attempt{attempt}'
        args = docker_command(root, release, binding, server, gpu, extra_assets, activate=True)
        args[2:2] = ['--detach', '--name', name, '--label', 'panda_rb.campaign=B02_M12',
                     '--label', 'panda_rb.server=' + server]
        if retry_technical:
            args.append('--retry-technical')
        atomic_json(control / 'deployment.json', deployment)
        atomic_json(root / 'work_dir/_panda_rb/local_owner.json', dict(server=server, campaign='B02_M12',
                    container=name, no_legacy_fallback=True, permanent_until_explicit_release=True))
        if legacy_processes(root):
            raise RuntimeError('An existing queue raced admission; retained hold, no process stopped')
        identifier = subprocess.check_output(args, text=True).strip()
        return dict(started=True, server=server, container=name, container_id=identifier,
                    stop_after='40 Students, 80 native observations, 96 curves; STOP_FOR_REVIEW',
                    sheets_uploaded=False)
