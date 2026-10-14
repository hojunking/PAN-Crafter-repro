"""Explicit local-server measurement/publisher cutover. Never restarts training."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .measurement import ROOT, architecture_key


def campaign_root(server, output='work_dir/directshift48'):
    from directshift48.plan import CAMPAIGN_ID, LANES
    return Path(output).resolve()/CAMPAIGN_ID/server/LANES[server]['dataset']


def configurations(root, server):
    from directshift48.publication import combined_projection, INDEX
    project = combined_projection(root, server)
    if project['conflicts']:
        raise ValueError('Scientific projection conflicts; do not profile/publish')
    configs = [json.loads(row[INDEX['Original description / notes']])['extra_parameters']['configuration']
               for row in project['rows']]
    # Include admitted current config even before first official evaluation.
    configs += [json.loads(p.read_text()) for p in sorted(Path(root).glob('runs/*/attempt_*/config.json'))]
    if not configs:
        raise ValueError('No admitted run config yet; start the campaign first')
    return list({architecture_key(c): c for c in configs}.values())


def _argv(pid):
    return [p.decode() for p in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if p]


def _starttime(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def _gpu_pids():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                     text=True, timeout=15)
    return {int(line.strip()) for line in output.splitlines() if line.strip()}


def _writable_container_path(path, mounts, *, repository_root=ROOT):
    """Resolve the effective bind mount before pausing any live process.

    The repository is intentionally read-only in standard launchers; a nested
    work_dir bind is the writable exception. Longest container-prefix matching
    mirrors Docker's mount precedence and also catches more-specific read-only
    mounts or a writable destination mapped to a different host directory.
    """
    root = Path(repository_root).resolve()
    path = Path(path).resolve()
    parents = [m for m in mounts if Path(m['Source']).resolve() == root]
    if len(parents) != 1:
        raise ValueError('Container repository bind mount mismatch')
    try:
        target = Path(parents[0]['Destination']) / path.relative_to(root)
    except ValueError as exc:
        raise ValueError('Cost storage must remain inside the mounted repository') from exc
    covering = [m for m in mounts if target == Path(m['Destination'])
                or Path(m['Destination']) in target.parents]
    selected = max(covering, key=lambda m: len(Path(m['Destination']).parts))
    if selected.get('Type') != 'bind' or selected.get('RW') is not True:
        raise ValueError('Cost storage lacks a writable bind mount; trainer untouched')
    mapped_host = (Path(selected['Source']) / target.relative_to(selected['Destination'])).resolve()
    if mapped_host != path:
        raise ValueError('Writable container cost path maps to a different host directory')
    return str(target)


def inspect_trainer(container, server):
    raw = json.loads(subprocess.check_output(['docker', 'inspect', container], text=True, timeout=15))[0]
    if not raw['State']['Running'] or raw['State'].get('Paused'):
        raise ValueError('Selected container is not running normally')
    top = subprocess.check_output(['docker', 'top', container, '-eo', 'pid,args'], text=True, timeout=15)
    candidates = []
    for line in top.splitlines()[1:]:
        pid = int(line.split()[0])
        argv = _argv(pid)
        if (any(argv[i:i+3] == ['-m', 'directshift48', 'run'] for i in range(len(argv)-2))
                and any(argv[i:i+2] == ['--server', server] for i in range(len(argv)-1))):
            candidates.append(pid)
    if len(candidates) != 1:
        raise ValueError('Require exactly one verified same-server DirectShift trainer')
    pid = candidates[0]
    if _gpu_pids() != {pid}:
        raise ValueError('Other GPU compute jobs detected; refuse nonisolated profiling')
    return {'pid': pid, 'starttime': _starttime(pid), 'argv': _argv(pid),
            'container_id': raw['Id'], 'container_name': container, 'server': server}


def resume_guard(pid, starttime, cancel, parent, seconds):
    """Independent watchdog survives parent failure; PID birth proof prevents reuse."""
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline and Path(f'/proc/{parent}').exists():
        if Path(cancel).exists():
            return
        time.sleep(.2)
    try:
        if _starttime(pid) == starttime:
            os.kill(pid, signal.SIGCONT)
    except (ProcessLookupError, FileNotFoundError):
        pass


@contextmanager
def paused_trainer(evidence, root, *, timeout=120):
    pid = evidence['pid']
    if _starttime(pid) != evidence['starttime'] or _argv(pid) != evidence['argv']:
        raise ValueError('Trainer identity changed before profiling')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cancel = root/('resume-guard-'+uuid.uuid4().hex+'.done')
    guard = subprocess.Popen([sys.executable, '-m', 'experiment_cost', '_resume-guard',
        str(pid), evidence['starttime'], str(cancel), str(os.getpid()), str(timeout+5)],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    started = time.time()
    stop_requested = False
    try:
        if guard.poll() is not None:
            raise RuntimeError('Independent resume watchdog failed to start; trainer untouched')
        stop_requested = True
        os.kill(pid, signal.SIGSTOP)
        limit = time.monotonic()+5
        while '\nState:\tT' not in Path(f'/proc/{pid}/status').read_text():
            if time.monotonic() > limit:
                raise TimeoutError('Trainer did not enter stopped state')
            time.sleep(.05)
        # Drain already dispatched asynchronous kernels before another process times CUDA.
        time.sleep(2)
        yield {**evidence, 'mode': 'EXACT_TRAINER_SIGSTOP_WITH_INDEPENDENT_RESUME_WATCHDOG',
               'pause_started_unix': started, 'max_pause_seconds': timeout,
               'other_compute_pids': [], 'training_state_retained_in_memory': True}
    finally:
        # Do not disable the independent watchdog if our resume attempt fails.
        # A dead/reused PID needs no signal, but also must not mask the original
        # measurement error or leave a guard waiting for a nonexistent process.
        resumed_or_gone = False
        try:
            if stop_requested and _starttime(pid) == evidence['starttime']:
                os.kill(pid, signal.SIGCONT)
            resumed_or_gone = True
        except (ProcessLookupError, FileNotFoundError):
            resumed_or_gone = True
        finally:
            if resumed_or_gone:
                cancel.touch()
                try:
                    guard.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # The cancellation file remains; never kill a watchdog
                    # merely because process scheduling is slow.
                    pass


def measure_needed(root, server, container, *, cost_root=None):
    from .store import find_profile
    from sheet_publish_v2.transport import atomic_json
    cost_root = Path(cost_root).resolve() if cost_root else ROOT/'work_dir/experiment_cost'/server
    configs = [c for c in configurations(root, server)
               if find_profile(c, server=server, cost_root=cost_root) is None]
    if not configs:
        return {'status': 'ALREADY_MEASURED', 'profiles': 0}
    # Container paths are only remapped through the verified repository mount.
    raw = json.loads(subprocess.check_output(['docker', 'inspect', container], text=True, timeout=15))[0]
    mounts = [m for m in raw['Mounts'] if Path(m['Source']).resolve() == ROOT.resolve()]
    if len(mounts) != 1:
        raise ValueError('Container repository bind mount mismatch')
    mount = Path(mounts[0]['Destination'])
    request = cost_root/'requests'/f'{uuid.uuid4().hex}.json'
    inside_cost_root = _writable_container_path(cost_root, raw['Mounts'])
    inside_request = _writable_container_path(request, raw['Mounts'])
    evidence = inspect_trainer(container, server)
    with paused_trainer(evidence, cost_root/'leases') as lease:
        atomic_json(request, {'configs': configs, 'server': server, 'isolation': lease,
            'deadline_unix': time.time()+110, 'cost_root': inside_cost_root})
        subprocess.run(['docker', 'exec', '-w', str(mount), container, 'timeout', '115',
            'python', '-m', 'experiment_cost', '_measure', '--request', inside_request],
            check=True, timeout=118)
    for config in configs:
        if find_profile(config, server=server, cost_root=cost_root) is None:
            raise ValueError('Measurement did not produce verified profile')
    return {'status': 'COMPLETE', 'profiles': len(configs), 'trainer_resumed': True}


def measure_request(path):
    import torch
    from panda_method.edge_data import EdgeDataSources
    from .measurement import build_architecture, profile_stage
    from .store import save_profile
    from hashlib import sha256
    request = json.loads(Path(path).read_text())
    torch.set_num_threads(4)
    for config in request['configs']:
        b = config['baseline_resolved']
        catalog = ROOT/b['source_registry']
        if sha256(catalog.read_bytes()).hexdigest() != b['source_registry_sha256']:
            raise ValueError('Source registry pin mismatch')
        overrides = None
        if b.get('lms_override_path'):
            override = ROOT/b['lms_override_path']
            if sha256(override.read_bytes()).hexdigest() != b['lms_override_sha256']:
                raise ValueError('LMS override pin mismatch')
            overrides = json.loads(override.read_text())['splits']
        sources = EdgeDataSources(catalog, root=ROOT, sensor=b['dataset'], lms_overrides=overrides)
        stage = build_architecture(config, sources, 'cuda')
        measured = profile_stage(stage, sources, device='cuda', gpu_isolation_assertion=request['isolation'],
                                 deadline_unix=request['deadline_unix'])
        envelope = save_profile(config, measured, server=request['server'], cost_root=request['cost_root'])
        print(json.dumps({'profile': envelope['path'], 'rr': measured['sheet_cost'],
                         'fr': measured['observations']['fr']['sheet_fields']}), flush=True)
        del stage, sources
        torch.cuda.empty_cache()


def start(root, server, container, *, credentials, cost_root=None):
    from .publication import enriched_projection, competing_publishers
    from sheet_publish_v2.transport import GoogleSheets, _validate_snapshot, validate_ownership
    result = measure_needed(root, server, container, cost_root=cost_root)
    project = enriched_projection(root, server, cost_root=cost_root)
    if project['conflicts']:
        raise ValueError('Cost projection conflicts; previous publisher untouched')
    # Authenticate and validate the *read-only* destination before replacing a
    # functioning old writer. Wrong credentials, changed headers or foreign rows
    # must not leave the lane without a publisher. The new writer will perform
    # its own fresh snapshot/backup/readback immediately before any actual write.
    snapshot = GoogleSheets(credentials).snapshot(project['destination'])
    _validate_snapshot(snapshot, project)
    validate_ownership(snapshot['rows'], project)
    for process in competing_publishers(server):
        pid = process['pid']
        if Path(f'/proc/{pid}/cwd').resolve() != ROOT.resolve() or _argv(pid) != process['argv']:
            raise ValueError('Publisher ownership changed; no signals sent')
        os.kill(pid, signal.SIGTERM)
        limit = time.monotonic()+10
        while Path(f'/proc/{pid}/cmdline').exists() and Path(f'/proc/{pid}/cmdline').read_bytes():
            if time.monotonic() > limit:
                raise TimeoutError('Old publisher did not stop; training untouched')
            time.sleep(.1)
    state = Path(root)/'sheet_publish_cost'
    state.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', '-m', 'experiment_cost', 'publish', '--server', server,
               '--root', str(root), '--credentials', str(Path(credentials).resolve()), '--apply', '--watch']
    if cost_root:
        command += ['--cost-root', str(Path(cost_root).resolve())]
    with (state/'publisher.log').open('a') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    return {**result, 'publisher_pid': process.pid, 'log': str(state/'publisher.log'),
            'training_restarted': False}
