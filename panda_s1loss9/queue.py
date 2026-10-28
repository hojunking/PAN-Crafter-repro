"""Initial SENSOR5 pair -> Stage1 loss grid -> sensor LR cases -> deferred WV3."""
import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from directshift48.reporting import atomic_json, utc_now
from .orchestration import ROOT, POLICY, PAUSED, initial_complete, predecessor_receipt
from . import launch

POLL_SECONDS = 15


def controller_pin():
    from panda_sensor5.queue import controller_pin as original
    files = list((ROOT / 'panda_s1loss9').rglob('*.py'))
    files += list((ROOT / 'scripts').glob('*panda_s1loss9*.sh'))
    return {**original(), **{str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(files)}}


def process_matches(argv, cwd, root, server):
    def option(key):
        if argv.count(key) == 1 and argv.index(key) + 1 < len(argv):
            return argv[argv.index(key) + 1]
        return None
    return (Path(cwd).resolve() == Path(root).resolve()
            and option('-m') == 'panda_sensor5.queue' and option('--server') == server)


def process_identity(path):
    return {'pid': int(path.name),
            'argv': (path / 'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
            'cwd': str((path / 'cwd').resolve()),
            'generation': (path / 'stat').read_text().rsplit(')', 1)[1].split()[19]}


def old_watchers(server):
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            value = process_identity(path)
            if process_matches(value['argv'], value['cwd'], ROOT, server):
                found.append(value)
        except (OSError, UnicodeError):
            continue
    return found


def stop_old_watchers(server):
    found = old_watchers(server)
    for value in found:
        path = Path('/proc') / str(value['pid'])
        try:
            if process_identity(path) != value:
                raise ValueError('Old queue process identity changed; no signal sent')
            os.kill(value['pid'], signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError):
            continue
    deadline = time.monotonic() + 20
    while old_watchers(server):
        if time.monotonic() >= deadline:
            raise ValueError('Old queue watcher did not stop; trainer untouched')
        time.sleep(.2)
    return found


def preserve_held_watchers(directory, server, held):
    """Keep watcher stop history even if acquiring its final lock times out."""
    path = Path(directory) / 'held_watchers.json'
    value = json.loads(path.read_text()) if path.exists() else {
        'schema': 'S1LOSS9_OLD_WATCHER_STOP_HISTORY_V1', 'policy': POLICY,
        'server': server, 'attempts': []}
    if value.get('policy') != POLICY or value.get('server') != server:
        raise ValueError('Old watcher stop receipt belongs to a different request')
    value['attempts'].append({'at': utc_now(), 'held_watchers': list(held)})
    atomic_json(path, value)
    # A retried enqueue must retain processes stopped by an earlier attempt.
    combined = {}
    for attempt in value['attempts']:
        for process in attempt['held_watchers']:
            combined[(process['pid'], process['generation'])] = process
    return list(combined.values())


def acquire_old_watcher_lock(lock, *, timeout=20.):
    """Process cmdline disappears just before exit releases its flock."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError('Old queue watcher lock did not release; trainer untouched')
            time.sleep(min(.2, remaining))


def inspect_optional(name):
    from panda_sensor5.queue import inspect_optional as inspect
    return inspect(name)


def sensor_state(server):
    from panda_sensor5.transition import state_of
    return state_of(ROOT, 'panda_sensor5', server)


def validate_takeover(server, chain, container, state):
    from panda_sensor5.launch import sensor_container_name
    from panda_sensor5.plan import CAMPAIGN_ID
    if (chain.get('stage') != 'WAIT_SENSOR5'
            or chain.get('sensor_container') != sensor_container_name(server)
            or chain.get('sensor_campaign_id') != CAMPAIGN_ID):
        raise ValueError('Expected current RGB/L4 v2 SENSOR5 queue; migrate v1 first or inspect advanced WV3')
    if container is None or state is None:
        raise ValueError('Current SENSOR5 container/state missing')
    if state.get('status') not in ('RUNNING', 'STOPPED_SAFE', 'FINISHED_REGISTERED_QUEUE', PAUSED):
        raise ValueError('Current SENSOR5 state requires inspection')
    active = [r for r in state.get('runs', []) if r.get('status') in ('RUNNING', 'STOPPED')]
    if len(active) > 1 or any(r.get('poisoned') or r.get('stop_checkpoint_error') for r in active):
        raise ValueError('SENSOR5 optimizer state is not unambiguous and healthy')
    return {'active_run_id': active[0]['run_id'] if active else None,
            'observed_step': active[0]['step'] if active else None}


def begin_takeover(server, directory, held_watchers):
    path = directory / 'takeover.json'
    if path.exists():
        value = json.loads(path.read_text())
        if value.get('policy') != POLICY:
            raise ValueError('Unknown existing takeover request')
        return value
    from panda_sensor5.launch import sensor_container_name
    old_directory = ROOT / 'work_dir/panda_sensor5_launchers' / server
    old_chain = json.loads((old_directory / 'chain.json').read_text())
    name = sensor_container_name(server)
    container, state = inspect_optional(name), sensor_state(server)
    current = validate_takeover(server, old_chain, container, state)
    value = {'policy': POLICY, 'at': utc_now(), 'server': server,
             'source_container': name, 'container_id': container['Id'],
             'original_chain': old_chain, 'held_watchers': held_watchers,
             'original_training_code_unchanged': True, **current}
    atomic_json(path, value)
    return value


def verify_safe_sensor_stop(server, container):
    """Authenticate saved optimizer/data/RNG checkpoint; never reset a run."""
    from panda_sensor5.transition import campaign_root, local, file_sha
    from panda_sensor5.runtime import code_pin
    state = sensor_state(server)
    cs = container['State']
    if (cs.get('Running') or cs.get('Status') != 'exited' or cs.get('OOMKilled')
            or cs.get('ExitCode') not in (0, 1)):
        raise ValueError('SENSOR5 container did not stop normally')
    terminal = state.get('status')
    if terminal == 'STOPPED_SAFE':
        if state.get('reason') != 'STOP_REQUESTED':
            raise ValueError('SENSOR5 stopped for an unrelated guard')
    elif terminal not in ('FINISHED_REGISTERED_QUEUE', PAUSED):
        raise ValueError('SENSOR5 did not preserve a safe stopping state')
    root = campaign_root(ROOT, 'panda_sensor5', server)
    admission = json.loads((root / 'admission.json').read_text())
    if admission.get('code') != code_pin():
        raise ValueError('Original SENSOR5 source pin changed; exact continuation prohibited')
    proofs = []
    for run in state.get('runs', []):
        if run.get('status') not in ('COMPLETED', 'STOPPED') or run.get('poisoned'):
            raise ValueError('SENSOR5 contains an unsafe run')
        if run['status'] == 'COMPLETED':
            continue
        directory = local(run['directory'], ROOT).resolve()
        if directory != (root / 'runs' / run['run_id'] / 'attempt_01').resolve():
            raise ValueError('Stopped SENSOR5 run has an unexpected directory')
        pointer = directory / 'models/latest_resume.json'
        artifact = json.loads(pointer.read_text())
        checkpoint = local(artifact['checkpoint_path'], ROOT)
        if (run.get('stop_checkpoint_error') or checkpoint.is_symlink()
                or checkpoint.resolve().parent != directory / 'models'
                or not checkpoint.is_file() or artifact.get('completed_update') != run.get('step')
                or artifact.get('campaign_resume_complete') is not True
                or artifact.get('checkpoint_verified') is not True
                or file_sha(checkpoint) != artifact.get('file_sha256')):
            raise ValueError('Stopped SENSOR5 run lacks an authenticated complete resume state')
        saved = json.loads((directory / 'run_state.json').read_text())
        if any(saved.get(k) != run.get(k) for k in ('run_id', 'status', 'step')):
            raise ValueError('Stopped SENSOR5 run and campaign states disagree')
        proofs.append({'run_id': run['run_id'], 'step': run['step'],
                       'checkpoint_sha256': artifact['file_sha256']})
    return {'status': terminal, 'resume_proofs': proofs, 'source_pin_preserved': True}


def wait_receipt(directory, mode):
    path = directory / (mode + '.json')
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    container = inspect_optional(receipt['container'])
    if container is None or container['Id'] != receipt['container_id']:
        raise ValueError('Expected controlled container identity changed')
    if container['State']['Running']:
        return False
    if (container['State'].get('Status') != 'exited' or container['State'].get('ExitCode') != 0
            or container['State'].get('OOMKilled')):
        raise ValueError('Controlled training did not complete normally: ' + receipt['container'])
    return True


def boundary_complete(server, chain):
    state = sensor_state(server)
    if not initial_complete(state, server):
        return False
    stopped = chain.get('safe_stop', {}).get('resume_proofs', [])
    if len(stopped) > 1:
        raise ValueError('Multiple interrupted runs require inspection')
    if not stopped:
        return True
    target = stopped[0]['run_id']
    from panda_sensor5 import plan
    configs = [plan.resolve_case(server, c, r) for g in plan.groups(server) for c, r in g['runs']]
    wanted = [c for c in configs if c['run_id'] == target]
    if len(wanted) != 1:
        raise ValueError('Interrupted boundary run is not registered')
    return any(r['run_id'] == target and r.get('status') == 'COMPLETED'
               and r.get('step') == wanted[0]['planned_horizon'] for r in state.get('runs', []))


def loss_campaign_complete(server):
    from . import plan
    from panda_sensor5.transition import file_sha, local
    root = ROOT / 'work_dir/panda_s1loss9' / plan.CAMPAIGN_ID / server / plan.dataset_for_server(server)
    path = root / 'campaign_state.json'
    if not path.exists():
        return False
    state = json.loads(path.read_text())
    if state.get('status') != 'FINISHED_REGISTERED_QUEUE':
        raise ValueError('Loss-grid container exited without full registered completion')
    configs = [plan.resolve_case(server, c, r) for g in plan.groups(server) for c, r in g['runs']]
    if len(state['runs']) != len(configs) or {r['run_id'] for r in state['runs']} != {c['run_id'] for c in configs}:
        raise ValueError('Loss-grid terminal state has missing/extra/duplicate runs')
    for cfg in configs:
        run = next(r for r in state['runs'] if r['run_id'] == cfg['run_id'])
        horizon = cfg['planned_horizon']
        if run.get('status') != 'COMPLETED' or run.get('step') != horizon:
            raise ValueError('Loss-grid registered run is incomplete')
        directory = local(run['directory'], ROOT).resolve()
        if directory != (root / 'runs' / run['run_id'] / 'attempt_01').resolve():
            raise ValueError('Loss-grid run output differs from registered root')
        evidence = directory / f'evaluation_{horizon:07d}.json'
        meta = directory / 'models' / f'step_{horizon:07d}.json'
        checkpoint = meta.with_suffix('.pt')
        if any(not p.is_file() or p.is_symlink() for p in (evidence, meta, checkpoint)):
            raise ValueError('Loss-grid final evidence is missing')
        ev, value = json.loads(evidence.read_text()), json.loads(meta.read_text())
        q, artifact = ev.get('qualification', {}), value.get('artifact', {})
        if (not all(q.get(k) for k in ('same_state_verified', 'metrics_complete', 'metrics_finite'))
                or not artifact.get('checkpoint_verified') or artifact.get('completed_update') != horizon
                or artifact.get('file_sha256') != file_sha(checkpoint)
                or artifact.get('model_state_sha256') != ev.get('state_before', {}).get('model_state_sha256')
                or value.get('config', {}).get('run_id') != cfg['run_id']):
            raise ValueError('Loss-grid final evaluation/checkpoint proof differs')
    return True


def advance(server, directory, chain, held_watchers=()):
    stage = chain.get('stage', 'TAKEOVER')
    if stage == 'TAKEOVER':
        receipt = begin_takeover(server, directory, held_watchers)
        container = inspect_optional(receipt['source_container'])
        if container is None or container['Id'] != receipt['container_id']:
            raise ValueError('Original SENSOR5 container identity changed')
        if container['State']['Running']:
            subprocess.run(['docker', 'kill', '--signal=SIGTERM', container['Id']], check=True)
            chain['stage'] = 'WAIT_SAFE_SENSOR_STOP'
        else:
            chain['safe_stop'] = verify_safe_sensor_stop(server, container)
            chain['stage'] = 'START_INITIAL_PAIR'
    elif stage == 'WAIT_SAFE_SENSOR_STOP':
        receipt = json.loads((directory / 'takeover.json').read_text())
        container = inspect_optional(receipt['source_container'])
        if container is None or container['Id'] != receipt['container_id']:
            raise ValueError('Original SENSOR5 container identity changed while stopping')
        if not container['State']['Running']:
            chain['safe_stop'] = verify_safe_sensor_stop(server, container)
            chain['stage'] = 'START_INITIAL_PAIR'
    elif stage == 'START_INITIAL_PAIR':
        if boundary_complete(server, chain):
            predecessor_receipt(ROOT, server)
            chain['stage'] = 'START_LOSS9'
        else:
            if not (directory / 'through-base.json').exists():
                stopped = chain.get('safe_stop', {}).get('resume_proofs', [])
                launch.launch(server, 'through-base', stopped[0]['run_id'] if stopped else None)
            chain['stage'] = 'WAIT_INITIAL_PAIR'
    elif stage == 'WAIT_INITIAL_PAIR':
        if wait_receipt(directory, 'through-base'):
            if sensor_state(server).get('status') != PAUSED:
                raise ValueError('SENSOR5 adapter did not pause at the requested initial-pair boundary')
            if not boundary_complete(server, chain):
                raise ValueError('SENSOR5 currently-started run did not finish before the new grid')
            predecessor_receipt(ROOT, server)
            chain['stage'] = 'START_LOSS9'
    elif stage == 'START_LOSS9':
        if not (directory / 'loss9.json').exists():
            launch.launch(server, 'loss9')
        chain['stage'] = 'WAIT_LOSS9'
    elif stage == 'WAIT_LOSS9':
        if wait_receipt(directory, 'loss9') and loss_campaign_complete(server):
            chain['stage'] = 'RESUME_SENSOR5'
    elif stage == 'RESUME_SENSOR5':
        from panda_sensor5.queue import campaign_complete
        if (sensor_state(server).get('status') == 'FINISHED_REGISTERED_QUEUE'
                and campaign_complete('panda_sensor5', server)):
            chain['stage'] = 'RESUME_TUNE30'
        else:
            if not (directory / 'resume-sensor5.json').exists():
                launch.launch(server, 'resume-sensor5')
            chain['stage'] = 'WAIT_REMAINING_SENSOR5'
    elif stage == 'WAIT_REMAINING_SENSOR5':
        from panda_sensor5.queue import campaign_complete
        if wait_receipt(directory, 'resume-sensor5'):
            if not campaign_complete('panda_sensor5', server):
                raise ValueError('Remaining SENSOR5 queue is incomplete')
            chain['stage'] = 'RESUME_TUNE30'
    elif stage in ('RESUME_TUNE30', 'RESUME_HARD36'):
        from panda_sensor5.queue import advance as advance_wv3
        advance_wv3(server, ROOT / 'work_dir/panda_sensor5_launchers' / server, chain)
    elif stage != 'COMPLETED':
        raise ValueError('Unknown priority queue stage: ' + str(stage))


def watch(server, directory, expected_pin):
    old_directory = ROOT / 'work_dir/panda_sensor5_launchers' / server
    with (directory / 'watcher.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / 'chain.json'
        chain = json.loads(path.read_text()) if path.exists() else {'stage': 'TAKEOVER', 'policy': POLICY}
        if chain.get('policy') != POLICY:
            raise ValueError('Unknown queue policy')
        if controller_pin() != expected_pin:
            raise ValueError('Priority queue source changed before takeover')
        # Enqueue validation happens before signalling even the old watcher.
        if not (directory / 'takeover.json').exists():
            from panda_sensor5.launch import sensor_container_name
            old_chain = json.loads((old_directory / 'chain.json').read_text())
            validate_takeover(server, old_chain, inspect_optional(sensor_container_name(server)), sensor_state(server))
        held = stop_old_watchers(server)
        held = preserve_held_watchers(directory, server, held)
        # Hold the old watcher lock for the full detour; accidental re-enqueue
        # cannot resume WV3 ahead of the user's new priority.
        with (old_directory / 'watcher.lock').open('a+') as old_lock:
            try:
                acquire_old_watcher_lock(old_lock)
            except Exception as exc:
                atomic_json(directory / 'status.json', {'status': 'BLOCKED',
                    'stage': chain.get('stage'), 'error': repr(exc), 'server': server,
                    'pid': os.getpid(), 'at': utc_now(), 'trainer_signalled': False})
                raise
            while True:
                try:
                    if controller_pin() != expected_pin:
                        raise ValueError('Priority queue source changed; inspect before re-enqueue')
                    advance(server, directory, chain, held)
                    chain.update(server=server, pid=os.getpid(), at=utc_now(), status=chain['stage'])
                    atomic_json(path, chain)
                    atomic_json(directory / 'status.json', chain)
                    if chain['stage'] == 'COMPLETED':
                        return 0
                except Exception as exc:
                    atomic_json(directory / 'status.json', {'status': 'BLOCKED', 'stage': chain.get('stage'),
                        'error': repr(exc), 'server': server, 'pid': os.getpid(), 'at': utc_now()})
                    raise
                time.sleep(POLL_SECONDS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', choices=('s1', 's3'), required=True)
    parser.add_argument('--enqueue', action='store_true')
    parser.add_argument('--pin', type=Path)
    args = parser.parse_args(argv)
    if (ROOT / 'gspread/server.txt').read_text().strip() != args.server:
        raise ValueError('Local server marker mismatch')
    directory = launch.directory(args.server)
    if not args.enqueue:
        if args.pin is None:
            parser.error('--enqueue or internal --pin required')
        return watch(args.server, directory, json.loads(args.pin.read_text()))
    with (directory / 'enqueue.lock').open('a+') as enqueue:
        fcntl.flock(enqueue, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (directory / 'watcher.lock').open('a+') as watcher:
            fcntl.flock(watcher, fcntl.LOCK_EX | fcntl.LOCK_NB)
        token = str(time.time_ns())
        pin = directory / ('source_' + token + '.json')
        atomic_json(pin, controller_pin())
        log = directory / ('watcher_' + token + '.log')
        with log.open('x') as output:
            child = subprocess.Popen([sys.executable, '-m', 'panda_s1loss9.queue', '--server', args.server,
                '--pin', str(pin)], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output,
                stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        receipt = {'server': args.server, 'pid': child.pid, 'log': str(log), 'pin': str(pin),
                   'at': utc_now(), 'policy': POLICY}
        atomic_json(directory / 'enqueue.json', receipt)
        print(json.dumps(receipt))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
