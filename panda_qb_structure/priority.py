"""s1: finish its current WV3 run, run QB, then continue remaining WV3.

``plan`` is read-only. ``apply`` saves the current trainer and resumes precisely
that run with the existing finish-current adapter, then starts a detached
watcher. Training configuration and previous campaign records are preserved.
Only local process ownership, resume integrity and complete-run evidence are
required; this controller does not pin an editable waiting queue's source.
"""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from directshift48.reporting import atomic_json, utc_now

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = 'work_dir/panda_qb_structure'
DIRECTORY = ROOT / OUTPUT / 'priority_s1'
CAMPAIGN_ID = 'PANDA_S13_QB_STRUCTURE_LOCALSTAGE1_20261031_v1'
POLICY = 'USER_20261031_CURRENT_WV3_THEN_QB_STRUCTURE_THEN_REMAINING_WV3'
ENQUEUE_ARGS = ['-u', '-m', 'panda_qb_structure.enqueue', '--server', 's1', '--wait']
WATCH_ARGS = ['-u', '-m', 'panda_qb_structure.priority', 'watch']
OLD_MODULES = ('panda_qb_refit.priority', 'panda_sensor_focus.queue',
               'panda_sensor_refine.queue', 'panda_s1loss9.queue',
               'panda_sensor5.queue', 'panda_hard36.queue')


def read(path):
    return json.loads(Path(path).read_text())


def process_identity(pid):
    """A PID is actionable only with its original generation, argv and cwd."""
    path = Path('/proc') / str(pid)
    try:
        stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        argv = (path / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        return {'pid': int(pid), 'generation': stat[19], 'argv': argv,
                'cwd': str((path / 'cwd').resolve()), 'uid': path.stat().st_uid}
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def module_of(identity):
    argv = identity['argv']
    return argv[argv.index('-m') + 1] if '-m' in argv and argv.index('-m') + 1 < len(argv) else None


def matching_processes(args=None, *, old=False):
    result = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        identity = process_identity(path.name)
        if identity is None:
            continue
        matches = module_of(identity) in OLD_MODULES if old else identity['argv'][1:] == args
        if matches:
            if identity['cwd'] != str(ROOT.resolve()) or identity['uid'] != os.getuid():
                raise ValueError('A matching queue controller belongs to another checkout/user')
            result.append(identity)
    if len(result) > 1:
        raise ValueError('Multiple queue owners; no process was signalled')
    return result


def inspect(name):
    from panda_sensor5.queue import inspect_optional
    return inspect_optional(name)


def authenticate_container(current, receipt, mode, target=None):
    package = receipt['package']
    module = 'panda_sensor5.wv3_continue' if package == 'panda_hard36' else 'panda_sensor5.wv3'
    expected = ['python', '-m', module, '--server', 's1', '--package', package, '--mode', mode]
    if target:
        expected += ['--target-run', target]
    if (current is None or current.get('Id') != receipt['container_id']
            or current.get('Image') != receipt['image']
            or current.get('Config', {}).get('Cmd') != expected
            or receipt.get('mode') != mode or receipt.get('target_run') != target):
        raise ValueError('WV3 container identity/command changed')
    return current['State']


def snapshot():
    """Describe an unambiguous current run without changing any file/process."""
    from panda_sensor5.transition import state_of
    if (ROOT / 'gspread/server.txt').read_text().strip() != 's1':
        raise ValueError('This optional WV3 priority handoff is s1 only')
    owner = matching_processes(old=True)
    shared = ROOT / 'work_dir/panda_sensor5_launchers/s1/wv3_tune30_resume-all.json'
    receipt = read(shared)
    current = inspect(receipt['container'])
    docker = authenticate_container(current, receipt, 'resume-all')
    state = state_of(ROOT, 'panda_tune30', 's1')
    active = [r for r in state.get('runs', []) if r['status'] in ('RUNNING', 'STOPPED')]
    if not docker.get('Running') or len(active) != 1 or active[0].get('poisoned'):
        raise ValueError('Expected one live, nonpoisoned current WV3 run')
    return {'policy': POLICY, 'campaign_id': CAMPAIGN_ID, 'server': 's1',
            'package': 'panda_tune30', 'target_run_id': active[0]['run_id'],
            'observed_step': active[0]['step'], 'original_receipt': receipt,
            'old_owner': owner[0] if owner else None, 'at': utc_now()}


def stop_owner(expected):
    if expected is None:
        if matching_processes(old=True):
            raise ValueError('A queue owner appeared after the plan')
        return
    current = process_identity(expected['pid'])
    if current is None:
        if matching_processes(old=True):
            raise ValueError('The original owner was replaced by another controller')
        return
    if current != expected or module_of(current) not in OLD_MODULES:
        raise ValueError('Queue owner identity changed; no signal sent')
    os.kill(expected['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 30
    while process_identity(expected['pid']) == expected:
        if time.monotonic() > deadline:
            raise RuntimeError('Old queue owner did not exit after SIGTERM')
        time.sleep(.2)
    if matching_processes(old=True):
        raise ValueError('A queue owner appeared during handoff')


def authenticate_safe_stop(request, current, state):
    """Intentional resume-all SIGTERM exits 1; the complete saved state is key."""
    from panda_sensor_refine.orchestration import authenticate_stop
    docker = authenticate_container(current, request['original_receipt'], 'resume-all')
    if (docker.get('Running') or docker.get('Status') != 'exited'
            or docker.get('ExitCode') not in (0, 1) or docker.get('OOMKilled')
            or state.get('status') != 'STOPPED_SAFE'
            or state.get('reason') != 'STOP_REQUESTED'):
        raise ValueError('WV3 did not make a checkpoint-safe requested stop')
    active = [r for r in state['runs'] if r['status'] in ('RUNNING', 'STOPPED', 'FAILED')]
    if (len(active) != 1 or active[0]['run_id'] != request['target_run_id']
            or active[0]['status'] != 'STOPPED'):
        raise ValueError('Current run changed or failed during handoff')
    proof = authenticate_stop(active[0], {'run_id': request['target_run_id']}, ROOT)
    return {'at': utc_now(), 'container': request['original_receipt'],
            'docker_state': docker, 'checkpoint': proof}


def safe_stop(request):
    from panda_sensor5.transition import state_of
    receipt = request['original_receipt']
    current = inspect(receipt['container'])
    docker = authenticate_container(current, receipt, 'resume-all')
    if docker.get('Running'):
        state = state_of(ROOT, request['package'], 's1')
        active = [r for r in state['runs'] if r['status'] in ('RUNNING', 'STOPPED')]
        if len(active) != 1 or active[0]['run_id'] != request['target_run_id']:
            raise ValueError('Current run advanced after planning; replan before signalling')
        subprocess.run(['docker', 'kill', '--signal=SIGTERM', receipt['container_id']], check=True)
    deadline = time.monotonic() + 300
    while True:
        current = inspect(receipt['container'])
        docker = authenticate_container(current, receipt, 'resume-all')
        if not docker.get('Running'):
            break
        if time.monotonic() > deadline:
            raise RuntimeError('Trainer has not finished its safe save; rerun apply to inspect/resume')
        print(json.dumps({'stage': 'WAIT_SAFE_SAVE', 'at': utc_now()}), flush=True)
        time.sleep(10)
    proof = authenticate_safe_stop(request, current, state_of(ROOT, request['package'], 's1'))
    atomic_json(DIRECTORY / 'safe_stop.json', proof)
    return proof


def archive_old_metadata():
    paths = [ROOT / 'work_dir/panda_sensor5_launchers/s1/wv3_tune30_resume-all.json']
    for directory in ('panda_qb_refit/priority_s1', 'panda_sensor_focus_launchers/s1'):
        paths += [ROOT / 'work_dir' / directory / name
                  for name in ('chain.json', 'status.json', 'enqueue.json')]
    for source in paths:
        if source.exists():
            target = DIRECTORY / 'history' / source.relative_to(ROOT / 'work_dir')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(source.read_bytes())


def launch_wv3(package, mode, target=None):
    """Recover the shared launch receipt if interrupted after a Docker launch."""
    from panda_sensor5 import wv3_launch
    key = 'finish_current' if mode == 'finish-current' else package + '_resume'
    path = DIRECTORY / (key + '.json')
    if path.exists():
        receipt = read(path)
        authenticate_container(inspect(receipt['container']), receipt, mode, target)
        return receipt
    intent_path = DIRECTORY / (key + '_intent.json')
    intent = read(intent_path) if intent_path.exists() else {
        'package': package, 'mode': mode, 'target': target, 'at': utc_now()}
    if (intent['package'], intent['mode'], intent['target']) != (package, mode, target):
        raise ValueError('Saved WV3 launch intent differs')
    atomic_json(intent_path, intent)
    shared = ROOT / 'work_dir/panda_sensor5_launchers/s1' / (
        'wv3_' + package.removeprefix('panda_') + '_' + mode + '.json')
    if shared.exists():
        receipt = read(shared)
        if (receipt.get('at', '') >= intent['at'] and receipt.get('mode') == mode
                and receipt.get('target_run') == target and receipt.get('package') == package):
            authenticate_container(inspect(receipt['container']), receipt, mode, target)
            atomic_json(path, receipt)
            return receipt
    # No previous shared launch was recorded. The GPU check in launch refuses
    # a running orphan; it cannot silently start a second trainer.
    receipt = wv3_launch.launch('s1', package, mode, target)
    atomic_json(path, receipt)
    return receipt


def ensure_process(args, record_name, log_name):
    matches = matching_processes(args)
    if matches:
        return matches[0]
    with (DIRECTORY / log_name).open('ab') as log:
        child = subprocess.Popen([sys.executable] + args, cwd=ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), start_new_session=True,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    identity = process_identity(child.pid)
    if identity is None:
        raise RuntimeError('Queue child exited immediately; inspect ' + log_name)
    atomic_json(DIRECTORY / record_name, {**identity, 'at': utc_now()})
    return identity


def finish_current_complete(request):
    from panda_sensor5.transition import completed_run
    receipt = read(DIRECTORY / 'finish_current.json')
    docker = authenticate_container(inspect(receipt['container']), receipt,
                                    'finish-current', request['target_run_id'])
    if docker.get('Running'):
        return None
    if docker.get('Status') != 'exited' or docker.get('ExitCode') != 0 or docker.get('OOMKilled'):
        raise ValueError('Current WV3 run did not complete normally')
    return completed_run(ROOT, request['package'], 's1', request['target_run_id'])


def qb_complete():
    path = ROOT / OUTPUT / 'launcher_s1/status.json'
    if not path.exists():
        return None
    status = read(path)
    if str(status.get('stage', '')).startswith('FAILED'):
        raise ValueError('QB launch/training failed; inspect its saved state and launcher log')
    if status.get('stage') != 'COMPLETED':
        return None
    if status.get('campaign_id') != CAMPAIGN_ID or status.get('sheet_verified') is not True:
        raise ValueError('QB completion lacks its own verified Sheet closeout')
    from .prerequisites import terminal
    return terminal(ROOT, 's1')


def advance(chain, request):
    stage = chain['stage']
    if stage == 'FINISH_CURRENT':
        proof = finish_current_complete(request)
        if proof is not None:
            atomic_json(DIRECTORY / 'boundary_complete.json', {'at': utc_now(), **proof})
            chain['stage'] = 'QB_STRUCTURE'
    elif stage == 'QB_STRUCTURE':
        proof = qb_complete()
        if proof is None:
            owner = ensure_process(ENQUEUE_ARGS, 'qb_waiter.json', 'qb_waiter.log')
            chain['qb_waiter_pid'] = owner['pid']
        else:
            atomic_json(DIRECTORY / 'qb_complete.json', proof)
            chain['stage'] = 'RESUME_TUNE30'
    elif stage in ('RESUME_TUNE30', 'RESUME_HARD36'):
        from panda_sensor5 import queue, wv3_launch
        package = 'panda_tune30' if stage == 'RESUME_TUNE30' else 'panda_hard36'
        next_stage = 'RESUME_HARD36' if package == 'panda_tune30' else 'COMPLETED'
        own_receipt = DIRECTORY / (package + '_resume.json')
        if not own_receipt.exists() and queue.campaign_complete(package, 's1'):
            chain['stage'] = next_stage
            return
        if not own_receipt.exists() and package == 'panda_hard36':
            wv3_launch.prepare_hard('s1')
        receipt = launch_wv3(package, 'resume-all')
        docker = authenticate_container(inspect(receipt['container']), receipt, 'resume-all')
        if not chain.get(package + '_publisher_ready'):
            queue.start_publisher('s1', package)
            chain[package + '_publisher_ready'] = True
        if docker.get('Running'):
            return
        if docker.get('Status') != 'exited' or docker.get('ExitCode') != 0 or docker.get('OOMKilled'):
            raise ValueError('Remaining WV3 trainer did not exit normally')
        if not queue.campaign_complete(package, 's1'):
            raise ValueError('Remaining WV3 ended without all registered runs')
        chain['stage'] = next_stage
    elif stage != 'COMPLETED':
        raise ValueError('Unknown QB priority stage: ' + str(stage))


def watch():
    request = read(DIRECTORY / 'request.json')
    if request.get('policy') != POLICY or request.get('campaign_id') != CAMPAIGN_ID:
        raise ValueError('Priority request identity differs')
    with ExitStack() as stack:
        # Process-local writer exclusion, not an experiment prerequisite.
        paths = [DIRECTORY / 'watcher.lock']
        paths += [ROOT / 'work_dir' / (p + '_launchers') / 's1/watcher.lock'
                  for p in ('panda_sensor_focus', 'panda_sensor_refine', 'panda_s1loss9',
                            'panda_sensor5', 'panda_hard36')]
        paths += [ROOT / 'work_dir/panda_qb_refit/priority_s1/watcher.lock']
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = stack.enter_context(path.open('a+'))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        chain_path = DIRECTORY / 'chain.json'
        chain = read(chain_path) if chain_path.exists() else {'stage': 'FINISH_CURRENT'}
        try:
            while True:
                advance(chain, request)
                chain.update(server='s1', campaign_id=CAMPAIGN_ID, pid=os.getpid(),
                             at=utc_now(), status=chain['stage'])
                atomic_json(chain_path, chain)
                atomic_json(DIRECTORY / 'status.json', chain)
                print(json.dumps(chain), flush=True)
                if chain['stage'] == 'COMPLETED':
                    return 0
                time.sleep(30)
        except BaseException as exc:
            atomic_json(DIRECTORY / 'status.json', {**chain, 'status': 'ERROR',
                        'error': repr(exc), 'at': utc_now()})
            raise


def apply():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    with (DIRECTORY / 'apply.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = matching_processes(WATCH_ARGS)
        if existing:
            return {'status': 'WATCHING', 'watcher': existing[0]}
        path = DIRECTORY / 'request.json'
        if path.exists():
            request = read(path)
            if request.get('policy') != POLICY or request.get('campaign_id') != CAMPAIGN_ID:
                raise ValueError('Saved priority request is for another campaign')
        else:
            request = snapshot()
            archive_old_metadata()
            atomic_json(path, request)
        stopped_owner = DIRECTORY / 'old_owner_stopped.json'
        if not stopped_owner.exists():
            stop_owner(request.get('old_owner'))
            atomic_json(stopped_owner, {'owner': request.get('old_owner'), 'at': utc_now()})
        elif matching_processes(old=True):
            raise ValueError('A legacy queue controller restarted after handoff')
        if not (DIRECTORY / 'safe_stop.json').exists():
            safe_stop(request)
        elif not (DIRECTORY / 'finish_current_intent.json').exists():
            # Revalidate an uninterrupted safe save before the first resume.
            from panda_sensor5.transition import state_of
            authenticate_safe_stop(request, inspect(request['original_receipt']['container']),
                                   state_of(ROOT, request['package'], 's1'))
        launch_wv3(request['package'], 'finish-current', request['target_run_id'])
        watcher = ensure_process(WATCH_ARGS, 'watcher.json', 'watcher.log')
        return {'status': 'FINISH_CURRENT_THEN_QB_THEN_WV3', 'watcher': watcher,
                'target_run_id': request['target_run_id']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'apply', 'watch'))
    args = parser.parse_args(argv)
    if (ROOT / 'gspread/server.txt').read_text().strip() != 's1':
        raise ValueError('Priority handoff is s1 only; s3 uses the independent enqueue command')
    if args.action == 'watch':
        return watch()
    if args.action == 'plan':
        request = DIRECTORY / 'request.json'
        value = ({'request': read(request),
                  'status': read(DIRECTORY / 'status.json') if (DIRECTORY / 'status.json').exists() else None}
                 if request.exists() else snapshot())
    else:
        value = apply()
    print(json.dumps(value, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
