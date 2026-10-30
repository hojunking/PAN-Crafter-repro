"""s1: finish the saved current WV3 run -> QB refit -> remaining TUNE30/HARD36.

The checkpoint-safe handoff is reviewed explicitly before starting this watcher.
It requires request.json, safe_stop.json and finish_current.json; it never stops
a trainer, changes a training registry, or restarts a failed container.
"""
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from directshift48.reporting import atomic_json, utc_now
from . import registry as G
from .prerequisites import read, sha, priority_evidence, terminal

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / G.OUTPUT / 'priority_s1'


def validate_finish_container(request, receipt, current):
    expected = ['python', '-m', 'panda_sensor5.wv3', '--server', 's1',
                '--package', request['package'], '--mode', 'finish-current',
                '--target-run', request['target_run_id']]
    if (current is None or current['Id'] != receipt['container_id']
            or current['Image'] != request['original_receipt']['image']
            or current['Image'] != receipt['image']
            or receipt['target_run'] != request['target_run_id']
            or current['Config']['Cmd'] != expected):
        raise ValueError('Current WV3 container/target was replaced')
    state = current['State']
    if state['Running']:
        return False
    if state.get('OOMKilled') or state.get('Status') != 'exited' or state.get('ExitCode') != 0:
        raise ValueError('Current WV3 run failed instead of completing normally')
    return True


def ensure_qb_waiter():
    """Adopt only the unique same-checkout host watcher; its own flock pins it."""
    args = ['-u', '-m', 'panda_qb_refit.enqueue', '--wait']
    matches = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = path.read_bytes().decode().strip('\0').split('\0')
            if argv[1:] == args:
                if (path.parent / 'cwd').resolve() != ROOT:
                    raise ValueError('Foreign QB waiter checkout')
                matches.append(int(path.parent.name))
        except (OSError, UnicodeError):
            continue
    if len(matches) > 1:
        raise ValueError('Multiple QB waiters')
    if matches:
        return matches[0]
    receipt = DIRECTORY / 'qb_waiter.json'
    if receipt.exists():
        raise ValueError('QB waiter exited before verified completion; inspect instead of restarting')
    control = ROOT / G.OUTPUT / 'launcher_s1'
    with (control / 'controller.log').open('ab') as log:
        child = subprocess.Popen([sys.executable] + args, cwd=ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), start_new_session=True,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    atomic_json(receipt, {'pid': child.pid, 'argv': [sys.executable] + args, 'at': utc_now()})
    return child.pid


def advance(chain, request):
    from panda_sensor5 import queue as old_queue
    from panda_sensor5.transition import campaign_root, completed_run
    stage = chain['stage']
    if stage == 'FINISH_CURRENT':
        receipt = read(DIRECTORY / 'finish_current.json')
        if not validate_finish_container(request, receipt, old_queue.inspect_optional(receipt['container'])):
            return
        proof = completed_run(ROOT, request['package'], 's1', request['target_run_id'])
        atomic_json(DIRECTORY / 'boundary_complete.json', {
            'request_sha256': sha(DIRECTORY / 'request.json'), 'completed_run': proof,
            'container_id': receipt['container_id'], 'container_exit_code': 0,
            'state_sha256': sha(campaign_root(ROOT, request['package'], 's1') / 'campaign_state.json'),
            'at': utc_now()})
        priority_evidence(ROOT)
        chain['stage'] = 'QB_REFIT'
    elif stage == 'QB_REFIT':
        status_path = ROOT / G.OUTPUT / 'launcher_s1/status.json'
        status = read(status_path) if status_path.exists() else {}
        if status.get('stage') == 'COMPLETED':
            if status.get('campaign_id') != G.CAMPAIGN_ID or not status.get('sheet_verified'):
                raise ValueError('QB completion has no verified Sheet closeout')
            proof = terminal(ROOT)
            atomic_json(DIRECTORY / 'qb_complete.json', proof)
            chain.update(stage='RESUME_TUNE30', qb_completed=proof['completed_count'],
                         qb_early_stopped=proof['early_stopped_count'])
        elif status.get('stage') == 'FAILED_TRAINING':
            raise ValueError('QB trainer failed; remaining WV3 remains deferred')
        else:
            priority_evidence(ROOT)
            chain['qb_waiter_pid'] = ensure_qb_waiter()
    elif stage in ('RESUME_TUNE30', 'RESUME_HARD36'):
        old_queue.advance('s1', ROOT / 'work_dir/panda_sensor5_launchers/s1', chain)
    elif stage != 'COMPLETED':
        raise ValueError('Unknown priority stage: ' + str(stage))


def main():
    from .enqueue import source_pin, verify_pin
    from panda_sensor_focus.queue import controller_pin
    if (ROOT / 'gspread/server.txt').read_text().strip() != 's1':
        raise ValueError('QB priority controller is s1 only')
    request = read(DIRECTORY / 'request.json')
    safe = read(DIRECTORY / 'safe_stop.json')
    if (request['policy'] != 'USER_20261030_FINISH_CURRENT_THEN_QB_THEN_WV3'
            or request['server'] != 's1' or request['package'] != 'panda_tune30'
            or safe['checkpoint']['run_id'] != request['target_run_id']):
        raise ValueError('Missing/mismatched reviewed current-run handoff')
    with ExitStack() as stack:
        paths = [DIRECTORY / 'watcher.lock'] + [ROOT / 'work_dir' / (p + '_launchers') / 's1/watcher.lock'
            for p in ('panda_sensor_focus', 'panda_sensor_refine', 'panda_s1loss9', 'panda_sensor5', 'panda_hard36')]
        for path in paths:
            lock = stack.enter_context(path.open('a+'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pin_path = DIRECTORY / 'controller_pin.json'
        pin = read(pin_path) if pin_path.exists() else source_pin()
        verify_pin(pin)
        if not pin_path.exists():
            atomic_json(pin_path, pin)
        chain_path = DIRECTORY / 'chain.json'
        chain = read(chain_path) if chain_path.exists() else {'stage': 'FINISH_CURRENT'}
        try:
            while True:
                verify_pin(pin)
                if controller_pin() != request['original_controller_pin']:
                    raise ValueError('Original focus/WV3 code changed')
                advance(chain, request)
                chain.update(server='s1', pid=os.getpid(), at=utc_now(), status=chain['stage'],
                             target_run_id=request['target_run_id'])
                atomic_json(chain_path, chain)
                atomic_json(DIRECTORY / 'status.json', chain)
                old = ROOT / 'work_dir/panda_sensor_focus_launchers/s1'
                mirrored = read(old / 'chain.json')
                resumed = chain['stage'] in ('RESUME_TUNE30', 'RESUME_HARD36', 'COMPLETED')
                mirrored.update(chain if resumed else {'stage': 'HELD_FOR_QB_PRIORITY',
                    'status': 'HELD_FOR_QB_PRIORITY', 'priority_stage': chain['stage'], 'at': utc_now()})
                atomic_json(old / 'chain.json', mirrored)
                atomic_json(old / 'status.json', mirrored)
                print(json.dumps(chain), flush=True)
                if chain['stage'] == 'COMPLETED':
                    return 0
                time.sleep(30)
        except BaseException as exc:
            atomic_json(DIRECTORY / 'status.json', {**chain, 'status': 'BLOCKED', 'error': repr(exc), 'at': utc_now()})
            raise


if __name__ == '__main__':
    raise SystemExit(main())
