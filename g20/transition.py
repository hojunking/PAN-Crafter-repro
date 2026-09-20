"""Explicit-start legacy drain without editing source or terminating trainers.

The frozen QG40 controller validates its hold before each block and every train
child. A takeover is therefore allowed while the LAST already-admitted case is
still training, or with its runner lock free. Earlier cases must drain first.
GPU and runner locks must subsequently be idle before G20 preparation starts.
This module is inert until an explicitly registered G20 start calls it.
"""
import datetime as dt
from contextlib import ExitStack
import os
from pathlib import Path
import shlex
import subprocess
import time

from g20.common import atomic_json, camp, locked, read, utcnow

DONE = {'OFFICIAL_EVAL_COMPLETE', 'UPLOAD_VERIFIED', 'ARCHIVED'}


def processes():
    rows = []
    result = subprocess.run(['ps', '-eo', 'pid=,args='], capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        raw = line.split(None, 1)
        if len(raw) != 2 or int(raw[0]) == os.getpid():
            continue
        try:
            args = shlex.split(raw[1])
        except ValueError:
            continue
        if '-c' in args[:3]:
            continue
        if any(Path(arg).name in {'qg40_runner.py', 'fh20r1_runner.py'} for arg in args[:3]):
            pid = int(raw[0])
            # A stale training_status from an earlier paused process cannot
            # authorize takeover of a new child that has not passed its hold.
            try:
                ticks = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
                boot = next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines()
                            if line.startswith('btime '))
                started = boot + ticks / os.sysconf('SC_CLK_TCK')
            except (OSError, ValueError, StopIteration):
                started = None
            rows.append(dict(pid=pid, args=args, started_epoch=started))
    return rows


def terminal_training(root, server, snapshot, running):
    admitted = [block for block in snapshot.get('blocks', {}).values() if block.get('status') == 'ADMITTED']
    if len(admitted) != 1:
        return None
    order = admitted[0].get('run_ids', [])
    if not order or any(snapshot.get('runs', {}).get(run, {}).get('status') not in DONE for run in order[:-1]):
        return None
    final = order[-1]
    status_path = Path(root) / 'work_dir' / final / 'meta/training_status.json'
    status = read(status_path)
    if (snapshot.get('runs', {}).get(final, {}).get('status') != 'RUNNING'
            or status.get('training_complete') or not 0 <= status.get('actual_updates', -1) < 48000):
        return None
    children = [p for p in running if 'train' in p['args'] and p.get('started_epoch') is not None and
                status_path.is_file() and status_path.stat().st_mtime > p['started_epoch'] + 2 and
                any(final in arg and arg.endswith('.yaml') for arg in p['args'])]
    return dict(last_admitted_run=final, trainer_pid=children[0]['pid'], snapshot=status) if len(children) == 1 else None


def wait_and_claim(root, server, window, poll_seconds=15):
    root = Path(root)
    folder = camp(root, server)
    requested = read(folder / 'requested_hold.json')
    if not requested:
        raise ValueError('Explicit G20 registration required before transition')
    hold_path = root / 'work_dir/_eval_phase/hold.json'
    original = read(folder / 'registration.json').get('original_wv3_hold', {})
    while dt.datetime.now(dt.timezone.utc) < window.train_finish_utc:
        current = read(hold_path)
        if current == requested:
            return True
        if current != original:
            raise ValueError('Legacy hold changed during G20 handoff; foreign owner preserved')
        legacy = root / 'work_dir/_qg40' / server
        snapshot, evidence = read(legacy / 'status.json'), None
        try:
            # Keep the old lock while publishing hold: prevents an idle recovery
            # process from admitting a legacy block in the handoff interval.
            with ExitStack() as stack:
                stack.enter_context(locked(legacy / '.runner.lock'))
                stack.enter_context(locked(root / 'work_dir/_fh20r1' / server / '.runner.lock'))
                pending_admitted = any(block.get('status') == 'ADMITTED' and
                    any(snapshot.get('runs', {}).get(run, {}).get('status') not in DONE for run in block.get('run_ids', []))
                    for block in snapshot.get('blocks', {}).values())
                if not pending_admitted and not any('train' in p['args'] for p in processes()):
                    atomic_json(hold_path, requested)
                    atomic_json(folder / 'transition.json', dict(status='LEGACY_ADMISSIONS_HELD',
                        at_utc=utcnow(), boundary='old_runner_idle', old_state=snapshot))
                    return True
        except BlockingIOError:
            # This fast path is only for the QG40 owner, never an arbitrary
            # occupied legacy lock or a different protocol's unfinished block.
            if original.get('protocol_id') == 'QG40_FUTURE_ADMISSIONS_HOLD_v1':
                evidence = terminal_training(root, server, snapshot, processes())
        if evidence:
            # Revalidate the same live child and block immediately before claim.
            latest = terminal_training(root, server, read(legacy / 'status.json'), processes())
            if latest and latest['trainer_pid'] == evidence['trainer_pid'] and latest['last_admitted_run'] == evidence['last_admitted_run']:
                if read(hold_path) != original:
                    raise ValueError('Hold owner changed during the final handoff check')
                atomic_json(hold_path, requested)
                atomic_json(folder / 'transition.json', dict(status='DRAINING_FINAL_LEGACY_CASE',
                    at_utc=utcnow(), evidence=latest, old_state=snapshot,
                    policy='No process signals. Existing final train/calibration/evaluation finishes; next train cannot authorize.'))
                return True
        atomic_json(folder / 'transition.json', dict(status='WAIT_ADMITTED_LEGACY_BLOCK',
            at_utc=utcnow(), old_state=snapshot, deadline_utc=window.deadline_utc.isoformat()))
        time.sleep(poll_seconds)
    return False


def legacy_busy(root, server):
    """Hold ownership alone is not proof the old full block finished."""
    handoff = read(camp(root, server) / 'transition.json')
    if handoff.get('status') == 'DRAINING_FINAL_LEGACY_CASE':
        final = handoff.get('evidence', {}).get('last_admitted_run')
        if not final:
            return True
        legacy = read(Path(root) / 'work_dir/_qg40' / server / 'status.json')
        row = legacy.get('runs', {}).get(final, {})
        training = read(Path(root) / 'work_dir' / final / 'meta/training_status.json')
        postrun = read(Path(root) / 'work_dir' / final / 'official/postrun_status.json')
        completed = (training.get('training_complete') is True and training.get('actual_updates') == 50000
                     and postrun.get('official_complete') is True)
        if row.get('status') not in DONE and not completed:
            # A child disappearing or pausing is not successful drain. Retain
            # its checkpoint and wait; never start G20 by abandoning the case.
            return True
    for campaign in ('_qg40', '_fh20r1'):
        try:
            with locked(Path(root) / 'work_dir' / campaign / server / '.runner.lock'):
                pass
        except BlockingIOError:
            return True
    return any('train' in p['args'] or 'postrun' in p['args'] or 'calibrate' in p['args'] for p in processes())
