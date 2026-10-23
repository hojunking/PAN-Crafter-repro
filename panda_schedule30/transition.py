"""Apply the explicit queue amendment, then launch its isolated successor.

Only the named weak run may receive SIGTERM. Other active runs finish normally.
Historical campaign state, admission, checkpoints and source pins are never edited.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

PREVIOUS_CAMPAIGN = 'PANDA_S13_SCALE_HARDTUNE_30H_20261022_v1'
REQUEST_ID = '20261023_positive_aligner_schedule_extension'
WEAK_RUNNING = {'s1': ('S040BASE', 'R1'), 's3': ('S020LR150H050', 'R2')}
SKIP_PENDING = {
    's1': ('06_s050base_r1', '07_s020base_r2', '08_s010base_r2'),
    's3': ('08_s020lr150h050_r2',),
}
TERMINAL = frozenset(('FINISHED_REGISTERED_QUEUE', 'FINISHED_WITH_EXPLICIT_SKIPS',
                      'FINISHED_WITH_BUDGET_SKIPS'))
SAFE_STOP_REASONS = frozenset(('STOP_REQUESTED', 'DURABLE_30H_DEADLINE'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def pending_amendment(server, campaign, existing=None):
    """Never request a pending skip for a case that already has a run record."""
    from panda_scale30 import plan as previous
    groups = previous.groups(server)
    mapping = {g['name']: set(tuple(item) for item in g['runs']) for g in groups}
    started = {(row['case'], row['replica']) for row in campaign.get('runs', [])}
    selected = {name for name in SKIP_PENDING[server] if not (mapping[name] & started)}
    if existing is not None:
        if (set(existing) != {'request_id', 'reason', 'skip_groups'}
                or not isinstance(existing['skip_groups'], list)
                or not set(existing['skip_groups']) <= set(mapping)):
            raise ValueError('Existing pending control is not a supported explicit amendment')
        selected |= set(existing['skip_groups'])
    if any(mapping[name] & started for name in selected):
        raise ValueError('Pending control includes a started group; history was not rewritten')
    if existing is not None and selected == set(existing['skip_groups']):
        return deepcopy(existing)
    return {'request_id': REQUEST_ID,
            'reason': ('User-requested schedule extension: exclude weak pending cases; '
                       's1 baseline R2 controls move to successor; preserve all prior evidence.'),
            'skip_groups': [g['name'] for g in groups if g['name'] in selected]}


def transition_decision(server, container_state, campaign):
    """Pure admission decision. An unrelated incomplete run always blocks handoff."""
    active = [row for row in campaign.get('runs', []) if row.get('status') == 'RUNNING']
    allowed = WEAK_RUNNING[server]
    if container_state.get('Running'):
        if len(active) == 1 and (active[0].get('case'), active[0].get('replica')) == allowed:
            return 'STOP_SELECTED_RUNNING'
        return 'WAITING_PREVIOUS'
    if container_state.get('Status') != 'exited' or container_state.get('ExitCode') != 0:
        raise RuntimeError('Previous container did not exit normally; inspect it before handoff')
    stopped = [row for row in campaign.get('runs', []) if row.get('status') != 'COMPLETED']
    if campaign.get('status') == 'STOPPED_SAFE' and campaign.get('reason') in SAFE_STOP_REASONS:
        if (len(stopped) != 1 or (stopped[0].get('case'), stopped[0].get('replica')) != allowed
                or stopped[0].get('status') != 'STOPPED' or stopped[0].get('poisoned')
                or stopped[0].get('stop_checkpoint_error')):
            raise RuntimeError('Only the explicitly excluded, safely checkpointed run may remain stopped')
        return 'VERIFY_SELECTED_STOP_CHECKPOINT'
    if campaign.get('status') not in TERMINAL or stopped:
        raise RuntimeError('Previous campaign has no acceptable terminal receipt')
    return 'READY_TO_CHECK_GPU'


def local_path(root, value):
    path = Path(value)
    if str(path).startswith('/workspace/'):
        return root / path.relative_to('/workspace')
    return path if path.is_absolute() else root / path


def stopped_checkpoint_receipt(root, campaign, server):
    row = next(row for row in campaign['runs']
               if (row.get('case'), row.get('replica')) == WEAK_RUNNING[server])
    directory = local_path(root, row['directory']).resolve()
    expected_root = root / 'work_dir/panda_scale30' / PREVIOUS_CAMPAIGN / server / 'WV3/runs'
    if not directory.is_relative_to(expected_root.resolve()):
        raise ValueError('Stopped run directory is outside the expected predecessor')
    artifact = json.loads((directory / 'models/latest_resume.json').read_text())
    checkpoint = local_path(root, artifact['checkpoint_path']).resolve()
    if (not checkpoint.is_relative_to(directory / 'models') or checkpoint.is_symlink()
            or not checkpoint.name.startswith('stop_') or not checkpoint.is_file()
            or artifact.get('completed_update') != row.get('step')
            or artifact.get('campaign_resume_complete') is not True):
        raise ValueError('Stopped run lacks a complete, matching resume artifact')
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if sha != artifact.get('file_sha256'):
        raise ValueError('Stopped checkpoint SHA differs from its immutable receipt')
    return {'run_id':row['run_id'], 'step':row['step'], 'checkpoint_path':str(checkpoint),
            'file_sha256':sha, 'retained_not_resumed':True}


def inspect_owned_container(root, name, server):
    result = json.loads(subprocess.check_output(['docker', 'inspect', name], text=True, timeout=30))[0]
    if (result.get('Name') != '/' + name
            or result.get('Config', {}).get('Cmd') != ['bash', 'scripts/run_panda_scale30.sh', server]
            or not any(m.get('Destination') == '/workspace'
                       and Path(m.get('Source', '')).resolve() == root.resolve()
                       for m in result.get('Mounts', []))):
        raise ValueError('Predecessor container identity or workspace mount differs; no signal sent')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', choices=('s1', 's3'), required=True)
    parser.add_argument('--stop-selected-running', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=20.)
    args = parser.parse_args(argv)
    if not 5 <= args.poll_seconds <= 60:
        raise ValueError('Poll interval must be between 5 and 60 seconds')
    root = Path(__file__).resolve().parents[1]
    if (root / 'gspread/server.txt').read_text().strip() != args.server:
        raise ValueError('Local server marker mismatch')
    directory = root / 'work_dir/panda_schedule30_launchers' / args.server
    directory.mkdir(parents=True, exist_ok=True)
    source_root = root / 'work_dir/panda_scale30' / PREVIOUS_CAMPAIGN / args.server / 'WV3'
    source_state = source_root / 'campaign_state.json'
    previous_name = f'panda-scale30-hardonly-wv3-{args.server}-v1'
    with (directory / '.transition.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from .runtime import code_pin
        expected_pin = code_pin()
        receipt = {'request_id':REQUEST_ID, 'server':args.server, 'pid':os.getpid(),
                   'registered_at':timestamp(), 'previous_container':previous_name,
                   'source_pin':expected_pin, 'old_budget_reset':False,
                   'new_wall_time_limit':None, 'selected_stop':list(WEAK_RUNNING[args.server])}

        def record(status, **details):
            receipt.update(status=status, updated_at=timestamp(), **details)
            atomic_json(directory / 'transition_receipt.json', receipt)
            print(json.dumps({k:v for k,v in receipt.items() if k != 'source_pin'}), flush=True)

        try:
            amendment = None
            signal_sent = False
            while True:
                if code_pin() != expected_pin:
                    raise RuntimeError('Successor source changed while waiting; no launch')
                container = inspect_owned_container(root, previous_name, args.server)
                campaign = json.loads(source_state.read_text())
                if amendment is None:
                    control = source_root / 'pending_control.json'
                    existing = json.loads(control.read_text()) if control.exists() else None
                    amendment = pending_amendment(args.server, campaign, existing)
                    if amendment != existing:
                        atomic_json(directory / 'prior_pending_control.json', existing)
                        atomic_json(control, amendment)
                    receipt['pending_control'] = amendment
                decision = transition_decision(args.server, container['State'], campaign)
                if decision == 'STOP_SELECTED_RUNNING':
                    if not args.stop_selected_running:
                        record('WAITING_SELECTED_RUN_FINISH_WITHOUT_STOP_PERMISSION')
                    elif not signal_sent:
                        fresh = inspect_owned_container(root, previous_name, args.server)
                        fresh_campaign = json.loads(source_state.read_text())
                        if (fresh['State'].get('Pid') != container['State'].get('Pid')
                                or transition_decision(args.server, fresh['State'], fresh_campaign)
                                != 'STOP_SELECTED_RUNNING'):
                            continue
                        record('SENDING_SCOPED_SIGTERM', process_pid=fresh['State']['Pid'])
                        subprocess.run(['docker', 'kill', '--signal=TERM', previous_name],
                                       check=True, timeout=30, stdout=subprocess.DEVNULL)
                        signal_sent = True
                    else:
                        record('WAITING_CHECKPOINT_SAFE_STOP')
                    time.sleep(args.poll_seconds)
                    continue
                if decision == 'WAITING_PREVIOUS':
                    record(decision, previous_status=campaign.get('status'),
                           previous_runs=[{key:row.get(key) for key in ('case','replica','step','status')}
                                          for row in campaign.get('runs', [])])
                    time.sleep(args.poll_seconds)
                    continue
                if decision == 'VERIFY_SELECTED_STOP_CHECKPOINT':
                    receipt['stopped_checkpoint'] = stopped_checkpoint_receipt(root, campaign, args.server)
                processes = subprocess.check_output(
                    ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
                    text=True, stderr=subprocess.STDOUT, timeout=20)
                if processes.strip():
                    record('WAITING_GPU_IDLE', active_compute=processes.strip())
                    time.sleep(args.poll_seconds)
                    continue
                record('STARTING_SUCCESSOR_PREFLIGHT', previous_status=campaign['status'],
                       previous_stop_reason=campaign.get('reason'))
                subprocess.run(['bash', 'scripts/start_panda_schedule30_costed.sh', args.server],
                               cwd=root, check=True)
                successor = f'panda-schedule30-hardonly-wv3-{args.server}-v1'
                started = json.loads(subprocess.check_output(['docker', 'inspect', successor],
                                                            text=True, timeout=30))[0]
                if not started['State'].get('Running'):
                    raise RuntimeError('Successor did not remain running; inspect its log')
                record('LAUNCHED', new_container=successor)
                return 0
        except BaseException as exc:
            record('FAILED_INSPECT_BEFORE_RETRY', error=repr(exc))
            raise


if __name__ == '__main__':
    raise SystemExit(main())
