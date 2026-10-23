"""Finish the current run, supersede unstarted bicubic groups, then launch L4.

Only the pending control protocol of the predecessor is written. Trainers are
never signalled. A superseded sampling watcher may be retired while waiting.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

PREVIOUS_CAMPAIGN = 'PANDA_S13_SCHEDULE_HARDONLY_20261023_v1'
REQUEST_ID = '20261023_next_pending_lanczos4_and_full_pan_difference'
TERMINAL = frozenset(('FINISHED_REGISTERED_QUEUE', 'FINISHED_WITH_EXPLICIT_SKIPS'))


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


def local_path(root, value):
    path = Path(value)
    if str(path).startswith('/workspace/'):
        return root / path.relative_to('/workspace')
    return path if path.is_absolute() else root / path


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def transition_decision(server, container_state, campaign):
    """Accept only a normally completed queue with explicit unstarted skips."""
    if server not in ('s1', 's3'):
        raise ValueError('Unknown local server')
    if container_state.get('Running'):
        return 'WAITING_CURRENT_RUN_COMPLETION'
    if container_state.get('Status') != 'exited' or container_state.get('ExitCode') != 0:
        raise RuntimeError('Predecessor did not exit normally; inspect before handoff')
    if campaign.get('status') == 'STOPPED_SAFE':
        return 'WAITING_PREVIOUS_OPERATOR_RESUME'
    if campaign.get('status') not in TERMINAL:
        raise RuntimeError('Predecessor has no normal queue completion receipt')
    from panda_schedule30 import plan as previous
    groups = previous.groups(server)
    allowed = {group['name'] for group in groups}
    amendments = campaign.get('pending_amendments', [])
    if any(not row.get('request_id') or not row.get('reason')
           or row.get('group') not in allowed
           or row.get('group') not in row.get('skip_groups', [])
           or not set(row.get('skip_groups', [])) <= allowed for row in amendments):
        raise RuntimeError('Invalid predecessor skip receipt')
    skipped = {row['group'] for row in amendments}
    expected = {tuple(run) for group in groups if group['name'] not in skipped for run in group['runs']}
    rows = campaign.get('runs', [])
    observed = {(row.get('case'), row.get('replica')) for row in rows}
    if (observed != expected or len(rows) != len(observed)
            or any(row.get('status') != 'COMPLETED' for row in rows)):
        raise RuntimeError('Declared predecessor queue is incomplete or has ambiguous history')
    return 'VERIFY_MIGRATION_RECEIPTS'


def unstarted_groups(server, campaign):
    """Any recorded run, even failed/paused, is protected from supersession."""
    from panda_schedule30 import plan as previous
    rows = campaign.get('runs', [])
    started = {(row.get('case'), row.get('replica')) for row in rows}
    registered = {tuple(run) for group in previous.groups(server) for run in group['runs']}
    if not started <= registered or len(rows) != len(started):
        raise ValueError('Unknown/duplicated predecessor run; no pending control written')
    return [group['name'] for group in previous.groups(server)
            if not any(tuple(run) in started for run in group['runs'])]


def arm_pending_control(root, server, campaign):
    """Atomically request skips only for groups with no recorded Student state.

    The predecessor reads this file once at each group boundary. A group which
    has already passed that check may finish; we never interrupt it. Repeated
    refresh removes a raced started group from the pending request, while its
    actual completion is retained by the final migration receipt.
    """
    directory = root / 'work_dir/panda_schedule30' / PREVIOUS_CAMPAIGN / server / 'WV3'
    path = directory / 'pending_control.json'
    if path.exists():
        old = json.loads(path.read_text())
        if set(old) != {'request_id', 'reason', 'skip_groups'} or not old['request_id'] or not old['reason']:
            raise ValueError('Malformed predecessor pending control')
        if old['request_id'] != REQUEST_ID:
            archive = directory / ('pending_control_before_next30_' + file_sha(path) + '.json')
            if not archive.exists():
                atomic_json(archive, old)
    value = {'request_id':REQUEST_ID,
             'reason':'User approved applying one Lanczos4 warp to the next fresh experiments; retain current and completed bicubic runs, supersede only unstarted cases into isolated NEXT30.',
             'skip_groups':unstarted_groups(server, campaign)}
    if not path.exists() or json.loads(path.read_text()) != value:
        atomic_json(path, value)
    return value


def migration_receipts(root, campaign, server):
    """Keep old results immutable and prove any completed-case deduplication."""
    from panda_schedule30 import plan as previous
    from . import plan
    rows = []
    source_root = root / 'work_dir/panda_schedule30' / PREVIOUS_CAMPAIGN / server / 'WV3/runs'
    for run in campaign['runs']:
        expected = json.loads(json.dumps(previous.resolve_case(server, run['case'], run['replica'])))
        step = expected['planned_horizon']
        if run.get('status') != 'COMPLETED' or run.get('step') != step or run.get('run_id') != expected['run_id']:
            raise ValueError('Predecessor completion lacks its registered run identity/horizon')
        directory = local_path(root, run['directory']).resolve()
        if not directory.is_relative_to(source_root.resolve()):
            raise ValueError('Predecessor result directory is outside its immutable campaign')
        checkpoint = directory / 'models' / f'step_{step:07d}.pt'
        metadata_path = checkpoint.with_suffix('.json')
        evaluation_path = directory / f'evaluation_{step:07d}.json'
        if any(not path.is_file() or path.is_symlink() for path in (checkpoint, metadata_path, evaluation_path)):
            raise ValueError('Predecessor lacks final checkpoint/evaluation evidence')
        metadata = json.loads(metadata_path.read_text())
        evaluation = json.loads(evaluation_path.read_text())
        cfg, artifact = metadata['config'], metadata['artifact']
        ident = evaluation.get('state_before', {})
        if (any(cfg.get(key) != value for key,value in expected.items())
                or artifact.get('completed_update') != step or ident.get('completed_update') != step
                or ident.get('run_id') != cfg['run_id'] or ident.get('seed') != cfg['seed']
                or evaluation.get('state_after') != ident or metadata.get('evaluation') != evaluation
                or artifact.get('identity') != ident
                or artifact.get('model_state_sha256') != ident.get('model_state_sha256')
                or artifact.get('file_sha256') != file_sha(checkpoint)
                or local_path(root, artifact['checkpoint_path']).resolve() != checkpoint.resolve()):
            raise ValueError('Predecessor provenance, same-state evaluation, or checkpoint SHA differs')
        rows.append({'case':cfg['case'], 'replica':cfg['replica'], 'run_id':cfg['run_id'],
            'seed':cfg['seed'], 'completed_update':step,
            'registered_config_sha256':previous.canonical_sha256(expected),
            'checkpoint_path':str(checkpoint), 'checkpoint_sha256':artifact['file_sha256'],
            'metadata_path':str(metadata_path), 'metadata_sha256':file_sha(metadata_path),
            'evaluation_path':str(evaluation_path), 'evaluation_sha256':file_sha(evaluation_path),
            'original_warp':'BICUBIC', 'reference_only_no_weights_imported':True,
            'reposted_to_sheet':False})
    skipped = []
    for group in plan.groups(server):
        for case, replica in group['runs']:
            cfg = plan.resolve_case(server, case, replica)
            prior_case = cfg.get('migration_predecessor_case')
            matches = [row for row in rows if row['case'] == prior_case and row['replica'] == replica]
            if matches and not cfg.get('migration_always_run', False):
                skipped.append({'group':group['name'], 'run_id':cfg['run_id'],
                    'predecessor_run_id':matches[0]['run_id'],
                    'reason':'Explicit migrated schedule case already completed before transition; preserve original result, do not restart it under L4'})
    return {'schema':'PANDA_NEXT30_MIGRATION_V1', 'predecessor_campaign':PREVIOUS_CAMPAIGN,
            'server':server, 'completed_runs':rows, 'successor_skips':skipped,
            'explicit_predecessor_skips':campaign.get('pending_amendments', []),
            'student_weights_imported':False, 'running_case_interrupted':False}


def inspect_owned_container(root, name, server):
    proc = subprocess.run(['docker', 'inspect', name], text=True, capture_output=True, timeout=30)
    if proc.returncode:
        if 'No such object' in proc.stderr or 'No such container' in proc.stderr:
            return None
        raise RuntimeError('Cannot inspect predecessor Docker container: ' + proc.stderr.strip())
    result = json.loads(proc.stdout)[0]
    if (result.get('Name') != '/' + name
            or result.get('Config', {}).get('Cmd') != ['bash', 'scripts/run_panda_schedule30.sh', server]
            or not any(m.get('Destination') == '/workspace'
                       and Path(m.get('Source', '')).resolve() == root.resolve()
                       for m in result.get('Mounts', []))):
        raise ValueError('Predecessor container identity or workspace mount differs')
    return result


def retire_sampling_watcher(root, server):
    """Retire only the exact superseded, waiting host watcher; no trainer signal."""
    directory = root / 'work_dir/panda_sampling30_launchers' / server
    pid_path, receipt_path = directory / 'pid', directory / 'transition_receipt.json'
    name = f'panda-sampling30-hardonly-wv3-{server}-v1'
    probe = subprocess.run(['docker', 'inspect', name], text=True, capture_output=True, timeout=30)
    if not probe.returncode:
        raise RuntimeError('Superseded sampling container already exists; inspect before replacement')
    if 'No such object' not in probe.stderr and 'No such container' not in probe.stderr:
        raise RuntimeError('Cannot verify absence of superseded sampling container')
    if not pid_path.exists():
        return {'status':'NO_REGISTERED_SAMPLING_WATCHER', 'trainer_signalled':False}
    pid = int(pid_path.read_text().strip())
    cmdline = Path(f'/proc/{pid}/cmdline')
    if not cmdline.exists():
        return {'status':'SAMPLING_WATCHER_ALREADY_EXITED', 'pid':pid, 'trainer_signalled':False}
    raw_command = cmdline.read_bytes()
    if not raw_command:
        return {'status':'SAMPLING_WATCHER_ALREADY_EXITED', 'pid':pid, 'trainer_signalled':False}
    command = raw_command.split(b'\0')
    if b'panda_sampling30.transition' not in command:
        raise RuntimeError('Superseded watcher PID is owned by another command; no signal sent')
    if not receipt_path.exists():
        raise RuntimeError('Superseded live watcher lacks a transition receipt; no signal sent')
    receipt = json.loads(receipt_path.read_text())
    if not str(receipt.get('status', '')).startswith('WAITING_'):
        raise RuntimeError('Superseded watcher is no longer waiting; no signal sent')
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if not cmdline.exists() or not cmdline.read_bytes():
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Superseded watcher did not terminate; no predecessor amendment')
    outcome = {'status':'RETIRED_WAITING_SAMPLING_WATCHER', 'pid':pid,
               'previous_receipt':receipt, 'trainer_signalled':False, 'retired_at':timestamp()}
    atomic_json(directory / 'superseded_by_next30.json', outcome)
    return outcome


def prepare_migration(root, server):
    """Reserve the next run boundary before the new source is frozen."""
    root = Path(root).resolve()
    if (root / 'gspread/server.txt').read_text().strip() != server:
        raise ValueError('Local server marker mismatch')
    name = f'panda-schedule30-hardonly-wv3-{server}-v1'
    container = inspect_owned_container(root, name, server)
    if container is None:
        raise RuntimeError('Predecessor has not started; no migration control written')
    source = root / 'work_dir/panda_schedule30' / PREVIOUS_CAMPAIGN / server / 'WV3/campaign_state.json'
    campaign = json.loads(source.read_text())
    transition_decision(server, container['State'], campaign)
    retired = retire_sampling_watcher(root, server)
    pending = None if campaign.get('status') in TERMINAL else arm_pending_control(root, server, campaign)
    value = {'request_id':REQUEST_ID, 'server':server, 'prepared_at':timestamp(),
             'status':'PREPARED_CURRENT_RUN_FINISHES_THEN_PENDING_GROUPS_SUPERSEDED',
             'sampling_watcher':retired, 'pending_control':pending,
             'trainer_signalled':False, 'successor_launched':False,
             'source_pin_registered':False}
    atomic_json(root / 'work_dir/panda_next30_launchers' / server / 'preparation_receipt.json', value)
    return value


def concise_receipt(value):
    """Keep console output small; the JSON receipt retains complete source pins."""
    keys = ('request_id','server','pid','status','previous_status','previous_runs',
            'new_container','error','trainer_signalled','successor_launched')
    result = {key:value[key] for key in keys if key in value}
    pending = value.get('pending_control')
    if pending is not None:
        result['pending_skip_groups'] = pending['skip_groups']
    retired = value.get('sampling_watcher', value.get('superseded_sampling_watcher'))
    if retired is not None:
        result['superseded_sampling_watcher'] = {key:retired[key] for key in
            ('status','pid','trainer_signalled') if key in retired}
    migration = value.get('migration')
    if migration is not None:
        result['predecessor_completed_runs'] = len(migration['completed_runs'])
        result['successor_migration_skips'] = migration['successor_skips']
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', choices=('s1', 's3'), required=True)
    parser.add_argument('--poll-seconds', type=float, default=20.)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(argv)
    if not 5 <= args.poll_seconds <= 60:
        raise ValueError('Poll interval must be between 5 and 60 seconds')
    root = Path(__file__).resolve().parents[1]
    if (root / 'gspread/server.txt').read_text().strip() != args.server:
        raise ValueError('Local server marker mismatch')
    if args.prepare_only:
        print(json.dumps(concise_receipt(prepare_migration(root, args.server))), flush=True)
        return 0
    directory = root / 'work_dir/panda_next30_launchers' / args.server
    directory.mkdir(parents=True, exist_ok=True)
    source_root = root / 'work_dir/panda_schedule30' / PREVIOUS_CAMPAIGN / args.server / 'WV3'
    source_state = source_root / 'campaign_state.json'
    previous_name = f'panda-schedule30-hardonly-wv3-{args.server}-v1'
    with (directory / '.transition.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from .runtime import code_pin
        expected_pin = code_pin()
        receipt = {'request_id':REQUEST_ID, 'server':args.server, 'pid':os.getpid(),
                   'registered_at':timestamp(), 'previous_container':previous_name,
                   'source_pin':expected_pin, 'new_wall_time_limit':None,
                   'current_run_stop_requested':False, 'pending_groups_superseded':True}

        def record(status, **details):
            receipt.update(status=status, updated_at=timestamp(), **details)
            atomic_json(directory / 'transition_receipt.json', receipt)
            print(json.dumps(concise_receipt(receipt)), flush=True)

        try:
            receipt['superseded_sampling_watcher'] = retire_sampling_watcher(root, args.server)
            while True:
                if code_pin() != expected_pin:
                    raise RuntimeError('Successor source changed while waiting; no launch')
                container = inspect_owned_container(root, previous_name, args.server)
                if container is None or not source_state.is_file():
                    record('WAITING_PREVIOUS_QUEUE_START')
                    time.sleep(args.poll_seconds)
                    continue
                campaign = json.loads(source_state.read_text())
                if campaign.get('status') not in TERMINAL:
                    receipt['pending_control'] = arm_pending_control(root, args.server, campaign)
                decision = transition_decision(args.server, container['State'], campaign)
                if decision.startswith('WAITING_'):
                    record(decision, previous_status=campaign.get('status'),
                           previous_runs=[{key:row.get(key) for key in ('case','replica','step','status')}
                                          for row in campaign.get('runs', [])])
                    time.sleep(args.poll_seconds)
                    continue
                if 'migration' not in receipt:
                    receipt['migration'] = migration_receipts(root, campaign, args.server)
                processes = subprocess.check_output(
                    ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
                    text=True, stderr=subprocess.STDOUT, timeout=20)
                if processes.strip():
                    record('WAITING_GPU_IDLE', active_compute=processes.strip())
                    time.sleep(args.poll_seconds)
                    continue
                record('STARTING_SUCCESSOR_PREFLIGHT', previous_status=campaign['status'])
                subprocess.run(['bash', 'scripts/start_panda_next30_costed.sh', args.server],
                               cwd=root, check=True)
                successor = f'panda-next30-hardonly-wv3-{args.server}-v1'
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
