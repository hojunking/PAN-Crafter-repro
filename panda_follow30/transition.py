"""Preserve started NEXT30 runs, skip pending low-scale runs, then launch FOLLOW30.

Only NEXT30's existing pending-control protocol is written. Its pinned source,
checkpoint files, and running trainers remain unchanged.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

PREVIOUS_CAMPAIGN = 'PANDA_S13_NEXT_HARDONLY_20261023_v1'
REQUEST_ID = '20261024_follow30_full_scale_pending_cutover'
REASON = ('User approved replacing unstarted lower-scale NEXT30 experiments with '
          'the registered scale-1 FOLLOW30 queue; preserve every started run and '
          'the full-scale corrected/DIFF controls and continuation parents.')
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


def group_policy(server):
    from panda_next30 import plan as previous
    if server not in ('s1', 's3'):
        raise ValueError('Unknown local server')
    return {group['name']: {'group':group, 'replaceable':all(
        previous.CASES[case]['scale'] < 1.
        for case,replica in group['runs'])} for group in previous.groups(server)}


def recorded_runs(server, campaign):
    policy = group_policy(server)
    registered = {tuple(run) for row in policy.values() for run in row['group']['runs']}
    rows = campaign.get('runs', [])
    started = {(row.get('case'), row.get('replica')) for row in rows}
    if not started <= registered or len(rows) != len(started):
        raise ValueError('Unknown/duplicated predecessor run; no pending control written')
    return started


def validated_skips(server, campaign):
    """Accept our explicit low-scale skips and proven historical migration skips."""
    policy = group_policy(server)
    allowed = {name for name,row in policy.items() if row['replaceable']}
    amendments = campaign.get('pending_amendments', [])
    for row in amendments:
        requested = row.get('skip_groups', [])
        if (row.get('request_id') != REQUEST_ID or row.get('reason') != REASON
                or not isinstance(requested, list) or len(requested) != len(set(requested))
                or row.get('group') not in requested or not set(requested) <= allowed):
            raise RuntimeError('Invalid or unauthorized predecessor skip receipt')
    skipped = {row['group'] for row in amendments}
    historical = campaign.get('migration_skips', [])
    registered_history = campaign.get('migration_receipt', {}).get('successor_skips', [])
    for row in historical:
        name = row.get('group')
        if name not in allowed or row not in registered_history:
            raise RuntimeError('Historical NEXT30 skip lacks its immutable migration receipt')
        from panda_next30 import plan as previous
        run_ids = {previous.resolve_case(server, case, replica)['run_id']
                   for case,replica in policy[name]['group']['runs']}
        if row.get('run_id') not in run_ids or not row.get('predecessor_run_id') or not row.get('reason'):
            raise RuntimeError('Historical NEXT30 skip identity differs')
        skipped.add(name)
    started = recorded_runs(server, campaign)
    if any(tuple(run) in started for name in skipped for run in policy[name]['group']['runs']):
        raise RuntimeError('A started predecessor group cannot be skipped')
    return skipped


def unstarted_groups(server, campaign):
    """Full-scale controls and every recorded run are protected from supersession."""
    started = recorded_runs(server, campaign)
    validated_skips(server, campaign)
    return [name for name,row in group_policy(server).items() if row['replaceable']
            and not any(tuple(run) in started for run in row['group']['runs'])]


def transition_decision(server, container_state, campaign):
    """Require normal completion of all retained cases and exact authorized skips."""
    if server not in ('s1', 's3'):
        raise ValueError('Unknown local server')
    if container_state.get('Running'):
        return 'WAITING_COMPLETE_PREVIOUS_QUEUE'
    if container_state.get('Status') != 'exited' or container_state.get('ExitCode') != 0:
        raise RuntimeError('Predecessor did not exit normally; inspect before handoff')
    if campaign.get('status') == 'STOPPED_SAFE':
        return 'WAITING_PREVIOUS_OPERATOR_RESUME'
    if campaign.get('status') not in TERMINAL:
        raise RuntimeError('Predecessor has no normal queue completion receipt')
    skipped = validated_skips(server, campaign)
    if bool(skipped) != (campaign.get('status') == 'FINISHED_WITH_EXPLICIT_SKIPS'):
        raise RuntimeError('Predecessor terminal status and explicit skip receipts disagree')
    expected = {tuple(run) for name,row in group_policy(server).items() if name not in skipped
                for run in row['group']['runs']}
    rows = campaign.get('runs', [])
    observed = {(row.get('case'), row.get('replica')) for row in rows}
    if (observed != expected or len(rows) != len(observed)
            or any(row.get('status') != 'COMPLETED' for row in rows)):
        raise RuntimeError('Declared predecessor queue is incomplete or has ambiguous history')
    return 'VERIFY_MIGRATION_RECEIPTS'


def verify_predecessor_pin(root, server, campaign=None):
    directory = root / 'work_dir/panda_next30' / PREVIOUS_CAMPAIGN / server / 'WV3'
    admission = json.loads((directory/'admission.json').read_text())
    from panda_next30 import plan as previous
    if admission.get('config') != json.loads(json.dumps(previous.default_config(server))):
        raise RuntimeError('Predecessor admitted configuration differs')
    if (campaign is not None and campaign.get('migration_skips')
            and campaign.get('migration_receipt') != admission.get('immutable_predecessor_migration')):
        raise RuntimeError('Historical migration differs from immutable predecessor admission')
    pin = admission.get('code', {})
    if not pin or any(not isinstance(name,str) or Path(name).is_absolute()
                      or '..' in Path(name).parts for name in pin):
        raise RuntimeError('Predecessor admitted source pin is invalid')
    for name,digest in pin.items():
        path = root/name
        if not path.is_file() or file_sha(path) != digest:
            raise RuntimeError('Predecessor source changed: ' + name)
    return {'source_files_verified':len(pin), 'admission_sha256':file_sha(directory/'admission.json')}


def arm_pending_control(root, server, campaign):
    """Refresh the existing protocol; a group racing past its boundary may finish.

    NEXT30 rejects attempts to skip started groups. Re-read durable state before
    writing, and remove any newly started group on every refresh. No signals or
    predecessor source/config/checkpoint edits are used.
    """
    directory = root / 'work_dir/panda_next30' / PREVIOUS_CAMPAIGN / server / 'WV3'
    state = directory/'campaign_state.json'
    if state.is_file():
        campaign = json.loads(state.read_text())
    if campaign.get('status') in TERMINAL:
        return None
    value = {'request_id':REQUEST_ID, 'reason':REASON,
             'skip_groups':unstarted_groups(server, campaign)}
    path = directory/'pending_control.json'
    if path.exists():
        old = json.loads(path.read_text())
        if set(old) != {'request_id','reason','skip_groups'} or old.get('request_id') != REQUEST_ID:
            raise ValueError('Another predecessor pending request exists; no overwrite')
        if old == value:
            return value
    atomic_json(path,value)
    return value


def _checkpoint_receipt(root, expected, directory, step):
    from panda_next30 import plan as previous
    checkpoint = directory / 'models' / f'step_{step:07d}.pt'
    metadata_path = checkpoint.with_suffix('.json')
    evaluation_path = directory / f'evaluation_{step:07d}.json'
    if any(not path.is_file() or path.is_symlink() for path in (checkpoint, metadata_path, evaluation_path)):
        raise ValueError('Predecessor lacks required checkpoint/evaluation evidence')
    metadata = json.loads(metadata_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())
    cfg, artifact = metadata['config'], metadata['artifact']
    ident = evaluation.get('state_before', {})
    if (any(cfg.get(key) != json.loads(json.dumps(value)) for key,value in expected.items())
            or artifact.get('completed_update') != step or ident.get('completed_update') != step
            or ident.get('run_id') != cfg['run_id'] or ident.get('seed') != cfg['seed']
            or evaluation.get('state_after') != ident or metadata.get('evaluation') != evaluation
            or artifact.get('identity') != ident
            or artifact.get('model_state_sha256') != ident.get('model_state_sha256')
            or artifact.get('file_sha256') != file_sha(checkpoint)
            or local_path(root, artifact['checkpoint_path']).resolve() != checkpoint.resolve()):
        raise ValueError('Predecessor provenance, same-state evaluation, or checkpoint SHA differs')
    return {'case':cfg['case'], 'replica':cfg['replica'], 'run_id':cfg['run_id'],
        'seed':cfg['seed'], 'completed_update':step,
        'registered_config_sha256':previous.canonical_sha256(expected),
        'checkpoint_path':str(checkpoint), 'checkpoint_sha256':artifact['file_sha256'],
        'metadata_path':str(metadata_path), 'metadata_sha256':file_sha(metadata_path),
        'evaluation_path':str(evaluation_path), 'evaluation_sha256':file_sha(evaluation_path),
        'original_warp':'LANCZOS4', 'reference_only_no_weights_imported':True,
        'reposted_to_sheet':False}


def migration_receipts(root, campaign, server):
    """Validate completed results and same-server 15K DIFF parents; do not copy them."""
    from panda_next30 import plan as previous
    from . import plan
    transition_decision(server, {'Running':False,'Status':'exited','ExitCode':0}, campaign)
    verify_predecessor_pin(root,server,campaign)
    rows, directories = [], {}
    source_root = root / 'work_dir/panda_next30' / PREVIOUS_CAMPAIGN / server / 'WV3/runs'
    for run in campaign['runs']:
        expected = json.loads(json.dumps(previous.resolve_case(server, run['case'], run['replica'])))
        step = expected['planned_horizon']
        if run.get('status') != 'COMPLETED' or run.get('step') != step or run.get('run_id') != expected['run_id']:
            raise ValueError('Predecessor completion lacks its registered run identity/horizon')
        directory = local_path(root, run['directory']).resolve()
        canonical = source_root/expected['run_id']/f"attempt_{expected['attempt']:02d}"
        if directory != canonical.resolve():
            raise ValueError('Predecessor result directory differs from its registered run/attempt')
        rows.append(_checkpoint_receipt(root, expected, directory, step))
        directories[expected['run_id']] = directory
    parents = []
    for group in plan.groups(server):
        for case,replica in group['runs']:
            cfg = plan.resolve_case(server, case, replica)
            if cfg.get('source_package') != 'panda_next30':
                continue
            source = previous.resolve_case(server,cfg['source_case'],replica,
                                           attempt=cfg['source_attempt'])
            directory = directories.get(source['run_id'])
            path = (root/cfg['source_checkpoint_relative_path']).resolve()
            if (directory is None or path != directory/'models'/f"step_{cfg['inherited_updates']:07d}.pt"):
                raise ValueError('Required same-server DIFF continuation parent is unavailable')
            receipt = _checkpoint_receipt(root,source,directory,cfg['inherited_updates'])
            receipt['successor_run_id'] = cfg['run_id']
            parents.append(receipt)
    return {'schema':'PANDA_FOLLOW30_PREDECESSOR_RECEIPT_V1',
            'predecessor_campaign':PREVIOUS_CAMPAIGN, 'server':server,
            'completed_runs':rows, 'successor_skips':[],
            'explicit_predecessor_skips':campaign.get('pending_amendments',[]),
            'historical_predecessor_migration_skips':campaign.get('migration_skips',[]),
            'required_continuation_parents':parents,
            'student_weights_imported':False, 'running_case_interrupted':False,
            'source_mutations':[]}


def inspect_owned_container(root, name, server):
    proc = subprocess.run(['docker', 'inspect', name], text=True, capture_output=True, timeout=30)
    if proc.returncode:
        if 'No such object' in proc.stderr or 'No such container' in proc.stderr:
            return None
        raise RuntimeError('Cannot inspect predecessor Docker container: ' + proc.stderr.strip())
    result = json.loads(proc.stdout)[0]
    if (result.get('Name') != '/' + name
            or result.get('Config', {}).get('Cmd') != ['bash', 'scripts/run_panda_next30.sh', server]
            or not any(m.get('Destination') == '/workspace'
                       and Path(m.get('Source', '')).resolve() == root.resolve()
                       for m in result.get('Mounts', []))):
        raise ValueError('Predecessor container identity or workspace mount differs')
    return result


def inspect_transition(root, server, *, apply_pending=False):
    path = root / 'work_dir/panda_next30' / PREVIOUS_CAMPAIGN / server / 'WV3/campaign_state.json'
    name = f'panda-next30-hardonly-wv3-{server}-v1'
    container = inspect_owned_container(root, name, server)
    if not path.is_file() or container is None:
        return {'status':'WAITING_PREVIOUS_QUEUE_START','source_mutations':[]}
    campaign = json.loads(path.read_text())
    decision = transition_decision(server, container['State'], campaign)
    proof = verify_predecessor_pin(root, server, campaign)
    proposed = unstarted_groups(server,campaign)
    pending = None
    if apply_pending and campaign.get('status') not in TERMINAL:
        pending = arm_pending_control(root,server,campaign)
    result = {'status':decision,'previous_status':campaign.get('status'),
              'previous_runs':[{k:r.get(k) for k in ('case','replica','step','status')}
                               for r in campaign.get('runs',[])],
              'proposed_skip_groups':proposed, 'pending_control':pending,
              'predecessor_source_pin':proof,
              'source_mutations':(['pending_control.json'] if pending is not None else [])}
    if decision == 'VERIFY_MIGRATION_RECEIPTS':
        result['migration'] = migration_receipts(root, campaign, server)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', choices=('s1','s3'), required=True)
    parser.add_argument('--poll-seconds', type=float, default=20.)
    parser.add_argument('--prepare-only', '--dry-run', action='store_true')
    parser.add_argument('--apply-pending', action='store_true',
                        help='Write authorized low-scale pending skips once, without launching')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args(argv)
    if args.prepare_only and args.apply_pending:
        parser.error('--prepare-only is read-only and cannot be combined with --apply-pending')
    if not 5 <= args.poll_seconds <= 60:
        raise ValueError('Poll interval must be between 5 and 60 seconds')
    root = Path(__file__).resolve().parents[1]
    if (root / 'gspread/server.txt').read_text().strip() != args.server:
        raise ValueError('Local server marker mismatch')
    if args.prepare_only or args.apply_pending:
        receipt = inspect_transition(root,args.server,apply_pending=args.apply_pending)
        print(json.dumps(receipt),flush=True)
        if args.require_complete and receipt['status'] != 'VERIFY_MIGRATION_RECEIPTS':
            raise RuntimeError('Retained NEXT30 queue is not complete; FOLLOW30 must wait')
        return 0
    directory = root / 'work_dir/panda_follow30_launchers' / args.server
    directory.mkdir(parents=True,exist_ok=True)
    with (directory / '.transition.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from .runtime import code_pin
        expected_pin = code_pin()
        while True:
            if code_pin() != expected_pin:
                raise RuntimeError('Successor source changed while waiting; no launch')
            receipt = inspect_transition(root,args.server,apply_pending=True)
            receipt.update(request_id=REQUEST_ID,server=args.server,pid=os.getpid(),
                           updated_at=timestamp(),trainer_signalled=False)
            atomic_json(directory/'transition_receipt.json',receipt)
            print(json.dumps({k:v for k,v in receipt.items() if k!='migration'}),flush=True)
            if receipt['status'].startswith('WAITING_'):
                time.sleep(args.poll_seconds)
                continue
            processes = subprocess.check_output(
                ['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],
                text=True,stderr=subprocess.STDOUT,timeout=20)
            if processes.strip():
                time.sleep(args.poll_seconds)
                continue
            subprocess.run(['bash','scripts/start_panda_follow30_costed.sh',args.server],cwd=root,check=True)
            receipt.update(status='LAUNCHED',launched_at=timestamp())
            atomic_json(directory/'transition_receipt.json',receipt)
            return 0


if __name__ == '__main__':
    raise SystemExit(main())
