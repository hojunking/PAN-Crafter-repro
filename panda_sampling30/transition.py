"""Wait for the existing schedule queue to finish, then launch sampling cases.

This module never signals a trainer or writes predecessor state or controls.
Completed same-server S020C30 R1/R2 baselines are required and retained in place.
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

PREVIOUS_CAMPAIGN = 'PANDA_S13_SCHEDULE_HARDONLY_20261023_v1'
REQUEST_ID = '20261023_sampling_ablation_after_complete_schedule_queue'
TERMINAL = frozenset(('FINISHED_REGISTERED_QUEUE', 'FINISHED_WITH_EXPLICIT_SKIPS'))
DONOR_MATCH_KEYS = (
    'sensor', 'source_split', 'sample_ids_sha256', 'sample_count', 'source_manifest_sha256',
    'catalog_sha256', 'source_h5_sha256', 'aligner_state_sha256', 'teacher_state_sha256',
    'donor_checkpoint_sha256', 'donor_run_id', 'donor_step', 'source_campaign_id',
    'mature30_selector_sha256', 'mature30_validation_inventory_sha256',
)


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
    """A running or cleanly paused predecessor is never stopped by this watcher."""
    if server not in ('s1', 's3'):
        raise ValueError('Unknown local server')
    if container_state.get('Running'):
        return 'WAITING_PREVIOUS_QUEUE_COMPLETION'
    if container_state.get('Status') != 'exited' or container_state.get('ExitCode') != 0:
        raise RuntimeError('Predecessor did not exit normally; inspect before handoff')
    if campaign.get('status') == 'STOPPED_SAFE':
        return 'WAITING_PREVIOUS_OPERATOR_RESUME'
    if campaign.get('status') not in TERMINAL:
        raise RuntimeError('Predecessor has no normal queue completion receipt')
    from panda_schedule30 import plan as previous
    groups = previous.groups(server)
    skipped = {row['group'] for row in campaign.get('pending_amendments', [])
               if row.get('request_id') and row.get('reason') and row.get('group') in row.get('skip_groups', [])}
    expected = {tuple(run) for group in groups if group['name'] not in skipped for run in group['runs']}
    rows = campaign.get('runs', [])
    observed = {(row.get('case'), row.get('replica')) for row in rows}
    if (observed != expected or len(rows) != len(observed)
            or any(row.get('status') != 'COMPLETED' for row in rows)):
        raise RuntimeError('Declared predecessor queue is incomplete or has ambiguous history')
    if not {('S020C30', 'R1'), ('S020C30', 'R2')} <= observed:
        raise RuntimeError('Both completed same-server S020C30 controls are required')
    return 'VERIFY_BASELINE_RECEIPTS'


def baseline_receipts(root, campaign, server):
    """Bind fresh comparison controls without copying weights or reposting rows."""
    from panda_schedule30 import plan as previous
    from .donor import source_receipt
    for replica in ('R1', 'R2'):
        wanted = previous.resolve_case(server, 'S020C30', replica)['run_id']
        matches = [row for row in campaign['runs'] if row.get('run_id') == wanted]
        if len(matches) != 1 or matches[0].get('status') != 'COMPLETED' or matches[0].get('step') != 30000:
            raise ValueError('Matching 30K baseline did not complete')
    selector_path, selector = source_receipt(server, root)
    selected = selector['selected']
    donor_path = local_path(root, selected['checkpoint_path']).resolve()
    if (not donor_path.is_file() or donor_path.is_symlink()
            or file_sha(donor_path) != selected['file_sha256']):
        raise ValueError('Current same-server donor checkpoint differs from its selection receipt')
    result = []
    source_root = root / 'work_dir/panda_schedule30' / PREVIOUS_CAMPAIGN / server / 'WV3/runs'
    for replica in ('R1', 'R2'):
        expected = json.loads(json.dumps(previous.resolve_case(server, 'S020C30', replica)))
        matches = [row for row in campaign['runs'] if row.get('run_id') == expected['run_id']]
        if len(matches) != 1 or matches[0].get('status') != 'COMPLETED' or matches[0].get('step') != 30000:
            raise ValueError('Matching 30K baseline did not complete')
        directory = local_path(root, matches[0]['directory']).resolve()
        if not directory.is_relative_to(source_root.resolve()):
            raise ValueError('Baseline directory is outside its immutable predecessor')
        checkpoint = directory / 'models/step_0030000.pt'
        metadata_path = checkpoint.with_suffix('.json')
        evaluation_path = directory / 'evaluation_0030000.json'
        if any(not path.is_file() or path.is_symlink() for path in (checkpoint, metadata_path, evaluation_path)):
            raise ValueError('Baseline lacks immutable final checkpoint/evaluation evidence')
        metadata = json.loads(metadata_path.read_text())
        artifact, cfg = metadata['artifact'], metadata['config']
        evaluation = json.loads(evaluation_path.read_text())
        identity = evaluation.get('state_before', {})
        if (any(cfg.get(key) != value for key,value in expected.items())
                or cfg.get('planned_horizon') != 30000 or cfg.get('inherited_updates', 0) != 0
                or artifact.get('completed_update') != 30000
                or identity.get('completed_update') != 30000
                or identity.get('run_id') != expected['run_id']
                or identity.get('seed') != expected['seed']
                or artifact.get('model_state_sha256') != identity.get('model_state_sha256')
                or file_sha(checkpoint) != artifact.get('file_sha256')
                or metadata.get('evaluation') != evaluation
                or artifact.get('identity') != evaluation.get('state_before')
                or evaluation.get('state_before') != evaluation.get('state_after')
                or local_path(root, artifact['checkpoint_path']).resolve() != checkpoint.resolve()):
            raise ValueError('Baseline provenance, same-state evaluation, or checkpoint SHA differs')
        binding = cfg.get('donor_binding', {})
        donor_identity = {key:binding.get(key) for key in DONOR_MATCH_KEYS}
        if (cfg.get('donor_checkpoint_sha256') != selected['file_sha256']
                or cfg.get('selector_receipt_sha256') != selector['receipt_sha256']
                or cfg.get('parent_run') != selected['source_run_id']
                or local_path(root, cfg.get('donor_checkpoint', '')).resolve() != donor_path
                or any(value is None for value in donor_identity.values())
                or donor_identity['donor_checkpoint_sha256'] != selected['file_sha256']
                or donor_identity['mature30_selector_sha256'] != selector['receipt_sha256']
                or donor_identity['donor_run_id'] != selected['source_run_id']
                or donor_identity['donor_step'] != selected['completed_update']
                or donor_identity['source_campaign_id'] != selected['source_campaign_id']
                or donor_identity['sensor'] != 'WV3'
                or donor_identity['source_split'] != 'RR_train_only'
                or donor_identity['catalog_sha256'] != expected['baseline_resolved']['source_registry_sha256']
                or donor_identity['mature30_validation_inventory_sha256'] != selector['validation_inventory']['inventory_sha256']):
            raise ValueError('Baseline donor/data binding differs from the selected same-server donor')
        result.append({'run_id':cfg['run_id'], 'replica':replica, 'seed':cfg['seed'],
                       'registered_config_sha256':previous.canonical_sha256(expected),
                       'dataset':cfg['dataset'], 'stream_seeds':cfg['stream_seeds'],
                       'augmentation':cfg['augmentation'], 'donor_identity':donor_identity,
                       'donor_selector_path':str(selector_path),
                       'donor_selector_sha256':selector['receipt_sha256'],
                       'checkpoint_path':str(checkpoint), 'checkpoint_sha256':artifact['file_sha256'],
                       'metadata_path':str(metadata_path), 'metadata_sha256':file_sha(metadata_path),
                       'evaluation_path':str(evaluation_path), 'evaluation_sha256':file_sha(evaluation_path),
                       'reference_only_no_weights_imported':True, 'reposted_to_sheet':False})
    return result


def require_matched_donor_binding(binding, controls):
    """Compare actual loaded A/T tensors and verified training inventory to controls."""
    actual = {key:binding.get(key) for key in DONOR_MATCH_KEYS}
    if (len(controls) != 2 or any(value is None for value in actual.values())
            or any(control.get('donor_identity') != actual for control in controls)):
        raise ValueError('Loaded sampling donor/data inventory differs from matched baseline controls')


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', choices=('s1', 's3'), required=True)
    parser.add_argument('--poll-seconds', type=float, default=20.)
    args = parser.parse_args(argv)
    if not 5 <= args.poll_seconds <= 60:
        raise ValueError('Poll interval must be between 5 and 60 seconds')
    root = Path(__file__).resolve().parents[1]
    if (root / 'gspread/server.txt').read_text().strip() != args.server:
        raise ValueError('Local server marker mismatch')
    directory = root / 'work_dir/panda_sampling30_launchers' / args.server
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
                   'source_pin':expected_pin, 'predecessor_modified':False,
                   'new_wall_time_limit':None, 'stop_or_skip_requested':False}

        def record(status, **details):
            receipt.update(status=status, updated_at=timestamp(), **details)
            atomic_json(directory / 'transition_receipt.json', receipt)
            print(json.dumps({k:v for k,v in receipt.items() if k != 'source_pin'}), flush=True)

        try:
            while True:
                if code_pin() != expected_pin:
                    raise RuntimeError('Successor source changed while waiting; no launch')
                container = inspect_owned_container(root, previous_name, args.server)
                if container is None or not source_state.is_file():
                    record('WAITING_PREVIOUS_QUEUE_START')
                    time.sleep(args.poll_seconds)
                    continue
                campaign = json.loads(source_state.read_text())
                decision = transition_decision(args.server, container['State'], campaign)
                if decision.startswith('WAITING_'):
                    record(decision, previous_status=campaign.get('status'),
                           previous_runs=[{key:row.get(key) for key in ('case','replica','step','status')}
                                          for row in campaign.get('runs', [])])
                    time.sleep(args.poll_seconds)
                    continue
                if 'baseline_controls' not in receipt:
                    receipt['baseline_controls'] = baseline_receipts(root, campaign, args.server)
                processes = subprocess.check_output(
                    ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
                    text=True, stderr=subprocess.STDOUT, timeout=20)
                if processes.strip():
                    record('WAITING_GPU_IDLE', active_compute=processes.strip())
                    time.sleep(args.poll_seconds)
                    continue
                record('STARTING_SUCCESSOR_PREFLIGHT', previous_status=campaign['status'])
                subprocess.run(['bash', 'scripts/start_panda_sampling30_costed.sh', args.server],
                               cwd=root, check=True)
                successor = f'panda-sampling30-hardonly-wv3-{args.server}-v1'
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
