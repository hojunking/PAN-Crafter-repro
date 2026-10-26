"""Safely replace SENSOR5 v1 with fresh RGB/Lanczos Stage1 and five Students.

The old campaign, containers, checkpoints, publication rows and WV3 receipts
remain intact. Before updating a live checkout, first stop its old processes
using the documented pre-update command; source pins deliberately reject edits.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from directshift48.reporting import atomic_json, utc_now
from .transition import ROOT, local, predecessor_receipt
from .launch import sensor_container_name

OLD_CAMPAIGN_ID = 'PANDA_S13_SENSOR5_STAGE12_20261026_v1'
REQUEST = 'USER_20261026_SENSOR5_RGB_LANCZOS4_FRESH_STAGE12'
RECEIPT = 'sensor5_rgb_l4_v2_migration.json'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            value.update(block)
    return value.hexdigest()


def inspect_optional(name):
    result = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True)
    if result.returncode:
        if 'No such object' in result.stderr or 'No such container' in result.stderr:
            return None
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)[0]


def old_root(root, server):
    sensor = {'s1': 'QB', 's3': 'GF2'}[server]
    return Path(root) / 'work_dir/panda_sensor5' / OLD_CAMPAIGN_ID / server / sensor


def old_container_name(server):
    sensor = {'s1': 'qb', 's3': 'gf2'}[server]
    return f'panda-sensor5-stage12-{sensor}-{server}-v1'


def process_matches(argv, cwd, root, server):
    def option(name):
        if argv.count(name) == 1 and argv.index(name) + 1 < len(argv):
            return argv[argv.index(name) + 1]
        return None
    if Path(cwd).resolve() != Path(root).resolve() or option('--server') != server:
        return False
    module = option('-m')
    if module == 'panda_sensor5.queue':
        return True
    return (module in ('panda_sensor5.best_publication', 'panda_sensor5.publication')
            and option('--root') == str(old_root(root, server).resolve())
            and '--apply' in argv and '--watch' in argv)


def process_identity(path):
    return {'pid': int(path.name),
        'argv': (path / 'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
        'cwd': str((path / 'cwd').resolve()),
        'generation': (path / 'stat').read_text().rsplit(')', 1)[1].split()[19]}


def matching_processes(root, server):
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            value = process_identity(path)
            if process_matches(value['argv'], value['cwd'], root, server):
                found.append(value)
        except (OSError, UnicodeError):
            continue
    return found


def stop_processes(root, server):
    found = matching_processes(root, server)
    for value in found:
        path = Path('/proc') / str(value['pid'])
        try:
            if process_identity(path) != value:
                raise ValueError('Process identity changed; no signal sent')
            os.kill(value['pid'], signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError):
            continue
    deadline = time.monotonic() + 20
    while matching_processes(root, server):
        if time.monotonic() >= deadline:
            raise ValueError('Old watcher has not stopped; trainer untouched')
        time.sleep(.2)
    return found


def stop_container(server):
    name = old_container_name(server)
    value = inspect_optional(name)
    if value is None or not value['State']['Running']:
        return value
    identity = value['Id']
    subprocess.run(['docker', 'kill', '--signal=SIGTERM', identity], check=True)
    print(json.dumps({'status': 'WAITING_CHECKPOINT_SAFE_STOP', 'container': name}), flush=True)
    # Never escalate to SIGKILL: an incomplete checkpoint must block migration.
    subprocess.run(['docker', 'wait', identity], check=True, timeout=180)
    value = inspect_optional(name)
    if value is None or value['Id'] != identity:
        raise ValueError('Old trainer container identity changed')
    return value


def verify_old_campaign(root, server, container):
    """Exit 1 is normal for CLI STOPPED_SAFE; require its actual saved evidence."""
    directory = old_root(root, server)
    state_path = directory / 'campaign_state.json'
    if not state_path.exists():
        if container is not None:
            raise ValueError('Old trainer exists without durable campaign state')
        return {'old_campaign_started': False}
    if container is None:
        raise ValueError('Old campaign exists but its container is missing')
    status = container['State']
    if (status.get('Running') or status.get('Status') != 'exited'
            or status.get('OOMKilled') or status.get('ExitCode') not in (0, 1)):
        raise ValueError('Old trainer did not stop safely')
    state = json.loads(state_path.read_text())
    terminal = state.get('status')
    if terminal == 'STOPPED_SAFE':
        if state.get('reason') != 'STOP_REQUESTED':
            raise ValueError('Old campaign stopped for an unrelated guard')
    elif terminal == 'FINISHED_REGISTERED_QUEUE':
        if status.get('ExitCode') != 0:
            raise ValueError('Completed old campaign exited unsuccessfully')
    else:
        raise ValueError('Old campaign is not safely stopped or complete')
    proofs = []
    for run in state.get('runs', []):
        if run.get('status') not in ('COMPLETED', 'STOPPED') or run.get('poisoned'):
            raise ValueError('Old campaign contains an unsafe run')
        run_dir = local(run['directory'], root).resolve()
        expected = directory / 'runs' / run['run_id'] / 'attempt_01'
        if run_dir != expected.resolve():
            raise ValueError('Old run is outside its registered campaign')
        if run.get('stop_checkpoint_error'):
            raise ValueError('Old run failed to preserve its stop checkpoint')
        if run['status'] != 'STOPPED':
            continue
        if type(run.get('step')) is not int or run['step'] < 1:
            raise ValueError('Stopped started run requires a saved update')
        pointer = run_dir / 'models/latest_resume.json'
        artifact = json.loads(pointer.read_text())
        checkpoint = local(artifact['checkpoint_path'], root)
        if (checkpoint.is_symlink() or checkpoint.resolve().parent != run_dir / 'models'
                or not checkpoint.is_file() or artifact.get('completed_update') != run['step']
                or artifact.get('campaign_resume_complete') is not True
                or artifact.get('checkpoint_verified') is not True
                or sha(checkpoint) != artifact.get('file_sha256')):
            raise ValueError('Stopped run lacks a verified complete resume artifact')
        saved = json.loads((run_dir / 'run_state.json').read_text())
        if any(saved.get(k) != run.get(k) for k in ('run_id', 'status', 'step')):
            raise ValueError('Stopped run and campaign state disagree')
        proofs.append({'run_id': run['run_id'], 'step': run['step'],
            'checkpoint_path': str(checkpoint.relative_to(Path(root))),
            'file_sha256': artifact['file_sha256'], 'resume_pointer_sha256': sha(pointer)})
    return {'old_campaign_started': True, 'old_status': terminal,
        'old_state_sha256': sha(state_path), 'container_id': container['Id'],
        'container_exit_code': status['ExitCode'], 'stopped_runs': proofs,
        'old_campaign_root': str(directory), 'old_results_modified': False}


def replacement_chain(chain, server, campaign_id):
    current = chain.get('stage', 'CUTOVER')
    if current not in ('CUTOVER', 'START_SENSOR5', 'WAIT_SENSOR5'):
        raise ValueError('WV3 continuation already advanced; inspect before replacing its queue')
    if (current == 'WAIT_SENSOR5'
            and chain.get('sensor_container') != old_container_name(server)):
        raise ValueError('Unexpected current sensor container')
    updated = dict(chain)
    updated.update(stage='CUTOVER' if current == 'CUTOVER' else 'START_SENSOR5',
        status='CUTOVER' if current == 'CUTOVER' else 'START_SENSOR5',
        sensor_campaign_id=campaign_id, sensor_migration_request=REQUEST)
    updated.pop('sensor_container', None)
    updated.pop('pid', None)
    return updated


def migrate(server, *, root=ROOT, apply=False, stop_running=False):
    from .contract import CAMPAIGN_ID
    root = Path(root).resolve()
    if (root / 'gspread/server.txt').read_text().strip() != server:
        raise ValueError('Local server marker mismatch')
    if CAMPAIGN_ID == OLD_CAMPAIGN_ID:
        raise ValueError('RGB/Lanczos campaign registration is not installed')
    directory = root / 'work_dir/panda_sensor5_launchers' / server
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'migration.lock').open('a+') as lock, \
            (directory / 'enqueue.lock').open('a+') as enqueue:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(enqueue, fcntl.LOCK_EX | fcntl.LOCK_NB)
        chain_path, receipt_path = directory / 'chain.json', directory / RECEIPT
        chain = json.loads(chain_path.read_text()) if chain_path.exists() else {'stage': 'CUTOVER'}
        if chain.get('sensor_campaign_id') == CAMPAIGN_ID:
            if not receipt_path.exists():
                raise ValueError('Migrated chain lacks its durable migration receipt')
            receipt = json.loads(receipt_path.read_text())
            if receipt.get('request') != REQUEST or receipt.get('new_campaign_id') != CAMPAIGN_ID:
                raise ValueError('Unexpected migration receipt')
            return {'status': 'ALREADY_MIGRATED', 'receipt': str(receipt_path)}
        updated = replacement_chain(chain, server, CAMPAIGN_ID)
        active = matching_processes(root, server)
        container = inspect_optional(old_container_name(server))
        if not apply:
            return {'status': 'DRY_RUN', 'server': server, 'old_stage': chain.get('stage'),
                'new_stage': updated['stage'], 'new_campaign_id': CAMPAIGN_ID,
                'watchers_to_stop': active,
                'trainer_running': bool(container and container['State']['Running']),
                'new_container': sensor_container_name(server),
                'old_results_preserved': True, 'wv3_queue_preserved': True}
        if active or (container and container['State']['Running']):
            if not stop_running:
                raise ValueError('Old processes still running; stop safely before update, '
                                 'or explicitly use --stop-running for an unchanged live checkout')
            stop_processes(root, server)
            container = stop_container(server)
        proof = verify_old_campaign(root, server, container)
        if chain.get('stage') != 'CUTOVER':
            predecessor_receipt(root, server)
        # Never re-use a populated new root or start over an existing v2 trainer.
        if inspect_optional(sensor_container_name(server)) is not None:
            raise ValueError('New sensor container already exists without committed migration')
        new_root = root / 'work_dir/panda_sensor5' / CAMPAIGN_ID / server / {'s1':'QB','s3':'GF2'}[server]
        if (new_root / 'campaign_state.json').exists():
            raise ValueError('New campaign already started without migration receipt')
        receipt = {'schema': 'PANDA_SENSOR5_RGB_L4_MIGRATION_V2', 'request': REQUEST,
            'server': server, 'new_campaign_id': CAMPAIGN_ID,
            'old_campaign_id': OLD_CAMPAIGN_ID, 'old_chain': chain, 'new_chain': updated,
            'old_campaign_proof': proof, 'fresh_stage1_and_students': True,
            'reuse_old_donor_or_calibration': False, 'wv3_queue_preserved': True}
        if receipt_path.exists():
            saved = json.loads(receipt_path.read_text())
            if {k:v for k,v in saved.items() if k != 'created_at'} != receipt:
                raise ValueError('Prepared migration differs; existing receipt preserved')
        else:
            atomic_json(receipt_path, dict(receipt, created_at=utc_now()))
        backup = directory / 'chain_before_rgb_l4_v2.json'
        if backup.exists() and json.loads(backup.read_text()) != chain:
            raise ValueError('Existing original chain backup differs')
        if not backup.exists():
            atomic_json(backup, chain)
        atomic_json(chain_path, updated)
        atomic_json(directory / 'status.json', updated)
        return {'status': 'MIGRATED_READY_TO_ENQUEUE', 'receipt': str(receipt_path),
                'new_stage': updated['stage'], 'new_campaign_id': CAMPAIGN_ID}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', choices=('s1', 's3'), required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--stop-running', action='store_true')
    args = parser.parse_args(argv)
    if args.stop_running and not args.apply:
        parser.error('--stop-running requires --apply')
    print(json.dumps(migrate(args.server, apply=args.apply, stop_running=args.stop_running)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
