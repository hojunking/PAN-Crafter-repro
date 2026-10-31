"""Launch each server's QB Student queue using its own completed local Stage1."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from directshift48.reporting import atomic_json, utc_now
from . import registry as G

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def control_dir(server):
    return ROOT / G.OUTPUT / ('launcher_' + server)


def training_args(server):
    return ['python', '-m', 'panda_qb_structure', 'run', '--server', server]


def container_name(server):
    return f'panda-qb-structure-{server}-v1'


def wait_for_gpu(wait, status):
    from directshift48.__main__ import require_idle_gpu
    while True:
        try:
            require_idle_gpu('cuda')
            return
        except RuntimeError as exc:
            if not wait or not str(exc).startswith('Active GPU compute workload;'):
                raise
            status('WAIT_LOCAL_GPU', reason=str(exc))
            time.sleep(30)


def same_environment(left, right):
    def canonical(value):
        return {**value, 'env': sorted(value['env']),
                'mounts': sorted(value['mounts'], key=lambda m: (m['Destination'], m['Source'], m['RW']))}
    return canonical(left) == canonical(right)


def command(server, name=None, *, token=None):
    if server not in G.RUNS:
        raise ValueError('Only registered s1/s3 QB queues are supported')
    from panda_sensor_refine.launch import source_container
    source_name, source = source_container(server, 'panda_sensor5')
    cmd = ['docker', 'run', '--init', '--gpus', 'all', '--pid=host', '--shm-size=8g',
           '--user', source['Config']['User'], '--workdir', source['Config']['WorkingDir']]
    cmd += ['--detach', '--name', name] if name else ['--rm']
    if name:
        if not token:
            raise ValueError('A detached launch must have an ownership token')
        cmd += ['--label', 'panda.campaign=' + G.CAMPAIGN_ID,
                '--label', 'panda.launch-token=' + token]
    for item in source['Config']['Env']:
        cmd += ['--env', item]
    mounts = []
    for mount in source['Mounts']:
        if mount['Type'] != 'bind' or any(any(c in mount[k] for c in (',', '\n', '\r'))
                                        for k in ('Source', 'Destination')):
            raise ValueError('Unsupported original Docker bind')
        if mount['Destination'].startswith('/workspace/work_dir/'):
            continue
        if mount['Destination'] == '/workspace' and mount['RW']:
            raise ValueError('The source workspace must remain read-only')
        spec = f"type=bind,src={mount['Source']},dst={mount['Destination']}"
        if not mount['RW']:
            spec += ',readonly'
        cmd += ['--mount', spec]
        mounts.append({k: mount[k] for k in ('Source', 'Destination', 'RW')})
    # The original s3 SENSOR5 image was used for GF2; include QB symlink targets.
    from edgeres48.docker_mounts import targets
    for path in targets(ROOT, 's2'):  # historical sensor key s2 denotes QB
        if any(m['Destination'] == str(path) for m in mounts):
            continue
        if any(c in str(path) for c in (',', '\n', '\r')):
            raise ValueError('Unsupported QB data path')
        cmd += ['--mount', f'type=bind,src={path},dst={path},readonly']
        mounts.append({'Source': str(path), 'Destination': str(path), 'RW': False})
    output = ROOT / G.OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    cmd += ['--mount', f'type=bind,src={output},dst=/workspace/{G.OUTPUT}', source['Image']]
    mounts.append({'Source': str(output), 'Destination': '/workspace/' + G.OUTPUT, 'RW': True})
    identity = {'source_container': source_name, 'image': source['Image'],
                'user': source['Config']['User'], 'working_dir': source['Config']['WorkingDir'],
                'env': source['Config']['Env'], 'mounts': mounts}
    return cmd, identity


def authenticate_container(server, current, intent, receipt=None):
    if current is None:
        raise ValueError('The registered QB container is missing')
    cfg, expected = current['Config'], intent['identity']
    labels = cfg.get('Labels') or {}
    mounts = [{k: m[k] for k in ('Source', 'Destination', 'RW')} for m in current['Mounts']]
    if (current['Image'] != expected['image'] or cfg['Cmd'] != training_args(server)
            or cfg['User'] != expected['user'] or cfg['WorkingDir'] != expected['working_dir']
            or sorted(cfg['Env']) != sorted(expected['env'])
            or sorted(mounts, key=lambda m: m['Destination']) != sorted(expected['mounts'], key=lambda m: m['Destination'])
            or labels.get('panda.campaign') != G.CAMPAIGN_ID
            or labels.get('panda.launch-token') != intent['token']
            or (receipt is not None and current['Id'] != receipt['container_id'])):
        raise ValueError('QB container ownership or execution environment differs')
    if receipt is None and not current['State']['Running']:
        raise ValueError('Cannot adopt an exited QB container without its launch receipt')
    return current


def ensure_publisher(server):
    expected = ['-u', '-m', 'panda_qb_structure', 'publish', '--server', server, '--apply', '--watch']
    matches = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = path.read_bytes().decode().strip('\0').split('\0')
            if argv[1:] == expected:
                if (path.parent / 'cwd').resolve() != ROOT.resolve():
                    raise ValueError('A QB publisher from another checkout exists')
                matches.append(int(path.parent.name))
        except (OSError, UnicodeError):
            continue
    if len(matches) > 1:
        raise ValueError('Multiple publishers are writing the same QB campaign')
    if matches:
        return matches[0]
    with (control_dir(server) / 'publisher.log').open('ab') as log:
        child = subprocess.Popen([sys.executable] + expected, cwd=ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), start_new_session=True,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    return child.pid


def launch_intent(server, current, receipt, registry_digest):
    control = control_dir(server)
    path = control / 'launch_intent.json'
    saved = read(path) if path.exists() else None
    if saved and (saved.get('campaign_id') != G.CAMPAIGN_ID
                  or saved.get('registry_digest') != registry_digest or saved.get('server') != server):
        campaign = ROOT / G.OUTPUT / G.CAMPAIGN_ID / server / 'QB/campaign_state.json'
        started = campaign.exists() and bool(read(campaign).get('runs'))
        if current is not None or receipt is not None or started:
            raise ValueError('An already-started campaign cannot be relabelled with a different registry')
        history = control / 'intent_history'
        history.mkdir(exist_ok=True)
        atomic_json(history / (uuid.uuid4().hex + '.json'), saved)
        saved = None
    if saved is None:
        if current is not None or receipt is not None:
            raise ValueError('An existing QB container/receipt has no matching launch intent')
        token = uuid.uuid4().hex
        _, identity = command(server, container_name(server), token=token)
        saved = {'server': server, 'campaign_id': G.CAMPAIGN_ID, 'token': token,
                 'identity': identity, 'registry_digest': registry_digest, 'at': utc_now()}
        atomic_json(path, saved)
    return saved


def execute(server, wait, status):
    from panda_sensor_refine.launch import inspect_optional
    from directshift48.__main__ import require_idle_gpu
    from donor30.plan import canonical_sha256
    from .prerequisites import terminal
    control = control_dir(server)
    receipt_path = control / 'launch.json'
    receipt = read(receipt_path) if receipt_path.exists() else None
    current = inspect_optional(container_name(server))
    digest = canonical_sha256(G.manifest())
    intent = launch_intent(server, current, receipt, digest)
    launch, identity = command(server, container_name(server), token=intent['token'])
    if not same_environment(identity, intent['identity']):
        raise ValueError('Runtime environment differs from this launch intent')
    if current is None:
        if receipt is not None:
            raise ValueError('A previously launched container is missing; inspect the retained state')
        wait_for_gpu(wait, status)
        check, _ = command(server)
        status('CHECKING_AND_PROFILING')
        subprocess.run(check + ['python', '-m', 'panda_qb_structure', 'check', '--server', server], cwd=ROOT, check=True)
        subprocess.run(check + ['python', '-m', 'panda_qb_structure', 'cost', '--server', server], cwd=ROOT, check=True)
        require_idle_gpu('cuda')
        subprocess.run(launch + training_args(server), cwd=ROOT, check=True)
        current = inspect_optional(container_name(server))
    authenticate_container(server, current, intent, receipt)
    if receipt is None:
        receipt = {'container': container_name(server), 'container_id': current['Id'],
                   'image': identity['image'], 'args': training_args(server),
                   'source_container': identity['source_container'], 'token': intent['token'],
                   'registry_digest': digest}
        atomic_json(receipt_path, receipt)
    folder = ROOT / G.OUTPUT / G.CAMPAIGN_ID / server / 'QB'
    while True:
        current = authenticate_container(server, inspect_optional(container_name(server)), intent, receipt)
        pid = ensure_publisher(server)
        if current['State']['Running']:
            status('RUNNING', publisher_pid=pid, **receipt)
        else:
            if (current['State'].get('Status') != 'exited' or current['State'].get('OOMKilled')
                    or current['State'].get('ExitCode') != 0):
                status('FAILED_TRAINING', docker_state=current['State'], auto_restart=False)
                return 1
            finished = terminal(ROOT, server)
            upload = folder / 'sheet_publish_panda_sensor_focus/terminal_upload.json'
            if upload.exists():
                report = read(upload)
                if report.get('readback_verified') and report.get('campaign_status') == finished['campaign_status']:
                    atomic_json(control / 'completion.json', finished)
                    status('COMPLETED', sheet_verified=True, completed_count=finished['completed_count'],
                           early_stopped_count=finished['early_stopped_count'], **receipt)
                    return 0
            status('WAIT_SHEET_READBACK', publisher_pid=pid, completed_count=finished['completed_count'],
                   early_stopped_count=finished['early_stopped_count'])
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True, choices=('s1', 's3'))
    parser.add_argument('--wait', action='store_true', help='Wait only for this server GPU to become available')
    args = parser.parse_args()
    if (ROOT / 'gspread/server.txt').read_text().strip() != args.server:
        raise ValueError('Requested server does not match the local server marker')
    control = control_dir(args.server)
    control.mkdir(parents=True, exist_ok=True)

    def status(stage, **extra):
        value = {'server': args.server, 'campaign_id': G.CAMPAIGN_ID, 'stage': stage,
                 'at': utc_now(), 'pid': os.getpid(), **extra}
        atomic_json(control / 'status.json', value)
        print(json.dumps(value), flush=True)

    # Only exclude duplicate writers to this same local output directory.
    with (control / 'controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return execute(args.server, args.wait, status)
        except Exception as exc:
            status('FAILED_LAUNCH', error=repr(exc), auto_restart=False)
            raise


if __name__ == '__main__':
    raise SystemExit(main())
