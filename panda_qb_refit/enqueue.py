"""Append QB refit after the existing s1 focus/WV3 queues and Sheet closeout.

The waiting process pins its code and registry. It never stops an existing job,
prunes a case, adopts a foreign container, or automatically restarts a failure.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from . import registry as G
from .prerequisites import evidence, read, terminal
from directshift48.reporting import atomic_json, utc_now

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / G.OUTPUT / 'launcher_s1'
TRAIN_ARGS = ['python', '-m', 'panda_qb_refit', 'run']
CONTAINER = 'panda-qb-refit-s1-v1'


def source_pin():
    from panda_sensor_focus.runtime import code_pin
    from .adapter import own_files
    from .prerequisites import sha
    from donor30.plan import canonical_sha256
    return {'files': {**code_pin(), **{str(p.relative_to(ROOT)): sha(p) for p in own_files(ROOT)}},
            'registry_digest': canonical_sha256(G.manifest())}


def verify_pin(expected):
    if source_pin() != expected:
        raise ValueError('QB waiting-controller code/registry changed; use a reviewed versioned boundary migration')


def command(name=None, *, token=None):
    from panda_sensor_refine.launch import source_container
    source_name, source = source_container('s1', 'panda_sensor5')
    cmd = ['docker', 'run', '--init', '--gpus', 'all', '--pid=host', '--shm-size=8g',
           '--user', source['Config']['User'], '--workdir', source['Config']['WorkingDir']]
    cmd += ['--detach', '--name', name] if name else ['--rm']
    if name:
        if not token:
            raise ValueError('Owned detached launch requires an intent token')
        cmd += ['--label', 'panda.campaign=' + G.CAMPAIGN_ID, '--label', 'panda.launch-token=' + token]
    for item in source['Config']['Env']:
        cmd += ['--env', item]
    mounts = []
    for mount in source['Mounts']:
        if mount['Type'] != 'bind' or any(any(c in mount[k] for c in (',', '\n', '\r')) for k in ('Source', 'Destination')):
            raise ValueError('Unsafe original Docker bind')
        if mount['Destination'].startswith('/workspace/work_dir/'):
            continue
        if mount['Destination'] == '/workspace' and mount['RW']:
            raise ValueError('Source workspace must remain read-only')
        spec = f"type=bind,src={mount['Source']},dst={mount['Destination']}"
        if not mount['RW']:
            spec += ',readonly'
        cmd += ['--mount', spec]
        mounts.append({k: mount[k] for k in ('Source', 'Destination', 'RW')})
    output = ROOT / G.OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    cmd += ['--mount', f'type=bind,src={output},dst=/workspace/{G.OUTPUT}', source['Image']]
    mounts.append({'Source': str(output), 'Destination': '/workspace/' + G.OUTPUT, 'RW': True})
    identity = {'source_container': source_name, 'image': source['Image'],
                'user': source['Config']['User'], 'working_dir': source['Config']['WorkingDir'],
                'env': source['Config']['Env'], 'mounts': mounts}
    return cmd, identity


def authenticate_container(current, intent, receipt=None):
    if current is None:
        raise ValueError('Registered QB container disappeared')
    cfg, expected = current['Config'], intent['identity']
    labels = cfg.get('Labels') or {}
    mounts = [{k: m[k] for k in ('Source', 'Destination', 'RW')} for m in current['Mounts']]
    if (current['Image'] != expected['image'] or cfg['Cmd'] != TRAIN_ARGS
            or cfg['User'] != expected['user'] or cfg['WorkingDir'] != expected['working_dir']
            or sorted(cfg['Env']) != sorted(expected['env'])
            or sorted(mounts, key=lambda m: m['Destination']) != sorted(expected['mounts'], key=lambda m: m['Destination'])
            or labels.get('panda.campaign') != G.CAMPAIGN_ID
            or labels.get('panda.launch-token') != intent['token']
            or (receipt is not None and current['Id'] != receipt['container_id'])):
        raise ValueError('QB followup container identity collision/replacement')
    if receipt is None and not current['State']['Running']:
        raise ValueError('Refuse to adopt an exited QB container without an authenticated launch receipt')
    return current


def ensure_publisher():
    expected = ['-u', '-m', 'panda_qb_refit', 'publish', '--apply', '--watch']
    matches = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = path.read_bytes().decode().strip('\0').split('\0')
            if len(argv) >= 2 and argv[1:] == expected:
                if (path.parent / 'cwd').resolve() != ROOT.resolve():
                    raise ValueError('A foreign-checkout QB publisher exists')
                matches.append(int(path.parent.name))
        except (OSError, UnicodeError):
            pass
    if len(matches) > 1:
        raise ValueError('Multiple QB refit publishers; inspect before proceeding')
    if matches:
        return matches[0]
    with (CONTROL / 'publisher.log').open('ab') as log:
        child = subprocess.Popen([sys.executable] + expected, cwd=ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), start_new_session=True,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    return child.pid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true', help='Wait for the existing focus/WV3 controller')
    args = parser.parse_args()
    if (ROOT / 'gspread/server.txt').read_text().strip() != 's1':
        raise ValueError('Execute only on s1; s3 GF2 and existing WV3 queues are untouched')
    CONTROL.mkdir(parents=True, exist_ok=True)

    def status(stage, **extra):
        value = {'server': 's1', 'campaign_id': G.CAMPAIGN_ID, 'stage': stage, 'at': utc_now(), **extra}
        atomic_json(CONTROL / 'status.json', value)
        print(json.dumps(value), flush=True)

    from panda_sensor_refine.launch import inspect_optional
    from directshift48.__main__ import require_idle_gpu
    with (CONTROL / 'controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pin_path = CONTROL / 'controller_pin.json'
        pin = read(pin_path) if pin_path.exists() else source_pin()
        verify_pin(pin)
        if not pin_path.exists():
            atomic_json(pin_path, pin)
        while True:
            verify_pin(pin)
            try:
                proof = evidence(ROOT, expected_code={k: v for k, v in pin['files'].items()
                    if not k.startswith('panda_qb_refit/')})
                break
            except ValueError as exc:
                if not args.wait or not str(exc).startswith('WAIT:'):
                    raise
                status('WAIT_EXISTING_QUEUES', reason=str(exc))
                time.sleep(30)
        atomic_json(CONTROL / 'predecessor.json', proof)
        intent_path, receipt_path = CONTROL / 'launch_intent.json', CONTROL / 'launch.json'
        current = inspect_optional(CONTAINER)
        if intent_path.exists():
            intent = read(intent_path)
            if intent['campaign_id'] != G.CAMPAIGN_ID or intent['registry_digest'] != pin['registry_digest']:
                raise ValueError('QB launch intent differs from pinned registry')
        else:
            if current is not None:
                raise ValueError('Foreign QB container exists without this controller launch intent')
            token = uuid.uuid4().hex
            _, identity = command(CONTAINER, token=token)
            intent = {'campaign_id': G.CAMPAIGN_ID, 'token': token, 'identity': identity,
                      'registry_digest': pin['registry_digest'], 'at': utc_now()}
            atomic_json(intent_path, intent)
        receipt = read(receipt_path) if receipt_path.exists() else None
        launch, identity = command(CONTAINER, token=intent['token'])
        if identity != intent['identity']:
            raise ValueError('Original runtime environment differs from the saved launch intent')
        if current is None:
            if receipt is not None:
                raise ValueError('Registered QB container missing; explicit authenticated recovery required')
            require_idle_gpu('cuda')
            check, _ = command()
            status('CHECKING_AND_PROFILING')
            subprocess.run(check + ['python', '-m', 'panda_qb_refit', 'check'], cwd=ROOT, check=True)
            subprocess.run(check + ['python', '-m', 'panda_qb_refit', 'cost'], cwd=ROOT, check=True)
            verify_pin(pin)
            require_idle_gpu('cuda')
            subprocess.run(launch + TRAIN_ARGS, cwd=ROOT, check=True)
            current = inspect_optional(CONTAINER)
        authenticate_container(current, intent, receipt)
        if receipt is None:
            receipt = {'container': CONTAINER, 'container_id': current['Id'], 'image': identity['image'],
                       'args': TRAIN_ARGS, 'source_container': identity['source_container'],
                       'token': intent['token'], 'registry_digest': pin['registry_digest']}
            atomic_json(receipt_path, receipt)
        folder = ROOT / G.OUTPUT / G.CAMPAIGN_ID / 's1/QB'
        while True:
            verify_pin(pin)
            current = authenticate_container(inspect_optional(CONTAINER), intent, receipt)
            pid = ensure_publisher()
            if current['State']['Running']:
                status('RUNNING', publisher_pid=pid, **receipt)
            else:
                if (current['State'].get('Status') != 'exited' or current['State'].get('OOMKilled')
                        or current['State'].get('ExitCode') != 0):
                    status('FAILED_TRAINING', docker_state=current['State'], auto_restart=False)
                    return 1
                finished = terminal(ROOT)
                upload = folder / 'sheet_publish_panda_sensor_focus/terminal_upload.json'
                if upload.exists():
                    report = read(upload)
                    if report.get('readback_verified') and report.get('campaign_status') == finished['campaign_status']:
                        atomic_json(CONTROL / 'completion.json', finished)
                        status('COMPLETED', sheet_verified=True,
                               completed_count=finished['completed_count'],
                               early_stopped_count=finished['early_stopped_count'], **receipt)
                        return 0
                status('WAIT_SHEET_READBACK', publisher_pid=pid,
                       completed_count=finished['completed_count'], early_stopped_count=finished['early_stopped_count'])
            time.sleep(30)


if __name__ == '__main__':
    raise SystemExit(main())
