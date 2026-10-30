"""s3-only append: wait for existing queues, profile, train, and verify Sheet closeout.

Run in a persistent host process. Never signals or replaces existing trainers.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from . import registry as G
from .prerequisites import evidence, read
from directshift48.reporting import atomic_json, utc_now

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / G.OUTPUT / 'launcher_s3'


def source_pin():
    """Match training's source pin without installing the process-local adapter."""
    from panda_sensor_focus.runtime import code_pin
    from .prerequisites import sha
    files = (p for p in (ROOT / 'panda_gf2_psnr').rglob('*.py') if 'tests' not in p.parts)
    return {**code_pin(), **{str(p.relative_to(ROOT)): sha(p) for p in files}}


def verify_source_pin(expected):
    if source_pin() != expected:
        raise RuntimeError('GF2 launcher source changed; stop and restart this waiter before '
                           'launching a revised queue. An admitted campaign requires an explicit migration.')


def command(name=None):
    from panda_sensor_refine.launch import source_container
    source_name, source = source_container('s3', 'panda_sensor5')
    cmd = ['docker', 'run', '--init', '--gpus', 'all', '--pid=host', '--shm-size=8g',
           '--user', source['Config']['User'], '--workdir', source['Config']['WorkingDir']]
    cmd += ['--detach', '--name', name] if name else ['--rm']
    for item in source['Config']['Env']:
        cmd += ['--env', item]
    for mount in source['Mounts']:
        if mount['Type'] != 'bind' or any(any(c in mount[k] for c in (',', '\n', '\r')) for k in ('Source', 'Destination')):
            raise ValueError('Unsafe original Docker bind')
        if mount['Destination'].startswith('/workspace/work_dir/'):
            continue
        spec = f"type=bind,src={mount['Source']},dst={mount['Destination']}"
        if not mount['RW']:
            spec += ',readonly'
        cmd += ['--mount', spec]
    output = ROOT / G.OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    cmd += ['--mount', f'type=bind,src={output},dst=/workspace/{G.OUTPUT}', source['Image']]
    return cmd, source_name, source['Image']


def ensure_publisher():
    expected = ['-m', 'panda_gf2_psnr', 'publish', '--apply', '--watch']
    matches = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = path.read_bytes().decode().strip('\0').split('\0')
            if len(argv) >= 2 and argv[1:] == ['-u'] + expected:
                matches.append(int(path.parent.name))
        except (OSError, UnicodeError):
            pass
    if len(matches) > 1:
        raise ValueError('Multiple GF2 PSNR publishers; inspect before proceeding')
    if matches:
        return matches[0]
    with (CONTROL / 'publisher.log').open('ab') as log:
        child = subprocess.Popen([sys.executable, '-u'] + expected, cwd=ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), start_new_session=True,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    return child.pid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true', help='Poll until existing focus/WV3 controller is complete')
    args = parser.parse_args()
    if (ROOT / 'gspread/server.txt').read_text().strip() != 's3':
        raise ValueError('Execute only on s3; s1 QB/WV3 are untouched')
    CONTROL.mkdir(parents=True, exist_ok=True)

    def status(stage, **extra):
        value = {'server': 's3', 'campaign_id': G.CAMPAIGN_ID, 'stage': stage, 'at': utc_now(), **extra}
        atomic_json(CONTROL / 'status.json', value)
        print(json.dumps(value), flush=True)

    from panda_sensor_refine.launch import inspect_optional
    from directshift48.__main__ import require_idle_gpu
    with (CONTROL / 'controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pinned = source_pin()
        campaign_root = ROOT / G.OUTPUT / G.CAMPAIGN_ID / 's3/GF2'
        admission = campaign_root / 'admission.json'
        if admission.exists() and read(admission).get('code') != pinned:
            raise ValueError('Existing GF2 admission has different source code; explicit migration required')
        atomic_json(CONTROL / 'launcher_source_pin.json', {
            'at': utc_now(), 'pid': os.getpid(), 'code': pinned,
            'campaign_id': G.CAMPAIGN_ID, 'runs': G.RUNS})

        def verify():
            try:
                verify_source_pin(pinned)
            except RuntimeError as exc:
                status('BLOCKED_SOURCE_CHANGED', reason=str(exc))
                raise

        while True:
            verify()
            try:
                proof = evidence(ROOT)
                break
            except (FileNotFoundError, ValueError) as exc:
                if not args.wait or not (isinstance(exc, FileNotFoundError) or str(exc).startswith('WAIT:')):
                    raise
                status('WAIT_EXISTING_QUEUES', reason=str(exc))
                time.sleep(30)
        verify()
        atomic_json(CONTROL / 'predecessor.json', proof)
        name = 'panda-gf2-psnr4508-s3-v1'
        train_args = ['python', '-m', 'panda_gf2_psnr', 'run']
        launch, source_name, image = command(name)
        current = inspect_optional(name)
        if current is None:
            require_idle_gpu('cuda')
            check, _, _ = command()
            status('CHECKING_AND_PROFILING')
            verify()
            subprocess.run(check + ['python', '-m', 'panda_gf2_psnr', 'check'], cwd=ROOT, check=True)
            verify()
            subprocess.run(check + ['python', '-m', 'panda_gf2_psnr', 'cost'], cwd=ROOT, check=True)
            require_idle_gpu('cuda')
            verify()
            subprocess.run(launch + train_args, cwd=ROOT, check=True)
            current = inspect_optional(name)
        if current['Image'] != image or current['Config']['Cmd'] != train_args:
            raise ValueError('GF2 followup container identity collision')
        receipt = {'container': name, 'container_id': current['Id'], 'image': image, 'args': train_args,
                   'source_container': source_name}
        path = CONTROL / 'launch.json'
        if path.exists() and read(path) != receipt:
            raise ValueError('Registered GF2 container was replaced')
        atomic_json(path, receipt)
        root = campaign_root
        while True:
            verify()
            pid = ensure_publisher()
            current = inspect_optional(name)
            if current is None or current['Id'] != receipt['container_id']:
                raise ValueError('GF2 container disappeared/replaced')
            if current['State']['Running']:
                status('RUNNING', publisher_pid=pid, **receipt)
            else:
                if current['State'].get('OOMKilled') or current['State'].get('ExitCode') != 0:
                    status('FAILED_TRAINING', docker_state=current['State'])
                    return 1
                state = read(root / 'campaign_state.json')
                expected = set(G.RUNS)
                rows = state.get('runs', [])
                if (state['status'] != 'FINISHED_REGISTERED_QUEUE' or len(rows) != len(expected)
                        or {(r['case'], r['replica']) for r in rows} != expected
                        or any(r['status'] != 'COMPLETED' or r['step'] != G.cases()[r['case']]['total_updates'] for r in rows)):
                    raise ValueError('GF2 followup ended without completing the registered eight runs')
                upload = root / 'sheet_publish_panda_sensor_focus/terminal_upload.json'
                if upload.exists() and read(upload).get('readback_verified'):
                    status('COMPLETED', sheet_verified=True, **receipt)
                    return 0
                status('WAIT_SHEET_READBACK', publisher_pid=pid)
            time.sleep(30)


if __name__ == '__main__':
    raise SystemExit(main())
