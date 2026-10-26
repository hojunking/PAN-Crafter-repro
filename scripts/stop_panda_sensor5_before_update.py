#!/usr/bin/env python3
"""Standalone pre-checkout safe stop; run from a fetched blob, before git pull.

Uses only the standard library and never imports the checkout being replaced.
Only exact SENSOR5 v1 processes for the declared root/server can be signalled.
It does not signal WV3, delete artifacts or start a new campaign.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import time


def option(argv, name):
    if argv.count(name) == 1 and argv.index(name) + 1 < len(argv):
        return argv[argv.index(name) + 1]
    return None


def identity(path):
    return {'pid': int(path.name), 'cwd': str((path / 'cwd').resolve()),
        'argv': (path / 'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
        'generation': (path / 'stat').read_text().rsplit(')', 1)[1].split()[19]}


def matching(root, server, old_root):
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            item = identity(path)
            argv = item['argv']
            if item['cwd'] != str(root) or option(argv, '--server') != server:
                continue
            module = option(argv, '-m')
            if (module == 'panda_sensor5.queue' or
                    (module in ('panda_sensor5.publication', 'panda_sensor5.best_publication')
                     and option(argv, '--root') == str(old_root)
                     and '--apply' in argv and '--watch' in argv)):
                found.append(item)
        except (OSError, UnicodeError):
            continue
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--server', choices=('s1', 's3'), required=True)
    args = parser.parse_args(argv)
    root, server = args.root.resolve(strict=True), args.server
    if (root / 'gspread/server.txt').read_text().strip() != server:
        raise ValueError('Local server marker mismatch')
    sensor = {'s1': 'QB', 's3': 'GF2'}[server]
    old_root = root / 'work_dir/panda_sensor5/PANDA_S13_SENSOR5_STAGE12_20261026_v1' / server / sensor
    launcher = root / 'work_dir/panda_sensor5_launchers' / server
    launcher.mkdir(parents=True, exist_ok=True)
    chain_path = launcher / 'chain.json'
    if chain_path.exists():
        chain = json.loads(chain_path.read_text())
        if chain.get('sensor_campaign_id', '').endswith('_v2'):
            raise ValueError('Already migrated to v2; no process changed')
        if chain.get('stage') not in ('CUTOVER', 'START_SENSOR5', 'WAIT_SENSOR5'):
            raise ValueError('WV3 continuation already advanced; no process changed')
    processes = matching(root, server, old_root)
    for item in processes:
        path = Path('/proc') / str(item['pid'])
        try:
            if identity(path) != item:
                raise ValueError('Watcher identity changed; stop and inspect')
            os.kill(item['pid'], signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError):
            continue
    deadline = time.monotonic() + 20
    while matching(root, server, old_root):
        if time.monotonic() >= deadline:
            raise ValueError('Watcher has not exited; trainer untouched')
        time.sleep(.2)
    name = f'panda-sensor5-stage12-{sensor.lower()}-{server}-v1'
    query = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True)
    container = None
    if query.returncode == 0:
        container = json.loads(query.stdout)[0]
        if container['State']['Running']:
            subprocess.run(['docker', 'kill', '--signal=SIGTERM', container['Id']], check=True)
            print('Waiting for complete checkpoint-safe stop; no SIGKILL will be used.', flush=True)
            subprocess.run(['docker', 'wait', container['Id']], check=True, timeout=180)
            container = json.loads(subprocess.run(['docker', 'inspect', name],
                check=True, capture_output=True, text=True).stdout)[0]
        state_path = old_root / 'campaign_state.json'
        state = json.loads(state_path.read_text())
        docker_state = container['State']
        if (docker_state.get('Running') or docker_state.get('Status') != 'exited'
                or docker_state.get('OOMKilled')
                or (state.get('status'), state.get('reason'), docker_state.get('ExitCode'))
                not in (('STOPPED_SAFE', 'STOP_REQUESTED', 1),
                        ('STOPPED_SAFE', 'STOP_REQUESTED', 0),
                        ('FINISHED_REGISTERED_QUEUE', None, 0))):
            raise ValueError('Trainer not confirmed safe; do not update checkout')
        for run in state.get('runs', []):
            if run.get('status') not in ('COMPLETED', 'STOPPED') or run.get('poisoned'):
                raise ValueError('Unsafe old run; do not update checkout')
            if run['status'] != 'STOPPED':
                continue
            run_dir = old_root / 'runs' / run['run_id'] / 'attempt_01'
            artifact = json.loads((run_dir / 'models/latest_resume.json').read_text())
            path = Path(artifact['checkpoint_path'])
            if str(path).startswith('/workspace/'):
                path = root / path.relative_to('/workspace')
            if (path.is_symlink() or path.resolve().parent != (run_dir / 'models').resolve()
                    or artifact.get('completed_update') != run.get('step')
                    or artifact.get('campaign_resume_complete') is not True
                    or artifact.get('checkpoint_verified') is not True):
                raise ValueError('Incomplete stop checkpoint; do not update checkout')
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(4 * 1024 ** 2), b''):
                    digest.update(block)
            if digest.hexdigest() != artifact.get('file_sha256'):
                raise ValueError('Stop checkpoint SHA changed; do not update checkout')
    elif 'No such object' not in query.stderr and 'No such container' not in query.stderr:
        raise RuntimeError(query.stderr.strip())
    archive = launcher / f'legacy_source_before_rgb_l4_{time.time_ns()}.tar.gz'
    with tarfile.open(archive, 'x:gz') as output:
        for path in sorted((root / 'panda_sensor5').rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.csv', '.md'):
                output.add(path, arcname=str(path.relative_to(root)))
        for path in sorted((root / 'scripts').glob('*panda_sensor5*')):
            if path.is_file():
                output.add(path, arcname=str(path.relative_to(root)))
    value = {'status': 'SAFE_TO_UPDATE_CHECKOUT', 'server': server,
        'old_container': name, 'old_campaign_root': str(old_root),
        'source_archive': str(archive),
        'source_archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'stopped_watchers': processes, 'old_results_modified': False}
    print(json.dumps(value, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
