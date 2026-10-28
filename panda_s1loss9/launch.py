"""Launch loss tuning and exact SENSOR5 continuation in the original runtime image."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from directshift48.reporting import atomic_json, utc_now

ROOT = Path(__file__).resolve().parents[1]


def directory(server):
    path = ROOT / 'work_dir/panda_s1loss9_launchers' / server
    path.mkdir(parents=True, exist_ok=True)
    return path


def original_container(server):
    from panda_sensor5.launch import sensor_container_name
    name = sensor_container_name(server)
    result = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True, check=True)
    source = json.loads(result.stdout)[0]
    if source['State']['Running'] or source['State'].get('OOMKilled'):
        raise ValueError('Original RGB/L4 SENSOR5 container must have safely stopped')
    if source['State'].get('Status') != 'exited' or source['State'].get('ExitCode') not in (0, 1):
        raise ValueError('Original SENSOR5 container did not exit safely')
    return name, source


def command(server, *, package, name=None):
    if package not in ('panda_sensor5', 'panda_s1loss9'):
        raise ValueError('Unsupported controlled campaign')
    if (ROOT / 'gspread/server.txt').read_text().strip() != server:
        raise ValueError('Local server marker mismatch')
    source_name, source = original_container(server)
    cmd = ['docker', 'run', '--init', '--gpus', 'all', '--pid=host', '--shm-size=8g',
           '--user', source['Config']['User'], '--workdir', source['Config']['WorkingDir']]
    cmd += ['--detach', '--name', name] if name else ['--rm']
    for item in source['Config']['Env']:
        cmd.extend(['--env', item])
    destination = '/workspace/work_dir/' + package
    own_output = False
    for mount in source['Mounts']:
        if mount['Type'] != 'bind':
            raise ValueError('Only original explicit bind mounts may be cloned')
        if any(any(x in mount[k] for x in (',', '\n', '\r')) for k in ('Source', 'Destination')):
            raise ValueError('Unsafe Docker bind mount path')
        if mount['Destination'].startswith('/workspace/work_dir/'):
            if mount['Destination'] != destination:
                continue
            own_output = True
        value = f"type=bind,src={mount['Source']},dst={mount['Destination']}"
        if not mount['RW']:
            value += ',readonly'
        cmd.extend(['--mount', value])
    if not own_output:
        output = ROOT / 'work_dir' / package
        output.mkdir(parents=True, exist_ok=True)
        cmd.extend(['--mount', f'type=bind,src={output},dst={destination}'])
    cmd.append(source['Image'])
    return cmd, source_name, source


def launch(server, mode, finish_run=None):
    if mode not in ('through-base', 'loss9', 'resume-sensor5'):
        raise ValueError('Unknown launch mode')
    if finish_run and mode != 'through-base':
        raise ValueError('Additional current-run boundary is only valid before loss9')
    from directshift48.__main__ import require_idle_gpu
    require_idle_gpu('cuda')
    output = directory(server)
    with (output / 'launch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = output / (mode + '.json')
        if receipt_path.exists():
            raise ValueError('Existing launch receipt must be inspected, never overwritten')
        package = 'panda_s1loss9' if mode == 'loss9' else 'panda_sensor5'
        if mode == 'loss9':
            from .orchestration import predecessor_receipt
            predecessor_receipt(ROOT, server)
            cmd, _, _ = command(server, package=package)
            subprocess.run(cmd + ['python', '-m', package, 'check', '--server', server], check=True)
            # Profiling writes only the new campaign's output and can safely
            # reuse completed profiles on a retry before the launch receipt.
            subprocess.run(cmd + ['python', '-m', package + '.cost', '--server', server,
                                 '--seconds', '1200'], check=True)
            require_idle_gpu('cuda')
        name = f'panda-s1loss9-{mode}-{server}-{time.time_ns()}'
        cmd, source_name, source = command(server, package=package, name=name)
        if mode == 'loss9':
            cmd += ['python', '-m', 'panda_s1loss9', 'run', '--server', server]
        else:
            cmd += ['python', '-m', 'panda_s1loss9.orchestration', '--server', server,
                    '--mode', 'through-base' if mode == 'through-base' else 'resume-all']
            if finish_run:
                cmd += ['--finish-run', finish_run]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        receipt = {'at': utc_now(), 'server': server, 'mode': mode, 'container': name,
                   'container_id': result.stdout.strip(), 'source_container': source_name,
                   'image': source['Image'], 'finish_run': finish_run,
                   'original_sensor5_training_code_unchanged': True}
        atomic_json(receipt_path, receipt)
        if mode == 'loss9':
            subprocess.run([sys.executable, '-m', 'panda_s1loss9.best_publication.restart',
                            '--server', server], cwd=ROOT, check=True)
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', choices=('s1', 's3'), required=True)
    parser.add_argument('--mode', choices=('through-base', 'loss9', 'resume-sensor5'), required=True)
    parser.add_argument('--finish-run')
    args = parser.parse_args(argv)
    print(json.dumps(launch(args.server, args.mode, args.finish_run)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
