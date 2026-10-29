"""Idempotent launches using the original authenticated Docker environments."""
import fcntl
import json
import subprocess
import sys

from directshift48.reporting import atomic_json, utc_now
from panda_sensor_refine.launch import inspect_optional, source_container
from .orchestration import ROOT, launcher_root, read


def command(server, package, *, name=None):
    if package not in ('panda_sensor_focus', 'panda_sensor_refine', 'panda_s1loss9',
                       'panda_sensor5', 'panda_tune30', 'panda_hard36'):
        raise ValueError('Unknown controlled package')
    if (ROOT / 'gspread/server.txt').read_text().strip() != server:
        raise ValueError('Local server marker mismatch')
    source_name, source = source_container(server, 'panda_sensor_refine' if package == 'panda_sensor_focus' else package)
    cmd = ['docker', 'run', '--init', '--gpus', 'all', '--pid=host', '--shm-size=8g',
           '--user', source['Config']['User'], '--workdir', source['Config']['WorkingDir']]
    cmd += ['--detach', '--name', name] if name else ['--rm']
    for item in source['Config']['Env']:
        cmd += ['--env', item]
    for mount in source['Mounts']:
        if mount['Type'] != 'bind' or any(any(ch in mount[k] for ch in (',', '\n', '\r')) for k in ('Source', 'Destination')):
            raise ValueError('Unsafe original bind mount')
        if mount['Destination'].startswith('/workspace/work_dir/'):
            continue
        value = f"type=bind,src={mount['Source']},dst={mount['Destination']}"
        if not mount['RW']:
            value += ',readonly'
        cmd += ['--mount', value]
    for relative in ('work_dir/' + package, 'work_dir/panda_sensor_focus_launchers'):
        path = ROOT / relative
        path.mkdir(parents=True, exist_ok=True)
        cmd += ['--mount', f'type=bind,src={path},dst=/workspace/{relative}']
    return cmd + [source['Image']], source_name, source


def arguments(server, kind, entry=None):
    if kind == 'focus':
        return 'panda_sensor_focus', ['python', '-m', 'panda_sensor_focus', 'run', '--server', server]
    if not kind.startswith('boundary_') or not entry:
        raise ValueError('Unknown focus launch kind')
    package = entry['package']
    if package in ('panda_tune30', 'panda_hard36'):
        if len(entry['run_ids']) != 1:
            raise ValueError('WV3 boundary must identify exactly one run')
        module = 'panda_sensor5.wv3' if package == 'panda_tune30' else 'panda_sensor5.wv3_continue'
        return package, ['python', '-m', module, '--server', server, '--package', package,
                         '--mode', 'finish-current', '--target-run', entry['run_ids'][0]]
    args = ['python', '-m', 'panda_sensor_focus.boundary', '--server', server, '--package', package,
            '--receipt', '/workspace/work_dir/panda_sensor_focus_launchers/' + server + '/' + kind + '_complete.json']
    for run_id in entry['run_ids']:
        args += ['--run-id', run_id]
    return package, args


def launch_job(server, kind, entry=None):
    from directshift48.__main__ import require_idle_gpu
    directory = launcher_root(ROOT, server)
    with (directory / 'launch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        package, args = arguments(server, kind, entry)
        receipt = directory / (kind + '.json')
        if receipt.exists():
            value = read(receipt)
            if value['args'] != args or value['package'] != package:
                raise ValueError('Existing launch request differs')
            return value
        token = read(directory / 'takeover.json')['token']
        name = f'panda-sensor-focus-{kind.replace("_", "-")}-{server}-{token}'
        cmd, source_name, source = command(server, package, name=name)
        current = inspect_optional(name)
        if current is None:
            require_idle_gpu('cuda')
            if kind == 'focus':
                check, _, _ = command(server, package)
                subprocess.run(check + ['python', '-m', package, 'check', '--server', server], check=True)
                subprocess.run(check + ['python', '-m', package + '.cost', '--server', server, '--seconds', '1200'], check=True)
                require_idle_gpu('cuda')
            current_id = subprocess.run(cmd + args, capture_output=True, text=True, check=True).stdout.strip()
        else:
            if current['Config']['Cmd'] != args or current['Image'] != source['Image']:
                raise ValueError('Container collision; no job launched')
            current_id = current['Id']
        value = {'server': server, 'kind': kind, 'package': package, 'args': args,
                 'container': name, 'container_id': current_id, 'source_container': source_name,
                 'image': source['Image'], 'at': utc_now()}
        atomic_json(receipt, value)
        return value


def wait_job(server, kind):
    saved = read(launcher_root(ROOT, server) / (kind + '.json'))
    current = inspect_optional(saved['container'])
    if (current is None or current['Id'] != saved['container_id']
            or current['Image'] != saved['image'] or current['Config']['Cmd'] != saved['args']):
        raise ValueError('Controlled container identity changed')
    if current['State']['Running']:
        return False
    if (current['State'].get('OOMKilled') or current['State'].get('Status') != 'exited'
            or current['State'].get('ExitCode') != 0):
        raise ValueError('Controlled job did not finish safely: ' + saved['container'])
    return True


def ensure_publisher(server):
    subprocess.run([sys.executable, '-m', 'panda_sensor_focus.best_publication.restart', '--server', server],
                   cwd=ROOT, check=True)
