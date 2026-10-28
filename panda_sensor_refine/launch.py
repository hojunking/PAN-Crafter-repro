"""Idempotent launches in each campaign's authenticated original Docker image."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from directshift48.reporting import atomic_json, utc_now
from .orchestration import ROOT, launcher_root, read


def inspect_optional(name):
    result = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True)
    if result.returncode:
        if 'No such object' in result.stderr or 'No such container' in result.stderr:
            return None
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)[0]


def source_container(server, package):
    if package in ('panda_sensor_refine', 'panda_s1loss9', 'panda_sensor5'):
        from panda_sensor5.launch import sensor_container_name
        name = sensor_container_name(server)
    else:
        from panda_sensor5.wv3_launch import source_container as original
        return original(server, package)
    source = inspect_optional(name)
    if (source is None or source['State']['Running'] or source['State'].get('OOMKilled')
            or source['State'].get('Status') != 'exited' or source['State'].get('ExitCode') not in (0, 1)):
        raise ValueError('Original SENSOR5 runtime container must be safely exited')
    return name, source


def command(server, package, *, name=None):
    if package not in ('panda_sensor_refine', 'panda_s1loss9', 'panda_sensor5', 'panda_tune30', 'panda_hard36'):
        raise ValueError('Unknown controlled package')
    if (ROOT / 'gspread/server.txt').read_text().strip() != server:
        raise ValueError('Local server marker mismatch')
    source_name, source = source_container(server, package)
    cmd = ['docker', 'run', '--init', '--gpus', 'all', '--pid=host', '--shm-size=8g',
           '--user', source['Config']['User'], '--workdir', source['Config']['WorkingDir']]
    cmd += ['--detach', '--name', name] if name else ['--rm']
    for env in source['Config']['Env']:
        cmd += ['--env', env]
    # Each campaign writes only its own output; controller evidence is separate.
    destinations = {f'/workspace/work_dir/{package}': ROOT / 'work_dir' / package,
                    '/workspace/work_dir/panda_sensor_refine_launchers': ROOT / 'work_dir/panda_sensor_refine_launchers'}
    for mount in source['Mounts']:
        if mount['Type'] != 'bind' or any(any(ch in mount[k] for ch in (',', '\n', '\r')) for k in ('Source', 'Destination')):
            raise ValueError('Unsafe or unsupported original bind mount')
        if mount['Destination'].startswith('/workspace/work_dir/'):
            continue
        value = f"type=bind,src={mount['Source']},dst={mount['Destination']}"
        if not mount['RW']: value += ',readonly'
        cmd += ['--mount', value]
    for destination, path in destinations.items():
        path.mkdir(parents=True, exist_ok=True)
        cmd += ['--mount', f'type=bind,src={path},dst={destination}']
    return cmd + [source['Image']], source_name, source


def job_arguments(server, kind, target=None):
    if kind == 'legacy': return 'panda_s1loss9', ['python', '-m', 'panda_sensor_refine.orchestration', '--server', server]
    if kind == 'refine': return 'panda_sensor_refine', ['python', '-m', 'panda_sensor_refine', 'run', '--server', server]
    if kind == 'sensor-boundary':
        args = ['python', '-m', 'panda_s1loss9.orchestration', '--server', server, '--mode', 'through-base']
        if target: args += ['--finish-run', target]
        return 'panda_sensor5', args
    if kind in ('tune-boundary', 'hard-boundary'):
        if not target: raise ValueError('WV3 boundary requires a registered started run')
        package = 'panda_tune30' if kind == 'tune-boundary' else 'panda_hard36'
        module = 'panda_sensor5.wv3' if package == 'panda_tune30' else 'panda_sensor5.wv3_continue'
        return package, ['python', '-m', module, '--server', server, '--package', package,
                         '--mode', 'finish-current', '--target-run', target]
    raise ValueError('Unknown job kind')


def prepare_costs(server, kind):
    if kind not in ('legacy', 'refine'): return
    package = 'panda_s1loss9' if kind == 'legacy' else 'panda_sensor_refine'
    cmd, _, _ = command(server, package)
    subprocess.run(cmd + ['python', '-m', package, 'check', '--server', server], check=True)
    subprocess.run(cmd + ['python', '-m', package + '.cost', '--server', server, '--seconds', '1200'], check=True)


def launch_job(server, kind, *, target=None):
    from directshift48.__main__ import require_idle_gpu
    directory = launcher_root(ROOT, server)
    with (directory / 'launch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / (kind + '.json')
        if path.exists():
            value = read(path)
            if value.get('target') != target or value.get('kind') != kind:
                raise ValueError('Existing launch receipt differs from requested job')
            return value
        package, args = job_arguments(server, kind, target)
        intent_path = directory / (kind + '_intent.json')
        if intent_path.exists():
            intent = read(intent_path)
            if intent['args'] != args: raise ValueError('Launch intent changed')
        else:
            token = read(directory / 'takeover.json')['token']
            intent = {'name': f'panda-sensor-refine-{kind}-{server}-{token}', 'args': args}
            atomic_json(intent_path, intent)
        cmd, source_name, source = command(server, package, name=intent['name'])
        existing = inspect_optional(intent['name'])
        if existing is None:
            require_idle_gpu('cuda')
            prepare_costs(server, kind)
            require_idle_gpu('cuda')
            result = subprocess.run(cmd + args, capture_output=True, text=True, check=True)
            container_id = result.stdout.strip()
        else:
            if existing['Config']['Cmd'] != args or existing['Image'] != source['Image']:
                raise ValueError('Container name collision; no process changed')
            container_id = existing['Id']
        value = {'server': server, 'kind': kind, 'package': package, 'target': target,
                 'container': intent['name'], 'container_id': container_id,
                 'source_container': source_name, 'image': source['Image'], 'at': utc_now()}
        atomic_json(path, value)
        return value


def wait_job(server, kind):
    value = read(launcher_root(ROOT, server) / (kind + '.json'))
    container = inspect_optional(value['container'])
    if container is None or container['Id'] != value['container_id']:
        raise ValueError('Controlled container identity changed')
    state = container['State']
    if state['Running']: return False
    if state.get('OOMKilled') or state.get('Status') != 'exited' or state.get('ExitCode') != 0:
        raise ValueError('Controlled job stopped unexpectedly: ' + value['container'])
    return True


def ensure_publisher(server, package):
    if package not in ('panda_s1loss9', 'panda_sensor_refine'): return
    subprocess.run([sys.executable, '-m', package + '.best_publication.restart', '--server', server],
                   cwd=ROOT, check=True)
