"""Finish started sensor work, replace unstarted cases, then resume unchanged WV3."""
import argparse
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from directshift48.reporting import atomic_json, utc_now
from panda_sensor_refine import queue as old_queue
from panda_sensor_refine.orchestration import authenticate_stop
from . import launch
from .orchestration import (ROOT, POLICY, SENSOR_PACKAGES, WV3_PACKAGES, launcher_root,
                            read, sha, registered, state_of, campaign_root, boundary_configs,
                            write_predecessor, predecessor_receipt)

POLL_SECONDS = 15


def controller_pin():
    files = sorted(p for p in (ROOT / 'panda_sensor_focus').rglob('*.py') if 'tests' not in p.parts)
    files += sorted((ROOT / 'scripts').glob('*panda_sensor_focus*.sh'))
    return {**old_queue.controller_pin(), **{str(p.relative_to(ROOT)): sha(p) for p in files}}


def watcher_matches(argv, cwd, server):
    return (old_queue.watcher_matches(argv, cwd, server) or
            (Path(cwd).resolve() == ROOT.resolve() and old_queue.option(argv, '--server') == server
             and old_queue.option(argv, '-m') == 'panda_sensor_refine.queue'))


def old_watchers(server):
    result = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            identity = old_queue.process_identity(path)
            if watcher_matches(identity['argv'], identity['cwd'], server):
                result.append(identity)
        except (OSError, UnicodeError):
            continue
    return result


def stop_watchers(server, directory):
    found = old_watchers(server)
    path = directory / 'held_watchers.json'
    value = read(path) if path.exists() else {'attempts': []}
    value['attempts'].append({'at': utc_now(), 'processes': found})
    atomic_json(path, value)
    for item in found:
        try:
            if old_queue.process_identity(Path('/proc') / str(item['pid'])) != item:
                raise ValueError('Old watcher generation changed; no signal sent')
            os.kill(item['pid'], signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError):
            pass


def managed_container(container, server):
    if old_queue.managed_container(container, server):
        return True
    argv = container['Config'].get('Cmd') or []
    checkout = any(m.get('Type') == 'bind' and m.get('Destination') == '/workspace'
                   and Path(m['Source']).resolve() == ROOT.resolve() for m in container.get('Mounts', []))
    return (checkout and old_queue.option(argv, '--server') == server
            and old_queue.option(argv, '-m') in ('panda_sensor_refine', 'panda_sensor_refine.orchestration'))


def running_containers(server):
    import json
    ids = subprocess.run(['docker', 'ps', '-q'], capture_output=True, text=True, check=True).stdout.split()
    if not ids:
        return []
    rows = json.loads(subprocess.run(['docker', 'inspect', *ids], capture_output=True, text=True, check=True).stdout)
    return [r for r in rows if managed_container(r, server)]


def authenticate_container(saved, current):
    if (current is None or current['Id'] != saved['id'] or current['Image'] != saved['image']
            or current['Config']['Cmd'] != saved['argv']):
        raise ValueError('Controlled trainer identity changed; no signal sent')
    return current['State']


def inventory(server):
    result = {}
    active_packages = []
    for package in (*SENSOR_PACKAGES, *WV3_PACKAGES):
        state = state_of(ROOT, package, server)
        if state is None:
            continue
        active = boundary_configs(package, server, state)
        if active:
            active_packages.append(package)
        result[package] = {'status': state['status'],
                           'runs': [{k: r.get(k) for k in ('run_id', 'case', 'replica', 'phase', 'status', 'step')}
                                    for r in state.get('runs', [])],
                           'boundary_ids': [c['run_id'] for c in active]}
    if len(active_packages) > 1:
        raise ValueError('Multiple campaigns have incomplete started runs')
    return result


def natural_refine_tail(server, snapshot, containers):
    if server != 's1' or len(containers) != 1:
        return False
    if old_queue.option(containers[0]['Config'].get('Cmd') or [], '-m') != 'panda_sensor_refine':
        return False
    state = snapshot.get('panda_sensor_refine', {})
    rows = {r['run_id']: r for r in state.get('runs', [])}
    terminal = {'COMPLETED', 'EARLY_STOPPED_HQNR', 'FAILED'}
    pending = [c for c in registered('panda_sensor_refine', server)
               if c['run_id'] not in rows or rows[c['run_id']]['status'] not in terminal]
    return bool(pending) and all(c['case'] in ('S1_A100_E030_R3', 'QB_A100_E030_R3') for c in pending)


def capture(server):
    snapshot = inventory(server)
    containers = running_containers(server)
    if len(containers) > 1:
        raise ValueError('Multiple managed trainers; no process changed')
    receipts = {}
    for package in WV3_PACKAGES:
        path = ROOT / 'work_dir/panda_sensor5_launchers' / server / f"wv3_{package.removeprefix('panda_')}_resume-all.json"
        if path.exists():
            receipts[str(path.relative_to(ROOT))] = {'value': read(path), 'sha256': sha(path)}
    return {'schema': 'SENSOR_FOCUS_TAKEOVER_V1', 'policy': POLICY, 'server': server,
            'token': str(time.time_ns()), 'at': utc_now(), 'controller_pin': controller_pin(),
            'inventory_before': snapshot, 'wv3_resume_receipts': receipts,
            'natural_refine_tail': natural_refine_tail(server, snapshot, containers),
            'containers': [{'name': c['Name'].lstrip('/'), 'id': c['Id'], 'image': c['Image'],
                            'argv': c['Config']['Cmd']} for c in containers], 'wv3_unchanged': True}


def prepare(server, directory):
    path = directory / 'takeover.json'
    if path.exists():
        value = read(path)
        if value['policy'] != POLICY or value['controller_pin'] != controller_pin():
            raise ValueError('Existing focus request has different source/policy')
        return value
    value = capture(server)
    atomic_json(path, value)
    return value


def finalize(server, directory, proposed):
    if proposed.get('finalized_under_old_watcher_locks'):
        return proposed
    if (directory / 'chain.json').exists():
        raise ValueError('Unfinalized takeover has previous mutations')
    current = capture(server)
    if current['controller_pin'] != proposed['controller_pin']:
        raise ValueError('Source changed before acquiring watcher locks')
    atomic_json(directory / 'preliminary_takeover.json', proposed)
    current.update(token=proposed['token'], finalized_under_old_watcher_locks=True)
    atomic_json(directory / 'takeover.json', current)
    return current


def safe_boundary_plan(server, directory):
    entries = []
    for package in (*SENSOR_PACKAGES, *WV3_PACKAGES):
        state = state_of(ROOT, package, server)
        if state is None:
            continue
        chosen = boundary_configs(package, server, state)
        rows = {r['run_id']: r for r in state.get('runs', [])}
        for cfg in chosen:
            row = rows.get(cfg['run_id'])
            if row and row['status'] == 'RUNNING':
                raise ValueError('Exited trainer left a RUNNING state')
            if row and row['status'] == 'STOPPED':
                authenticate_stop(row, cfg)
        if chosen:
            entries.append({'package': package, 'run_ids': [c['run_id'] for c in chosen]})
    value = {'server': server, 'policy': POLICY, 'entries': entries}
    path = directory / 'boundary_plan.json'
    if path.exists() and read(path) != value:
        raise ValueError('Original stopped boundary plan changed')
    atomic_json(path, value)
    return value


def focus_complete(server):
    from . import plan
    from .orchestration import final_evidence
    from panda_sensor_refine.early_stop import verify
    root = ROOT / 'work_dir/panda_sensor_focus' / plan.CAMPAIGN_ID / server / plan.dataset_for_server(server)
    state = read(root / 'campaign_state.json')
    cfgs = [plan.resolve_case(server, c, r) for g in plan.groups(server) for c, r in g['runs']]
    rows = {r['run_id']: r for r in state['runs']}
    if (not state['status'].startswith('FINISHED_') or len(rows) != len(state['runs'])
            or set(rows) != {c['run_id'] for c in cfgs}):
        raise ValueError('Focus queue lacks exact terminal run accounting')
    for cfg in cfgs:
        row = rows[cfg['run_id']]
        if row['status'] == 'COMPLETED' and row['step'] == cfg['planned_horizon']:
            final_evidence(root, cfg)
        elif row['status'] == 'EARLY_STOPPED_HQNR':
            verify(row, cfg)
        else:
            raise ValueError('Unexpected unfinished focus run')
    return True


def validate_boundary_completion(server, directory, entry, proof):
    from donor30.plan import canonical_sha256
    expected_statuses = {'COMPLETED', 'EARLY_STOPPED_HQNR', 'FAILED_SUPPORT',
                         'FAILED_PREFLIGHT_SUPPORT', 'HELD_FAILED_DONOR'}
    admission = campaign_root(ROOT, entry['package'], server) / 'admission.json'
    if (proof.get('schema') != 'SENSOR_FOCUS_BOUNDARY_COMPLETION_V1'
            or proof.get('server') != server or proof.get('package') != entry['package']
            or proof.get('run_ids') != entry['run_ids'] or proof.get('status') != 'PAUSED_FOR_SENSOR_FOCUS'
            or set(proof.get('outcomes', {})) != set(entry['run_ids'])
            or any(x.get('status') not in expected_statuses for x in proof['outcomes'].values())
            or proof.get('controller_pin_sha256') != canonical_sha256(controller_pin())
            or proof.get('takeover_sha256') != sha(directory / 'takeover.json')
            or proof.get('boundary_plan_sha256') != sha(directory / 'boundary_plan.json')
            or proof.get('original_admission_sha256') != sha(admission)
            or proof.get('original_training_registry_unchanged') is not True):
        raise ValueError('Boundary completion identity/evidence differs from selected IDs')


def advance(server, directory, chain, request):
    stage = chain['stage']
    if stage == 'STOP_OLD':
        for saved in request['containers']:
            state = authenticate_container(saved, launch.inspect_optional(saved['name']))
            if state['Running']:
                subprocess.run(['docker', 'kill', '--signal=SIGTERM', saved['id']], check=True)
        chain['stage'] = 'WAIT_SAFE_STOP'
    elif stage in ('WAIT_SAFE_STOP', 'WAIT_RETAINED_REFINE'):
        for saved in request['containers']:
            state = authenticate_container(saved, launch.inspect_optional(saved['name']))
            if state['Running']:
                return
            allowed = (0,) if stage == 'WAIT_RETAINED_REFINE' else (0, 1)
            if state.get('OOMKilled') or state.get('Status') != 'exited' or state.get('ExitCode') not in allowed:
                raise ValueError('Previous trainer did not finish/stop safely')
        plan = safe_boundary_plan(server, directory)
        chain.update(stage='BOUNDARIES', boundary_index=0, boundaries=plan['entries'])
    elif stage == 'BOUNDARIES':
        i = chain['boundary_index']
        if i == len(chain['boundaries']):
            write_predecessor(ROOT, server)
            chain['stage'] = 'START_FOCUS'
        else:
            entry = chain['boundaries'][i]
            kind = f'boundary_{i:02d}'
            launch.launch_job(server, kind, entry)
            if launch.wait_job(server, kind):
                if entry['package'] in WV3_PACKAGES:
                    from panda_sensor5.transition import completed_run
                    completed_run(ROOT, entry['package'], server, entry['run_ids'][0])
                else:
                    proof = read(directory / (kind + '_complete.json'))
                    validate_boundary_completion(server, directory, entry, proof)
                chain['boundary_index'] += 1
    elif stage == 'START_FOCUS':
        predecessor_receipt(ROOT, server)
        launch.launch_job(server, 'focus')
        chain['stage'] = 'WAIT_FOCUS'
    elif stage == 'WAIT_FOCUS':
        if launch.wait_job(server, 'focus') and focus_complete(server):
            old_queue.reconcile_wv3_receipts(server, directory, request)
            chain['stage'] = 'RESUME_TUNE30'
    elif stage in ('RESUME_TUNE30', 'RESUME_HARD36'):
        from panda_sensor5.queue import advance as original
        original(server, ROOT / 'work_dir/panda_sensor5_launchers' / server, chain)
    elif stage != 'COMPLETED':
        raise ValueError('Unknown focus queue state: ' + str(stage))


def watch(server, directory):
    request = read(directory / 'takeover.json')
    if request['controller_pin'] != controller_pin():
        raise ValueError('Frozen focus controller changed')
    with ExitStack() as stack:
        own = stack.enter_context((directory / 'watcher.lock').open('a+'))
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stop_watchers(server, directory)
        from panda_s1loss9.queue import acquire_old_watcher_lock
        for package in ('panda_sensor_refine', 'panda_s1loss9', 'panda_sensor5', 'panda_hard36'):
            path = ROOT / 'work_dir' / (package + '_launchers') / server
            path.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context((path / 'watcher.lock').open('a+'))
            acquire_old_watcher_lock(lock)
        request = finalize(server, directory, request)
        path = directory / 'chain.json'
        chain = read(path) if path.exists() else {'stage': 'WAIT_RETAINED_REFINE' if request['natural_refine_tail'] else 'STOP_OLD', 'policy': POLICY}
        try:
            while chain['stage'] != 'COMPLETED':
                if request['controller_pin'] != controller_pin():
                    raise ValueError('Queued source changed')
                advance(server, directory, chain, request)
                if (directory / 'focus.json').exists() and not chain.get('publisher_ready'):
                    try:
                        launch.ensure_publisher(server)
                        chain['publisher_ready'] = True
                        chain.pop('publisher_error', None)
                    except Exception as exc:
                        chain['publisher_error'] = repr(exc)
                chain.update(server=server, pid=os.getpid(), at=utc_now(), status=chain['stage'])
                atomic_json(path, chain)
                atomic_json(directory / 'status.json', chain)
                if chain['stage'] != 'COMPLETED':
                    time.sleep(POLL_SECONDS)
            return 0
        except BaseException as exc:
            atomic_json(directory / 'status.json', {**chain, 'status': 'BLOCKED', 'error': repr(exc), 'at': utc_now()})
            raise


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', choices=('s1', 's3'), required=True)
    parser.add_argument('--enqueue', action='store_true')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args(argv)
    if args.enqueue == args.watch:
        parser.error('Exactly one of --enqueue/--watch is required')
    if (ROOT / 'gspread/server.txt').read_text().strip() != args.server:
        raise ValueError('Local server marker mismatch')
    directory = launcher_root(ROOT, args.server)
    directory.mkdir(parents=True, exist_ok=True)
    if args.watch:
        return watch(args.server, directory)
    with (directory / 'enqueue.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (directory / 'watcher.lock').open('a+') as watcher:
            fcntl.flock(watcher, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from .donor import validate_source
        validate_source(args.server, root=ROOT)
        prepare(args.server, directory)
        log = directory / 'controller.log'
        with log.open('ab') as output:
            child = subprocess.Popen([sys.executable, '-u', '-m', 'panda_sensor_focus.queue',
                                      '--server', args.server, '--watch'], cwd=ROOT,
                                     stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        value = {'policy': POLICY, 'server': args.server, 'pid': child.pid, 'log': str(log), 'at': utc_now()}
        atomic_json(directory / 'enqueue.json', value)
        print(json.dumps(value), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
