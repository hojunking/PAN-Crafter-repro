"""Retained sensor screen -> focused R4/tuning -> unchanged deferred WV3."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from directshift48.reporting import atomic_json, utc_now
from . import launch
from .orchestration import (ROOT, POLICY, RETAINED_PAIRS, SUPPORT_ERROR,
    launcher_root, legacy_root, legacy_configs, read, sha, local,
    authenticate_stop, support_failure, final_evidence, predecessor_receipt)

POLL_SECONDS = 15
PACKAGES = ('panda_sensor5', 'panda_s1loss9', 'panda_tune30', 'panda_hard36')


def controller_pin():
    from panda_s1loss9.queue import controller_pin as original
    paths = sorted((ROOT / 'panda_sensor_refine').rglob('*.py'))
    paths += sorted((ROOT / 'scripts').glob('*panda_sensor_refine*.sh'))
    return {**original(), **{str(p.relative_to(ROOT)): sha(p) for p in paths}}


def state_path(package, server):
    plan = importlib.import_module(package + '.plan')
    dataset = plan.dataset_for_server(server) if package in ('panda_sensor5', 'panda_s1loss9') else 'WV3'
    return ROOT / 'work_dir' / package / plan.CAMPAIGN_ID / server / dataset / 'campaign_state.json'


def state_of(package, server):
    path = state_path(package, server)
    return read(path) if path.exists() else None


def configs_of(package, server):
    plan = importlib.import_module(package + '.plan')
    return [plan.resolve_case(server, c, r) for g in plan.groups(server) for c, r in g['runs']]


def option(argv, name):
    return argv[argv.index(name) + 1] if argv.count(name) == 1 and argv.index(name) + 1 < len(argv) else None


def watcher_matches(argv, cwd, server):
    if Path(cwd).resolve() != ROOT.resolve() or option(argv, '--server') != server:
        return False
    if option(argv, '-m') in ('panda_sensor5.queue', 'panda_s1loss9.queue', 'panda_hard36.queue'):
        return True
    return ('watch' in argv and any(Path(a).name == 's1loss9_recovery.py' for a in argv))


def process_identity(path):
    return {'pid': int(path.name), 'argv': (path / 'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
            'cwd': str((path / 'cwd').resolve()),
            'generation': (path / 'stat').read_text().rsplit(')', 1)[1].split()[19]}


def old_watchers(server):
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit(): continue
        try:
            if path.stat().st_uid != os.getuid(): continue
            value = process_identity(path)
            if watcher_matches(value['argv'], value['cwd'], server): found.append(value)
        except (OSError, UnicodeError): continue
    return found


def stop_watchers(server, directory):
    path = directory / 'held_watchers.json'
    history = read(path) if path.exists() else {'policy': POLICY, 'attempts': []}
    found = old_watchers(server)
    # Persist exact process generation before any signal, including crash recovery.
    history['attempts'].append({'at': utc_now(), 'processes': found})
    atomic_json(path, history)
    for value in found:
        process = Path('/proc') / str(value['pid'])
        try:
            if process_identity(process) != value:
                raise ValueError('Old watcher identity changed; no signal sent')
            os.kill(value['pid'], signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError): pass
    deadline = time.monotonic() + 20
    while old_watchers(server):
        if time.monotonic() >= deadline: raise ValueError('Old watcher did not stop; trainers untouched')
        time.sleep(.1)


def managed_container(container, server):
    argv = container['Config'].get('Cmd') or []
    checkout = any(m.get('Type') == 'bind' and m.get('Destination') == '/workspace'
                   and Path(m['Source']).resolve() == ROOT.resolve() for m in container.get('Mounts', []))
    if not checkout or option(argv, '--server') != server: return False
    module = option(argv, '-m')
    if module in (*PACKAGES, 'panda_s1loss9.orchestration', 'panda_sensor5.wv3', 'panda_sensor5.wv3_continue'):
        return True
    return 'run' in argv and any(Path(a).name == 's1loss9_recovery.py' for a in argv)


def running_containers(server):
    ids = subprocess.run(['docker', 'ps', '-q'], capture_output=True, text=True, check=True).stdout.split()
    if not ids: return []
    values = json.loads(subprocess.run(['docker', 'inspect', *ids], capture_output=True, text=True, check=True).stdout)
    return [v for v in values if managed_container(v, server)]


def validate_inventory(server):
    snapshot, boundaries = {}, []
    for package in PACKAGES:
        state = state_of(package, server)
        if state is None:
            if package == 'panda_sensor5': raise ValueError('Local RGB/L4 SENSOR5 campaign is required')
            continue
        cfgs = {c['run_id']: c for c in configs_of(package, server)}
        rows = state.get('runs', [])
        if len({r['run_id'] for r in rows}) != len(rows) or any(r['run_id'] not in cfgs for r in rows):
            raise ValueError('Unknown/duplicate existing campaign run')
        active = [r for r in rows if r['status'] in ('RUNNING', 'STOPPED')]
        if len(active) > 1 or any(r.get('poisoned') or r.get('stop_checkpoint_error') for r in active):
            raise ValueError('Current optimizer states require inspection')
        for row in rows:
            if row['status'] == 'FAILED':
                if package != 'panda_s1loss9': raise ValueError('Unrelated failed campaign requires inspection')
                support_failure(row, cfgs[row['run_id']])
        if state.get('status') == 'FAILED' and not any(r['status'] == 'FAILED' for r in rows):
            raise ValueError('Unclassified controller failure requires inspection')
        snapshot[package] = state
        if package in ('panda_tune30', 'panda_hard36') and active:
            boundaries.append({'kind': 'tune-boundary' if package == 'panda_tune30' else 'hard-boundary',
                               'package': package, 'target': active[0]['run_id']})
    from panda_s1loss9.orchestration import initial_complete
    sensor = snapshot['panda_sensor5']
    active_sensor = [r for r in sensor['runs'] if r['status'] in ('RUNNING', 'STOPPED')]
    if active_sensor or not initial_complete(sensor, server):
        boundaries.append({'kind': 'sensor-boundary', 'package': 'panda_sensor5',
                           'target': active_sensor[0]['run_id'] if active_sensor else None})
    return snapshot, boundaries


def capture_takeover(server):
    snapshot, boundaries = validate_inventory(server)
    containers = running_containers(server)
    if len(containers) > 1: raise ValueError('Multiple managed trainers; no process changed')
    receipts = {}
    old = ROOT / 'work_dir/panda_sensor5_launchers' / server
    for package in ('panda_tune30', 'panda_hard36'):
        p = old / f"wv3_{package.removeprefix('panda_')}_resume-all.json"
        if p.exists(): receipts[str(p.relative_to(ROOT))] = {'value': read(p), 'sha256': sha(p)}
    value = {'schema': 'SENSOR_REFINE_TAKEOVER_V1', 'policy': POLICY, 'server': server,
             'token': str(time.time_ns()), 'at': utc_now(), 'controller_pin': controller_pin(),
             'boundaries': boundaries, 'state_before': snapshot, 'wv3_resume_receipts': receipts,
             'containers': [{'name': c['Name'].lstrip('/'), 'id': c['Id'], 'image': c['Image'],
                             'argv': c['Config']['Cmd']} for c in containers],
             'retained_pairs': list(RETAINED_PAIRS[server]), 'wv3_unchanged': True}
    return value


def prepare_takeover(server, directory):
    path = directory / 'takeover.json'
    if path.exists():
        value = read(path)
        if value['policy'] != POLICY or value['controller_pin'] != controller_pin():
            raise ValueError('Existing refinement request has different code/policy')
        return value
    value = capture_takeover(server)
    atomic_json(path, value)
    return value


def finalize_takeover(server, directory, proposed):
    """Refresh the preliminary read only after every old watcher lock is held."""
    if proposed.get('finalized_under_old_watcher_locks'):
        return proposed
    if (directory / 'chain.json').exists():
        raise ValueError('Unfinalized request already has mutations; inspect explicitly')
    current = capture_takeover(server)
    if current['controller_pin'] != proposed['controller_pin']:
        raise ValueError('Source changed while acquiring old watcher locks')
    atomic_json(directory / 'takeover_before_watcher_locks.json', proposed)
    current.update(token=proposed['token'], finalized_under_old_watcher_locks=True)
    atomic_json(directory / 'takeover.json', current)
    return current


def stop_trainers(request):
    for saved in request['containers']:
        current = launch.inspect_optional(saved['name'])
        if (current is None or current['Id'] != saved['id'] or current['Image'] != saved['image']
                or current['Config']['Cmd'] != saved['argv']):
            raise ValueError('Trainer identity changed; no signal sent')
        if current['State']['Running']:
            subprocess.run(['docker', 'kill', '--signal=SIGTERM', saved['id']], check=True)


def stopped_and_verified(server, request):
    for saved in request['containers']:
        current = launch.inspect_optional(saved['name'])
        if current is None or current['Id'] != saved['id']:
            raise ValueError('Controlled stop container disappeared')
        state = current['State']
        if state['Running']: return False
        if state.get('Status') != 'exited' or state.get('OOMKilled') or state.get('ExitCode') not in (0, 1):
            raise ValueError('Controlled trainer did not stop safely')
    for package in PACKAGES:
        state = state_of(package, server)
        if state is None: continue
        cfgs = {c['run_id']: c for c in configs_of(package, server)}
        for row in state.get('runs', []):
            if row['status'] == 'RUNNING':
                raise ValueError('Exited trainer left a RUNNING state without exact stop proof')
            if row['status'] == 'STOPPED': authenticate_stop(row, cfgs[row['run_id']])
            if row['status'] == 'FAILED':
                if package != 'panda_s1loss9': raise ValueError('New failure during safe stop')
                support_failure(row, cfgs[row['run_id']])
    return True


def retire_sensor_pending(server, directory):
    state = state_of('panda_sensor5', server)
    rows = {r['run_id']: r for r in state.get('runs', [])}
    from panda_s1loss9.orchestration import initial_complete
    if not initial_complete(state, server): raise ValueError('Initial SENSOR5 pair is incomplete')
    completed, skipped = [], []
    root = state_path('panda_sensor5', server).parent
    for cfg in configs_of('panda_sensor5', server):
        row = rows.get(cfg['run_id'])
        if row is None:
            skipped.append({'run_id': cfg['run_id'], 'reason': 'USER_REPLACED_OLD_DONOR_TUNING'})
        elif row['status'] == 'COMPLETED' and row['step'] == cfg['planned_horizon']:
            completed.append(final_evidence(root, cfg))
        else: raise ValueError('A started SENSOR5 run cannot be retired')
    value = {'policy': POLICY, 'server': server, 'completed': completed,
             'excluded_unstarted': skipped, 'original_campaign_not_marked_complete': True}
    path = directory / 'sensor5_retirement.json'
    if path.exists() and read(path) != value: raise ValueError('SENSOR5 retirement receipt changed')
    atomic_json(path, value)


def verify_core_failure(root, cfg, record, rows, state):
    from . import plan, runtime
    if (record.get('run_id') != cfg['run_id'] or record.get('case') != cfg['case']
            or not record.get('error', '').startswith('SUPPORT_VIOLATION:')):
        raise ValueError('Unknown refinement failure type or identity')
    key = cfg['run_id']
    if record.get('evidence_kind') == 'FAILED_PREFLIGHT_NO_PRODUCTION_UPDATES':
        path = local(record['receipt_path'])
        expected = root / 'preflight' / (cfg['case'] + '_' + cfg['replica'] + '_support_failure.json')
        if (key in rows or (root / 'runs' / key).exists() or path.resolve() != expected.resolve()
                or path.is_symlink() or sha(path) != record.get('receipt_sha256')):
            raise ValueError('Preflight failure has production state or changed receipt')
        proof = read(path)
        if (proof.get('schema') != 'SENSOR_REFINE_PREFLIGHT_SUPPORT_FAILURE_V1'
                or proof.get('evidence_kind') != record['evidence_kind']
                or plan.canonical_sha256(proof.get('requested_config')) != plan.canonical_sha256(cfg)
                or proof.get('run_id') != key
                or proof.get('error') != record['error'] or proof.get('step') != 0
                or record.get('step') != 0 or proof.get('production_run_absent') is not True
                or proof.get('production_updates') != 0 or proof.get('production_weights_started') is not False
                or proof.get('disposable_preflight_only') is not True
                or proof.get('code_pin') != runtime.code_pin()
                or record.get('code_pin_sha256') != plan.canonical_sha256(proof['code_pin'])
                or record.get('config_sha256') != plan.canonical_sha256(cfg)):
            raise ValueError('Preflight failure config/source/evidence authentication failed')
        donor = proof.get('donor')
        if cfg['phase'] == 'stage2':
            selected = state.get('donors', {}).get(cfg['donor_version'])
            if not isinstance(donor, dict) or selected is None:
                raise ValueError('Preflight failure lacks selected donor evidence')
            for field in ('checkpoint_path', 'checkpoint_sha256', 'source_config', 'selector_receipt_path', 'selector_sha256'):
                if donor.get(field) != selected.get(field): raise ValueError('Preflight donor differs from declared source')
            module = 'panda_s1loss9.selection' if cfg['donor_version'] == 'S1_A200_E030' else 'panda_sensor_refine.selection'
            selector = importlib.import_module(module).validate_receipt(read(local(donor['selector_receipt_path'])))
            if (sha(local(donor['checkpoint_path'])) != donor['checkpoint_sha256']
                    or selector['receipt_sha256'] != donor['selector_sha256']
                    or record.get('donor_checkpoint_sha256') != donor['checkpoint_sha256']):
                raise ValueError('Preflight source checkpoint/selection changed')
        elif donor is not None or record.get('donor_checkpoint_sha256') is not None:
            raise ValueError('Fresh donor preflight incorrectly records an external source')
    else:
        row = rows.get(key)
        if (row is None or row.get('status') != 'FAILED' or row.get('poisoned') is not False
                or row.get('step', 0) <= 0 or row.get('error') != repr(ValueError(record['error']))):
            raise ValueError('Production support failure lacks intact failed run')
        proof = authenticate_stop(row, cfg)
        if record.get('checkpoint_sha256') != proof['checkpoint_sha256'] or record.get('step') != proof['step']:
            raise ValueError('Refinement failure checkpoint proof changed')


def core_complete(server):
    from . import early_stop
    from . import plan
    root = ROOT / 'work_dir/panda_sensor_refine' / plan.CAMPAIGN_ID / server / plan.dataset_for_server(server)
    state = read(root / 'campaign_state.json')
    configs = [plan.resolve_case(server, c, r) for g in plan.groups(server) for c, r in g['runs']]
    rows = {r['run_id']: r for r in state['runs']}
    failures, held = state.get('isolated_support_failures', {}), state.get('held_runs', {})
    stopped = state.get('early_stopped_runs', {})
    terminal = ('FINISHED_WITH_EXPLICIT_EARLY_STOPS' if stopped else
                'FINISHED_WITH_EXPLICIT_FAILURES' if failures or held else 'FINISHED_REGISTERED_QUEUE')
    if state.get('status') != terminal:
        raise ValueError('Refinement campaign lacks explicit registered accounting')
    expected = {c['run_id']: c for c in configs}
    if (len(rows) != len(state['runs']) or set(rows) & set(held) or set(failures) & set(held)
            or not set(stopped) <= set(rows) or set(stopped) & (set(failures) | set(held))
            or set(rows) | set(held) | set(failures) != set(expected)):
        raise ValueError('Refinement completion registry differs')
    external = predecessor_receipt(ROOT, server)['external_donor']
    failed_donors = {expected[key]['case'] for key in failures}
    for cfg in configs:
        key = cfg['run_id']
        if key in held:
            reason = held[key]
            authorized_donor = cfg['donor_version'] in failed_donors or (
                cfg['donor_version'] == 'S1_A200_E030' and external['status'] == 'FAILED_SUPPORT')
            if (cfg['phase'] != 'stage2' or reason.get('status') != 'HELD_FAILED_DONOR'
                    or reason.get('donor_case') != cfg['donor_version'] or not authorized_donor):
                raise ValueError('Held refinement run lacks a failed donor proof')
        elif key in failures:
            verify_core_failure(root, cfg, failures[key], rows, state)
        elif key in stopped:
            if early_stop.verify(rows[key], cfg) != stopped[key]:
                raise ValueError('Refinement early-stop receipt differs from registry')
        else:
            row = rows[key]
            if row['status'] != 'COMPLETED' or row['step'] != cfg['planned_horizon']:
                raise ValueError('Refinement run incomplete')
            final_evidence(root, cfg)
    return True


def reconcile_wv3_receipts(server, directory, request):
    """Archive only authenticated interrupted control receipts, never WV3 results."""
    for relative, saved in request['wv3_resume_receipts'].items():
        path = ROOT / relative
        backup = directory / ('prior_' + path.name)
        if backup.exists():
            if read(backup) != saved: raise ValueError('WV3 receipt archive changed')
            if not path.exists(): continue
            # A newly launched unchanged successor is already installed.
            if sha(path) != saved['sha256']: continue
        package = saved['value']['package']
        state = state_of(package, server)
        if state and state.get('status') == 'FINISHED_REGISTERED_QUEUE': continue
        if not path.exists() or sha(path) != saved['sha256']:
            raise ValueError('Original WV3 resume receipt changed unexpectedly')
        container = launch.inspect_optional(saved['value']['container'])
        if container is None or container['Id'] != saved['value']['container_id'] or container['State']['Running']:
            raise ValueError('Previous WV3 continuation is not safely inactive')
        if state is None or state.get('status') not in ('PAUSED_FOR_SENSOR5', 'STOPPED_SAFE', 'FINISHED_WITH_BUDGET_SKIPS'):
            raise ValueError('WV3 state does not authenticate an interrupted continuation')
        atomic_json(backup, saved)
        path.unlink()


def advance(server, directory, chain, request):
    stage = chain['stage']
    if stage == 'STOP_OLD':
        stop_trainers(request)
        chain['stage'] = 'WAIT_SAFE_STOP'
    elif stage == 'WAIT_SAFE_STOP':
        if stopped_and_verified(server, request):
            # A trainer can finish one run and begin another between the final
            # inventory and SIGTERM. Resume the authenticated actually stopped
            # run, retaining the original snapshot as evidence of the boundary.
            snapshot, boundaries = validate_inventory(server)
            proof = {'policy': POLICY, 'server': server, 'boundaries': boundaries,
                     'active': {p: [r for r in s.get('runs', []) if r['status'] == 'STOPPED']
                                for p, s in snapshot.items()}}
            atomic_json(directory / 'stopped_boundaries.json', proof)
            chain.update(stage='BOUNDARIES', boundary_index=0, boundaries=boundaries)
    elif stage == 'BOUNDARIES':
        i = chain['boundary_index']
        boundaries = chain.get('boundaries', request['boundaries'])
        if i >= len(boundaries):
            retire_sensor_pending(server, directory)
            chain['stage'] = 'START_LEGACY'
        else:
            item = boundaries[i]
            launch.launch_job(server, item['kind'], target=item['target'])
            if launch.wait_job(server, item['kind']):
                if item['target']:
                    from panda_sensor5.transition import completed_run
                    completed_run(ROOT, item['package'], server, item['target'])
                chain['boundary_index'] += 1
    elif stage == 'START_LEGACY':
        launch.launch_job(server, 'legacy')
        chain['stage'] = 'WAIT_LEGACY'
    elif stage == 'WAIT_LEGACY':
        if launch.wait_job(server, 'legacy'):
            predecessor_receipt(ROOT, server)
            chain['stage'] = 'START_REFINE'
    elif stage == 'START_REFINE':
        predecessor_receipt(ROOT, server)
        launch.launch_job(server, 'refine')
        chain['stage'] = 'WAIT_REFINE'
    elif stage == 'WAIT_REFINE':
        if launch.wait_job(server, 'refine') and core_complete(server):
            reconcile_wv3_receipts(server, directory, request)
            chain['stage'] = 'RESUME_TUNE30'
    elif stage in ('RESUME_TUNE30', 'RESUME_HARD36'):
        from panda_sensor5.queue import advance as original
        original(server, ROOT / 'work_dir/panda_sensor5_launchers' / server, chain)
    elif stage != 'COMPLETED':
        raise ValueError('Unknown refinement queue stage: ' + str(stage))


def watch(server, directory):
    request = read(directory / 'takeover.json')
    if request['controller_pin'] != controller_pin(): raise ValueError('Frozen queue source changed')
    with ExitStack() as stack:
        own = stack.enter_context((directory / 'watcher.lock').open('a+'))
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stop_watchers(server, directory)
        from panda_s1loss9.queue import acquire_old_watcher_lock
        for package in ('panda_s1loss9', 'panda_sensor5', 'panda_hard36'):
            p = ROOT / 'work_dir' / (package + '_launchers') / server
            p.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context((p / 'watcher.lock').open('a+'))
            acquire_old_watcher_lock(lock)
        request = finalize_takeover(server, directory, request)
        path = directory / 'chain.json'
        chain = read(path) if path.exists() else {'stage': 'STOP_OLD', 'policy': POLICY}
        try:
            while chain['stage'] != 'COMPLETED':
                if request['controller_pin'] != controller_pin(): raise ValueError('Queued source changed')
                advance(server, directory, chain, request)
                # Publishing is independently retriable; never kill healthy training
                # or launch twice because Google Sheets is temporarily unavailable.
                for kind, package in (('legacy', 'panda_s1loss9'), ('refine', 'panda_sensor_refine')):
                    if (directory / (kind + '.json')).exists() and not chain.get(kind + '_publisher_ready'):
                        try:
                            launch.ensure_publisher(server, package)
                            chain[kind + '_publisher_ready'] = True
                            chain.pop(kind + '_publisher_error', None)
                        except Exception as exc:
                            chain[kind + '_publisher_error'] = repr(exc)
                chain.update(server=server, pid=os.getpid(), at=utc_now(), status=chain['stage'])
                atomic_json(path, chain); atomic_json(directory / 'status.json', chain)
                if chain['stage'] != 'COMPLETED': time.sleep(POLL_SECONDS)
            return 0
        except BaseException as exc:
            atomic_json(directory / 'status.json', {**chain, 'status': 'BLOCKED', 'error': repr(exc), 'at': utc_now()})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True, choices=('s1', 's3'))
    parser.add_argument('--enqueue', action='store_true')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args(argv)
    if args.enqueue == args.watch: parser.error('Exactly one of --enqueue or --watch is required')
    if (ROOT / 'gspread/server.txt').read_text().strip() != args.server: raise ValueError('Server marker mismatch')
    directory = launcher_root(ROOT, args.server); directory.mkdir(parents=True, exist_ok=True)
    if args.watch:
        try:
            return watch(args.server, directory)
        except BaseException as exc:
            path = directory / 'status.json'
            previous = read(path) if path.exists() else {'stage': 'ACQUIRE_OLD_WATCHER_LOCKS'}
            atomic_json(path, {**previous, 'status': 'BLOCKED', 'error': repr(exc), 'at': utc_now()})
            raise
    with (directory / 'enqueue.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (directory / 'watcher.lock').open('a+') as watcher:
            fcntl.flock(watcher, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request = prepare_takeover(args.server, directory)
        log = directory / 'controller.log'
        with log.open('ab') as output:
            child = subprocess.Popen([sys.executable, '-u', '-m', 'panda_sensor_refine.queue',
                    '--server', args.server, '--watch'], cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        value = {'policy': POLICY, 'server': args.server, 'pid': child.pid, 'log': str(log), 'at': utc_now()}
        atomic_json(directory / 'enqueue.json', value)
        print(json.dumps(value), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
