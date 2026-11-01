"""Finish the local current QB40 case, replace its waiting queue, then resume WV3 on s1."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from directshift48.reporting import atomic_json, utc_now
from panda_qb40.priority import process_identity, module_of
from . import registry as G

ROOT = Path(__file__).resolve().parents[1]
POLICY = 'USER_20261101_FINISH_CURRENT_QB40_THEN_A4_E030_THREE_SEEDS'


def read(path):
    return json.loads(Path(path).read_text())


def control(server):
    return ROOT / G.OUTPUT / ('handoff_' + server)


def old_root(server):
    from panda_qb40 import registry as old
    return ROOT / old.OUTPUT / old.CAMPAIGN_ID / server / 'QB'


def processes(module, extra=None):
    found = []
    for path in Path('/proc').glob('[0-9]*'):
        identity = process_identity(path.name)
        if identity is None or module_of(identity) != module:
            continue
        if extra is not None and identity['argv'][identity['argv'].index('-m') + 2:] != extra:
            continue
        if identity['cwd'] != str(ROOT.resolve()) or identity['uid'] != os.getuid():
            raise ValueError('Matching controller belongs to another checkout/user')
        found.append(identity)
    return found


def stop_process(identity):
    current = process_identity(identity['pid'])
    if current is None:
        return
    if current != identity:
        raise ValueError('Controller PID was reused; no signal sent')
    os.kill(identity['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 30
    while process_identity(identity['pid']) == identity:
        if time.monotonic() > deadline:
            raise RuntimeError('Queue controller did not stop')
        time.sleep(.2)


def snapshot(server):
    from panda_qb40 import enqueue as old
    from panda_sensor_refine.launch import inspect_optional
    path = old_root(server) / 'campaign_state.json'
    state = read(path) if path.exists() else {'runs': []}
    active = [r for r in state['runs'] if r['status'] in ('RUNNING', 'STOPPED')]
    if len(active) > 1 or any(r.get('poisoned') for r in active):
        raise ValueError('Expected at most one safe current QB40 run')
    current = inspect_optional(old.container_name(server))
    request = dict(policy=POLICY, server=server, campaign_id=G.CAMPAIGN_ID, at=utc_now(),
        target_run=active[0]['run_id'] if active else None,
        observed_step=active[0]['step'] if active else None,
        observed_campaign_status=state.get('status'), old_container=None)
    if current is not None:
        intent = read(old.control_dir(server) / 'launch_intent.json')
        receipt = read(old.control_dir(server) / 'launch.json')
        old.authenticate_container(server, current, intent, receipt)
        request.update(old_container=receipt, old_intent=intent)
    elif active:
        raise ValueError('Active QB40 run has no authenticated container')
    return request


def stop_old(request):
    from panda_qb40 import enqueue as old
    from panda_sensor_refine.launch import inspect_optional
    from panda_sensor_refine.orchestration import authenticate_stop
    server = request['server']
    for module, extra in [('panda_qb40.priority', ['watch']),
                          ('panda_qb40.enqueue', ['--server', server]),
                          ('panda_qb40.enqueue', ['--server', server, '--wait'])]:
        for identity in processes(module, extra):
            stop_process(identity)
    receipt = request['old_container']
    if receipt is None:
        return dict(status='NO_OLD_CAMPAIGN', at=utc_now())
    current = inspect_optional(receipt['container'])
    old.authenticate_container(server, current, request['old_intent'], receipt)
    if current['State']['Running']:
        state = read(old_root(server) / 'campaign_state.json')
        active = [r for r in state['runs'] if r['status'] in ('RUNNING', 'STOPPED')]
        if [r['run_id'] for r in active] != ([request['target_run']] if request['target_run'] else []):
            raise ValueError('Current case changed before cutover; replan without editing training state')
        subprocess.run(['docker', 'kill', '--signal=SIGTERM', current['Id']], check=True)
    deadline = time.monotonic() + 300
    while True:
        current = inspect_optional(receipt['container'])
        old.authenticate_container(server, current, request['old_intent'], receipt)
        if not current['State']['Running']:
            break
        if time.monotonic() > deadline:
            raise RuntimeError('Waiting for complete checkpoint save; rerun apply')
        time.sleep(2)
    if current['State'].get('OOMKilled'):
        raise ValueError('Old container was OOM-killed')
    state = read(old_root(server) / 'campaign_state.json')
    if request['target_run']:
        row = next(r for r in state['runs'] if r['run_id'] == request['target_run'])
        if row['status'] == 'STOPPED' and state.get('status') == 'STOPPED_SAFE':
            proof = authenticate_stop(row, {'run_id': row['run_id']}, ROOT)
        elif row['status'] == 'COMPLETED':
            from .finish_current import target_config
            from panda_sensor_refine.orchestration import final_evidence
            proof = final_evidence(old_root(server), target_config(server, row['run_id'], state))
        else:
            raise ValueError('Current QB40 case has no authentic safe stop/completion')
    else:
        # A terminal known Stage1 support failure is retained, not retried.
        from types import SimpleNamespace
        from panda_qb40 import failures, plan
        camp = SimpleNamespace(root=old_root(server), server=server, state=state,
                               config=plan.default_config(server))
        failures.synchronize(camp)
        if any(r['status'] not in ('COMPLETED', 'EARLY_STOPPED_HQNR', 'FAILED') for r in state['runs']):
            raise ValueError('Unaccounted legacy trainer state')
        if any(r['status'] == 'FAILED' and r['run_id'] not in state.get('failed_runs', {}) for r in state['runs']):
            raise ValueError('Unrecognized old training failure requires diagnosis')
        proof = dict(status='NO_ACTIVE_CASE', original_status=state.get('status'))
    return dict(at=utc_now(), docker_state=current['State'], proof=proof)


def finish_args(request):
    return ['python', '-m', 'panda_qb_rre.finish_current', '--server', request['server'],
            '--target-run', request['target_run']]


def authenticate_finish(current, intent):
    if current is None:
        raise ValueError('Finish-current container is missing')
    cfg, identity = current['Config'], intent['identity']
    mounts = [{k: m[k] for k in ('Source', 'Destination', 'RW')} for m in current['Mounts']]
    if (current['Image'] != identity['image'] or cfg['Cmd'] != intent['args']
            or cfg['User'] != identity['user'] or cfg['WorkingDir'] != identity['working_dir']
            or sorted(cfg['Env']) != sorted(identity['env'])
            or sorted(mounts, key=lambda m: m['Destination']) != sorted(identity['mounts'], key=lambda m: m['Destination'])
            or (cfg.get('Labels') or {}).get('panda.launch-token') != intent['token']):
        raise ValueError('Finish-current container identity differs')
    return current


def launch_finish(request):
    from panda_qb40 import enqueue as old
    from panda_sensor_refine.launch import inspect_optional
    from directshift48.__main__ import require_idle_gpu
    server = request['server']; folder = control(server)
    path = folder / 'finish_intent.json'
    name = f'panda-qb-rre-current-{server}-v1'
    if path.exists():
        intent = read(path)
    else:
        token = uuid.uuid4().hex
        _, identity = old.command(server, name, token=token)
        intent = dict(name=name, token=token, identity=identity, args=finish_args(request), at=utc_now())
        atomic_json(path, intent)
    if intent['args'] != finish_args(request):
        raise ValueError('Saved target differs from finish-current intent')
    current = inspect_optional(name)
    if current is None:
        require_idle_gpu('cuda')
        cmd, identity = old.command(server, name, token=intent['token'])
        if not old.same_environment(identity, intent['identity']):
            raise ValueError('Finish-current environment differs')
        subprocess.run(cmd + intent['args'], cwd=ROOT, check=True)
        current = inspect_optional(name)
    authenticate_finish(current, intent)
    receipt = dict(container=name, container_id=current['Id'], target_run=request['target_run'], at=utc_now())
    atomic_json(folder / 'finish_launch.json', receipt)
    return receipt


def ensure_process(module, extra, log_path):
    matches = processes(module, extra)
    if len(matches) > 1:
        raise ValueError('Duplicate local queue owners')
    if matches:
        return matches[0]
    with log_path.open('ab') as log:
        child = subprocess.Popen([sys.executable, '-u', '-m', module] + extra, cwd=ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), start_new_session=True,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    identity = process_identity(child.pid)
    if identity is None:
        raise RuntimeError('Queue child exited immediately; inspect ' + str(log_path))
    return identity


def close_old(server):
    folder = old_root(server)
    if not (folder / 'campaign_state.json').exists():
        return {'status': 'NO_OLD_CAMPAIGN'}
    for identity in processes('panda_qb40', ['publish', '--server', server, '--apply', '--watch']):
        stop_process(identity)
    subprocess.run([sys.executable, '-m', 'panda_qb40', 'publish', '--server', server, '--apply'],
                   cwd=ROOT, check=True)
    report = read(folder / 'sheet_publish_panda_qb40/last_poll.json')
    if report.get('readback_verified') is not True:
        raise ValueError('Old results lack final Sheet readback')
    return dict(at=utc_now(), readback_verified=True, report=report)


def advance(server, chain, request):
    folder = control(server)
    if chain['stage'] == 'FINISH_CURRENT':
        if request['target_run']:
            from panda_sensor_refine.launch import inspect_optional
            from panda_qb40.enqueue import ensure_publisher
            intent = read(folder / 'finish_intent.json')
            current = authenticate_finish(inspect_optional(intent['name']), intent)
            if current['State']['Running']:
                ensure_publisher(server)
                return
            if current['State'].get('ExitCode') != 0 or current['State'].get('OOMKilled'):
                raise ValueError('Current case did not reach its authenticated boundary')
            proof = read(old_root(server) / 'rre_priority_boundary.json')
            if proof['target_run'] != request['target_run']:
                raise ValueError('Boundary target differs')
            atomic_json(folder / 'boundary_complete.json', proof)
        if not (folder / 'old_sheet_closeout.json').exists():
            atomic_json(folder / 'old_sheet_closeout.json', close_old(server))
        chain['stage'] = 'RRE_FOCUS'
    elif chain['stage'] == 'RRE_FOCUS':
        from .enqueue import control_dir
        path = control_dir(server) / 'status.json'
        status = read(path) if path.exists() else {}
        if str(status.get('stage', '')).startswith('FAILED'):
            raise ValueError('RRE focus launch/training failed; inspect retained evidence')
        if status.get('stage') == 'COMPLETED':
            if status.get('sheet_verified') is not True or status.get('campaign_id') != G.CAMPAIGN_ID:
                raise ValueError('RRE focus completion lacks its own Sheet readback')
            from .prerequisites import terminal
            atomic_json(folder / 'focus_complete.json', terminal(ROOT, server))
            chain['stage'] = 'RESUME_TUNE30' if server == 's1' else 'COMPLETED'
        else:
            owner = ensure_process('panda_qb_rre.enqueue', ['--server', server, '--wait'], folder / 'focus_enqueue.log')
            chain['focus_owner_pid'] = owner['pid']
    elif chain['stage'] in ('RESUME_TUNE30', 'RESUME_HARD36'):
        # Reuse the unchanged WV3 boundary controller with separate receipts.
        from panda_qb40 import priority
        priority.DIRECTORY = folder / 'wv3'
        priority.DIRECTORY.mkdir(exist_ok=True)
        priority.advance(chain, {})
    elif chain['stage'] != 'COMPLETED':
        raise ValueError('Unknown handoff stage')


def watch(server):
    folder = control(server); request = read(folder / 'request.json')
    if request['policy'] != POLICY or request['server'] != server:
        raise ValueError('Handoff request differs')
    with (folder / 'watcher.lock').open('a+') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = folder / 'chain.json'
        chain = read(path) if path.exists() else {'stage': 'FINISH_CURRENT'}
        try:
            while True:
                advance(server, chain, request)
                chain.update(server=server, campaign_id=G.CAMPAIGN_ID, at=utc_now(), pid=os.getpid())
                atomic_json(path, chain); atomic_json(folder / 'status.json', chain)
                print(json.dumps(chain), flush=True)
                if chain['stage'] == 'COMPLETED':
                    return 0
                time.sleep(30)
        except BaseException as exc:
            atomic_json(folder / 'status.json', {**chain, 'status': 'ERROR', 'error': repr(exc), 'at': utc_now()})
            raise


def apply(server):
    folder = control(server); folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'apply.lock').open('a+') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owners = processes('panda_qb_rre.handoff', ['watch', '--server', server])
        if owners:
            return dict(status='WATCHING', owners=owners)
        path = folder / 'request.json'
        request = read(path) if path.exists() else snapshot(server)
        if request.get('policy') != POLICY or request.get('server') != server:
            raise ValueError('Existing request differs')
        atomic_json(path, request)
        if not (folder / 'safe_stop.json').exists():
            atomic_json(folder / 'safe_stop.json', stop_old(request))
        if request['target_run']:
            launch_finish(request)
        owner = ensure_process('panda_qb_rre.handoff', ['watch', '--server', server], folder / 'watcher.log')
        return dict(status='FINISH_CURRENT_THEN_RRE_FOCUS', current_run=request['target_run'], watcher=owner)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'apply', 'watch'))
    parser.add_argument('--server', required=True, choices=('s1', 's3'))
    args = parser.parse_args(argv)
    if (ROOT / 'gspread/server.txt').read_text().strip() != args.server:
        raise ValueError('Requested server differs from local marker')
    if args.action == 'watch':
        return watch(args.server)
    if args.action == 'plan':
        request = snapshot(args.server)
        result = {k: request[k] for k in ('policy', 'server', 'campaign_id', 'target_run',
                                         'observed_step', 'observed_campaign_status')}
        result['next_jobs'] = list(G.RUNS[args.server])
    else:
        result = apply(args.server)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
