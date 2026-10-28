"""Continue independent registered runs after an authenticated support failure.

Control-only adapter: no model, loss, support, dataset, seed or source-pin edits.
The failed donor stays FAILED; its dependent Student is explicitly held.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ERROR = "ValueError('SUPPORT_VIOLATION: every predicted PAN view must support fixed ROI8/stencil7')"
TERMINAL = 'FINISHED_WITH_EXPLICIT_SKIPS'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def context(server):
    from panda_s1loss9 import plan
    root = ROOT / 'work_dir/panda_s1loss9' / plan.CAMPAIGN_ID / server / plan.dataset_for_server(server)
    configs = [plan.resolve_case(server, c, r) for g in plan.groups(server) for c, r in g['runs']]
    return root, root / 'support_failure_recovery', configs


def validate_failure(state, failed_id, student_id):
    rows = [r for r in state['runs'] if r['run_id'] == failed_id]
    if (len(rows) != 1 or rows[0].get('status') != 'FAILED'
            or rows[0].get('error') != ERROR or rows[0].get('phase') != 'donor'
            or rows[0].get('poisoned') is not False or rows[0].get('stop_checkpoint_error')
            or any(r['run_id'] == student_id for r in state['runs'])):
        raise ValueError('Only a known pre-optimizer support failure with an unstarted child may be held')
    return rows[0]


def validate_accounting(state, configs, failed_id, student_id):
    validate_failure(state, failed_id, student_id)
    expected = {c['run_id']: c for c in configs}
    rows = {r['run_id']: r for r in state['runs']}
    if len(rows) != len(state['runs']) or set(rows) != set(expected) - {student_id}:
        raise ValueError('Missing, extra or duplicate registered runs')
    for key, cfg in expected.items():
        if key not in (failed_id, student_id):
            row = rows[key]
            if row.get('status') != 'COMPLETED' or row.get('step') != cfg['planned_horizon']:
                raise ValueError('Unfinished independent run: ' + key)


def verify_final_evidence(root, cfg):
    d = Path(root) / 'runs' / cfg['run_id'] / 'attempt_01'
    step = cfg['planned_horizon']
    evidence = d / f'evaluation_{step:07d}.json'
    metadata = d / 'models' / f'step_{step:07d}.json'
    checkpoint = metadata.with_suffix('.pt')
    if any(not p.is_file() or p.is_symlink() for p in (evidence, metadata, checkpoint)):
        raise ValueError('Missing final evidence')
    ev, meta = read(evidence), read(metadata)
    q, a = ev.get('qualification', {}), meta.get('artifact', {})
    if (not all(q.get(k) for k in ('same_state_verified', 'metrics_complete', 'metrics_finite'))
            or not a.get('checkpoint_verified') or a.get('completed_update') != step
            or sha(checkpoint) != a.get('file_sha256')
            or a.get('model_state_sha256') != ev.get('state_before', {}).get('model_state_sha256')
            or meta.get('config', {}).get('run_id') != cfg['run_id']):
        raise ValueError('Invalid final checkpoint/evaluation proof')


def verify_manifest(server):
    from panda_s1loss9.queue import controller_pin
    root, directory, configs = context(server)
    m = read(directory / 'manifest.json')
    if m['server'] != server or m['adapter_sha256'] != sha(__file__) or m['controller_pin'] != controller_pin():
        raise ValueError('Recovery adapter or original controller source changed')
    failed = validate_failure(read(root / 'campaign_state.json'), m['failed_run_id'], m['held_student_id'])
    for relative, expected in m['immutable_evidence'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Failed evidence changed: ' + relative)
    if failed != m['failed_run']:
        raise ValueError('Original failed run state changed')
    return root, directory, configs, m


def prepare(server):
    from directshift48.reporting import atomic_json, utc_now
    from panda_s1loss9.queue import controller_pin
    root, directory, configs = context(server)
    if (directory / 'manifest.json').exists():
        verify_manifest(server)
        return
    state = read(root / 'campaign_state.json')
    failed = [r for r in state['runs'] if r.get('status') == 'FAILED']
    if state.get('status') != 'FAILED' or len(failed) != 1:
        raise ValueError('Expected exactly one inspected failed donor')
    cfg = next(c for c in configs if c['run_id'] == failed[0]['run_id'])
    child = next(c for c in configs if c['phase'] == 'stage2' and c['donor_version'] == cfg['case'])
    validate_failure(state, cfg['run_id'], child['run_id'])
    d = root / 'runs' / cfg['run_id'] / 'attempt_01'
    pointer = read(d / 'models/latest_resume.json')
    checkpoint = d / 'models' / Path(pointer['checkpoint_path']).name
    if (pointer.get('completed_update') != failed[0]['step']
            or pointer.get('campaign_resume_complete') is not True
            or pointer.get('checkpoint_verified') is not True
            or sha(checkpoint) != pointer.get('file_sha256')):
        raise ValueError('Failed run lacks an intact exact-resume checkpoint')
    directory.mkdir(exist_ok=True)
    atomic_json(directory / 'campaign_before.json', state)
    launcher = ROOT / 'work_dir/panda_s1loss9_launchers' / server
    atomic_json(directory / 'chain_before.json', read(launcher / 'chain.json'))
    atomic_json(directory / 'status_before.json', read(launcher / 'status.json'))
    files = [d / 'run_state.json', d / 'config.json', d / 'train.jsonl',
             d / 'models/latest_resume.json', checkpoint, checkpoint.with_suffix('.json')]
    atomic_json(directory / 'manifest.json', {
        'schema': 'S1LOSS9_AUTHENTICATED_SUPPORT_FAILURE_RECOVERY_V1', 'at': utc_now(),
        'server': server, 'failed_run_id': cfg['run_id'], 'held_student_id': child['run_id'],
        'failed_run': failed[0], 'reason': 'Preserve failed scientific case; continue independent previously authorized queue',
        'adapter_sha256': sha(__file__), 'controller_pin': controller_pin(),
        'immutable_evidence': {str(p.relative_to(ROOT)): sha(p) for p in files},
        'method_changed': False, 'failed_marked_completed': False,
        'hold_reason': 'No completed, validation-selected Stage1 donor for dependent Student'})
    verify_manifest(server)


def probe(server):
    """Read the exact next batch; no optimizer update or production file write."""
    import torch
    from panda_s1loss9 import plan
    from panda_s1loss9.stage1 import load_stage1
    from panda_s1loss9.runtime import code_pin
    from panda_s1loss9.aligner import support_for_shifts
    from panda_method.edge_data import EdgeDataSources
    from panda_stage2_condq.training import configure_precision
    from ta2.evaluation import evaluator_identity
    from directshift48.__main__ import require_idle_gpu
    require_idle_gpu('cuda')
    configure_precision()
    root, _, _, m = verify_manifest(server)
    d = root / 'runs' / m['failed_run_id'] / 'attempt_01'
    cfg, ptr = read(d / 'config.json'), read(d / 'models/latest_resume.json')
    src = EdgeDataSources(ROOT / 'ablr2/sensor_sources.json', root=ROOT,
                          sensor=cfg['dataset'], lms_overrides=plan.lms_overrides(cfg['dataset']))
    trainer, stream, _ = load_stage1(d / 'models' / Path(ptr['checkpoint_path']).name,
        ptr['file_sha256'], cfg, src, device='cuda', code_pin=code_pin(), evaluator_pin=evaluator_identity())
    violations = []
    with torch.no_grad():
        for batch in stream.next_update():
            u = trainer.stage.native_prediction(batch)['u']
            good = support_for_shifts(*batch.pan.shape[-2:], u)[..., 7:-7, 7:-7].flatten(1).all(1)
            for i in (~good).nonzero().flatten().tolist():
                violations.append({'sample_id': batch.lineage.sample_ids[i], 'dy_dx': u[i].cpu().tolist(),
                                   'augmentation': batch.lineage.augmentations[i]})
    if not violations:
        raise ValueError('Saved next batch did not reproduce the support failure')
    print(json.dumps({'checkpoint_step': trainer.updates, 'attempted_update': trainer.updates + 1,
                      'violations': violations, 'optimizer_updates_performed': 0}), flush=True)


def run_remaining(server):
    from directshift48.reporting import atomic_json, utc_now
    from directshift48.__main__ import require_idle_gpu
    from panda_s1loss9.runtime import Campaign, BudgetExpired
    from panda_s1loss9 import plan
    require_idle_gpu('cuda')
    root, directory, configs, m = verify_manifest(server)
    camp = Campaign(server)
    excluded = {m['failed_run_id'], m['held_student_id']}
    original_check = camp.check
    def check(**kwargs):
        original_check(**kwargs)
        if sha(__file__) != m['adapter_sha256']:
            raise RuntimeError('Recovery adapter changed during execution')
    camp.check = check
    with (root / '.controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = {s: signal.signal(s, lambda *a: setattr(camp, 'stop_requested', True))
               for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            camp.admission(ROOT / 'ablr2/sensor_sources.json')
            camp.state.pop('error', None)
            camp.state.pop('traceback', None)
            camp.state['recovery_control'] = {'manifest': str(directory / 'manifest.json'),
                'failed_run_id': m['failed_run_id'], 'held_student_id': m['held_student_id'],
                'method_changed': False, 'opened_at': utc_now()}
            camp.persist()
            for group in camp.config['groups']:
                selected = [plan.resolve_case(server, c, r) for c, r in group['runs']]
                selected = [c for c in selected if c['run_id'] not in excluded]
                if not selected:
                    continue
                camp.admit_group(group)
                for cfg in selected:
                    camp.check()
                    if not camp.completed(cfg):
                        camp.validate_preflight(cfg)
                        camp.run_case(cfg)
                    if cfg['case'] in plan.DONORS:
                        camp.register_donor(cfg['case'])
            validate_accounting(camp.state, configs, m['failed_run_id'], m['held_student_id'])
            camp.state['status'] = TERMINAL
            camp.state['recovery_control'].update(completed_runs=16, failed_runs=1, dependent_runs_held=1)
        except BudgetExpired as exc:
            camp.state.update(status='STOPPED_SAFE', reason=str(exc))
            raise
        except BaseException as exc:
            camp.state.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
            raise
        finally:
            camp.persist()
            for sig, handler in old.items():
                signal.signal(sig, handler)
    verify_manifest(server)
    atomic_json(directory / 'loss_grid_accounted.json', {'at': utc_now(), 'status': TERMINAL,
        'completed_runs': 16, 'failed_run_id': m['failed_run_id'], 'held_student_id': m['held_student_id'],
        'all_registered_runs_successful': False, 'independent_runs_complete': True})


def watch(server):
    from directshift48.reporting import atomic_json, utc_now
    from panda_s1loss9 import launch, queue
    from directshift48.__main__ import require_idle_gpu
    root, directory, configs, m = verify_manifest(server)
    original = launch.directory(server)
    with ExitStack() as stack:
        for path in [original / 'watcher.lock', ROOT / 'work_dir/panda_sensor5_launchers' / server / 'watcher.lock']:
            f = stack.enter_context(path.open('a+'))
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / 'chain.json'
        chain = read(path) if path.exists() else {'stage': 'RECOVER_LOSS9', 'policy': queue.POLICY,
            'recovery_manifest': str(directory / 'manifest.json')}
        try:
            while chain['stage'] != 'COMPLETED':
                verify_manifest(server)
                if chain['stage'] == 'RECOVER_LOSS9':
                    receipt_path = directory / 'launch.json'
                    if not receipt_path.exists():
                        require_idle_gpu('cuda')
                        name = f'panda-s1loss9-recovery-{server}-{time.time_ns()}'
                        cmd, _, source = launch.command(server, package='panda_s1loss9', name=name)
                        result = subprocess.run(cmd + ['python', 'tools/s1loss9_recovery.py', 'run', '--server', server],
                                                capture_output=True, text=True, check=True)
                        atomic_json(receipt_path, {'container': name, 'container_id': result.stdout.strip(),
                                                  'image': source['Image'], 'at': utc_now()})
                    receipt = read(receipt_path)
                    container = queue.inspect_optional(receipt['container'])
                    if container is None or container['Id'] != receipt['container_id']:
                        raise ValueError('Recovery container identity changed')
                    if container['State']['Running']:
                        if read(root / 'campaign_state.json').get('status') == 'RUNNING':
                            from panda_s1loss9.best_publication.restart import restart, find_writers
                            if not find_writers(root.resolve(), server):
                                restart(server)
                    else:
                        if container['State'].get('ExitCode') != 0 or container['State'].get('OOMKilled'):
                            raise ValueError('Recovery stopped on a new error; inspect without bypassing it')
                        s = read(root / 'campaign_state.json')
                        if s.get('status') != TERMINAL:
                            raise ValueError('Recovery missing explicit partial-completion state')
                        validate_accounting(s, configs, m['failed_run_id'], m['held_student_id'])
                        for cfg in configs:
                            if cfg['run_id'] not in (m['failed_run_id'], m['held_student_id']):
                                verify_final_evidence(root, cfg)
                        chain['stage'] = 'RESUME_SENSOR5'
                else:
                    queue.advance(server, original, chain)
                chain.update(pid=os.getpid(), server=server, at=utc_now(), status=chain['stage'])
                for dest in [path, original / 'chain.json', original / 'status.json']:
                    atomic_json(dest, chain)
                if chain['stage'] != 'COMPLETED':
                    time.sleep(15)
        except BaseException as exc:
            report = {**chain, 'status': 'BLOCKED', 'error': repr(exc), 'at': utc_now()}
            atomic_json(original / 'status.json', report)
            atomic_json(directory / 'last_error.json', report)
            raise


def enqueue(server):
    from directshift48.reporting import atomic_json, utc_now
    root, directory, _, _ = verify_manifest(server)
    original = ROOT / 'work_dir/panda_s1loss9_launchers' / server
    with (original / 'enqueue.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (original / 'watcher.lock').open('a+') as watcher:
            fcntl.flock(watcher, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (directory / 'controller.log').open('ab') as log:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'watch', '--server', server],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        value = {'pid': child.pid, 'at': utc_now(), 'log': str(directory / 'controller.log')}
        atomic_json(directory / 'enqueue.json', value)
        print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'probe', 'run', 'watch', 'enqueue'))
    parser.add_argument('--server', required=True, choices=('s1', 's3'))
    args = parser.parse_args()
    if (ROOT / 'gspread/server.txt').read_text().strip() != args.server:
        raise ValueError('Server marker mismatch')
    {'prepare': prepare, 'probe': probe, 'run': run_remaining,
     'watch': watch, 'enqueue': enqueue}[args.command](args.server)
