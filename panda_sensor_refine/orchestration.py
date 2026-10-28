"""Control-only retained LOSS9 screen; original training and evidence stay intact."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
from pathlib import Path
import signal
import time
import traceback

from directshift48.reporting import atomic_json, utc_now

ROOT = Path(__file__).resolve().parents[1]
POLICY = 'USER_20261028_SENSOR_REFINE_R4_KEEP_WV3_UNCHANGED'
LEGACY_TERMINAL = 'FINISHED_RETAINED_SCREEN_WITH_EXPLICIT_SKIPS'
SUPPORT_ERROR = "ValueError('SUPPORT_VIOLATION: every predicted PAN view must support fixed ROI8/stencil7')"
RETAINED_PAIRS = {
    's1': ('A200_E003', 'A020_E003', 'A050_E003', 'A200_E0003',
           'A200_E030', 'A020_E0003', 'A050_E0003', 'A050_E030'),
    's3': ('A200_E030', 'A020_E003', 'A050_E030', 'A200_E003',
           'A050_E003', 'A200_E0003'),
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def local(path, root=ROOT):
    path = Path(path)
    return Path(root) / path.relative_to('/workspace') if str(path).startswith('/workspace/') else path


def launcher_root(root, server):
    return Path(root) / 'work_dir/panda_sensor_refine_launchers' / server


def legacy_root(root, server):
    from panda_s1loss9 import plan
    return Path(root) / 'work_dir/panda_s1loss9' / plan.CAMPAIGN_ID / server / plan.dataset_for_server(server)


def legacy_configs(server):
    from panda_s1loss9 import plan
    return [plan.resolve_case(server, c, r) for group in plan.groups(server) for c, r in group['runs']]


def selected_configs(server, state, configs=None):
    """Finish any already-started pair first; preserve requested sensor ordering."""
    configs = legacy_configs(server) if configs is None else configs
    by_id = {c['run_id']: c for c in configs}
    active = [r for r in state.get('runs', []) if r['status'] in ('RUNNING', 'STOPPED')]
    if len(active) > 1 or any(r['run_id'] not in by_id for r in state.get('runs', [])):
        raise ValueError('Unknown or multiple active legacy runs')
    pairs = [by_id[r['run_id']]['loss_pair'] for r in active]
    pairs += list(RETAINED_PAIRS[server])
    pairs = list(dict.fromkeys(pairs))
    return [cfg for pair in pairs for cfg in configs if cfg['loss_pair'] == pair]


def authenticate_stop(run, cfg, root=ROOT):
    if run.get('poisoned') or run.get('stop_checkpoint_error'):
        raise ValueError('Optimizer or exact-resume checkpoint is unsafe')
    directory = local(run['directory'], root).resolve()
    if directory.name != 'attempt_01' or directory.parent.name != cfg['run_id']:
        raise ValueError('Resume directory differs from registered run')
    pointer = read(directory / 'models/latest_resume.json')
    checkpoint = local(pointer['checkpoint_path'], root)
    if (checkpoint.is_symlink() or checkpoint.resolve().parent != directory / 'models'
            or pointer.get('completed_update') != run['step']
            or pointer.get('campaign_resume_complete') is not True
            or pointer.get('checkpoint_verified') is not True
            or sha(checkpoint) != pointer.get('file_sha256')):
        raise ValueError('Complete optimizer/data/RNG resume proof missing')
    saved = read(directory / 'run_state.json')
    if any(saved.get(k) != run.get(k) for k in ('run_id', 'status', 'step')):
        raise ValueError('Run and campaign resume states differ')
    return {'run_id': cfg['run_id'], 'step': run['step'],
            'checkpoint': str(checkpoint.resolve().relative_to(Path(root).resolve())),
            'checkpoint_sha256': pointer['file_sha256']}


def support_failure(run, cfg, root=ROOT):
    if (run.get('status') != 'FAILED' or cfg['phase'] != 'donor'
            or run.get('error') != SUPPORT_ERROR or run.get('poisoned') is not False
            or run.get('stop_checkpoint_error')):
        raise ValueError('Only authenticated pre-optimizer support failure may be held')
    proof = authenticate_stop(run, cfg, root)
    directory = local(run['directory'], root)
    proof.update(error=SUPPORT_ERROR, failed_record=run,
                 run_state_sha256=sha(directory / 'run_state.json'),
                 trace_sha256=sha(directory / 'train.jsonl'))
    return proof


def record_preflight_failure(camp, cfg, exc, directory):
    """Isolate a known disposable failure without fabricating a production run."""
    if not isinstance(exc, ValueError) or not str(exc).startswith('SUPPORT_VIOLATION:'):
        return None
    if (any(r['run_id'] == cfg['run_id'] for r in camp.state['runs'])
            or (camp.root / 'runs' / cfg['run_id']).exists()):
        return None
    camp.verify_code()
    donor = None
    if cfg['phase'] == 'stage2':
        source = camp.state['donors'].get(cfg['donor_version'])
        if source is None: raise ValueError('Legacy preflight failure has no declared donor')
        donor = {k: source[k] for k in ('checkpoint_path', 'checkpoint_sha256', 'selector_receipt_path', 'selector_sha256')}
    proof = {'schema': 'SENSOR_REFINE_LEGACY_PREFLIGHT_SUPPORT_FAILURE_V1',
             'run_id': cfg['run_id'], 'case': cfg['case'], 'step': 0,
             'error': str(exc), 'requested_config': cfg, 'legacy_code_pin': camp.code,
             'production_updates': 0, 'production_run_absent': True,
             'production_weights_started': False, 'donor': donor}
    proof = json.loads(json.dumps(proof))
    path = directory / ('legacy_preflight_' + cfg['case'] + '.json')
    if path.exists() and read(path) != proof: raise ValueError('Legacy preflight failure evidence changed')
    atomic_json(path, proof)
    record = {'run_id': cfg['run_id'], 'step': 0, 'error': str(exc),
              'evidence_kind': 'FAILED_PREFLIGHT_NO_PRODUCTION_UPDATES',
              'receipt_path': str(path.relative_to(ROOT)), 'receipt_sha256': sha(path)}
    camp.state['sensor_refine_control'].setdefault('preflight_failures', {})[cfg['run_id']] = record
    camp.persist()
    validate_preflight_failure(ROOT, camp.server, cfg, record, camp.state)
    return record


def validate_preflight_failure(root, server, cfg, record, state):
    from panda_s1loss9 import runtime, selection, plan
    campaign = legacy_root(root, server)
    path = Path(root) / record['receipt_path']
    expected = launcher_root(root, server) / ('legacy_preflight_' + cfg['case'] + '.json')
    if (path.resolve() != expected.resolve() or path.is_symlink()
            or sha(path) != record.get('receipt_sha256')
            or any(r['run_id'] == cfg['run_id'] for r in state['runs'])
            or (campaign / 'runs' / cfg['run_id']).exists()):
        raise ValueError('Legacy preflight failure has production evidence or changed receipt')
    proof = read(path)
    if (proof.get('schema') != 'SENSOR_REFINE_LEGACY_PREFLIGHT_SUPPORT_FAILURE_V1'
            or proof.get('run_id') != cfg['run_id'] or record.get('run_id') != cfg['run_id']
            or plan.canonical_sha256(proof.get('requested_config')) != plan.canonical_sha256(cfg)
            or proof.get('legacy_code_pin') != runtime.code_pin()
            or proof.get('production_updates') != 0 or proof.get('step') != 0 or record.get('step') != 0
            or proof.get('production_run_absent') is not True or proof.get('production_weights_started') is not False
            or proof.get('error') != record.get('error') or not proof.get('error', '').startswith('SUPPORT_VIOLATION:')):
        raise ValueError('Legacy preflight failure config/source evidence differs')
    donor = proof.get('donor')
    if cfg['phase'] == 'stage2':
        selected = state.get('donors', {}).get(cfg['donor_version'])
        if not isinstance(donor, dict) or selected is None or any(donor.get(k) != selected.get(k) for k in donor):
            raise ValueError('Legacy preflight donor differs from selected reference')
        receipt = selection.validate_receipt(read(local(donor['selector_receipt_path'], root)))
        if (sha(local(donor['checkpoint_path'], root)) != donor['checkpoint_sha256']
                or receipt['receipt_sha256'] != donor['selector_sha256']):
            raise ValueError('Legacy preflight source changed')
    elif donor is not None:
        raise ValueError('Fresh legacy donor preflight must not claim an inherited donor')
    return record


def final_evidence(root, cfg):
    directory = Path(root) / 'runs' / cfg['run_id'] / 'attempt_01'
    step = cfg['planned_horizon']
    evidence = directory / f'evaluation_{step:07d}.json'
    meta_path = directory / 'models' / f'step_{step:07d}.json'
    checkpoint = meta_path.with_suffix('.pt')
    if any(not p.is_file() or p.is_symlink() for p in (evidence, meta_path, checkpoint)):
        raise ValueError('Final checkpoint/evaluation evidence missing')
    ev, meta = read(evidence), read(meta_path)
    q, a = ev.get('qualification', {}), meta.get('artifact', {})
    if (not all(q.get(k) for k in ('same_state_verified', 'metrics_complete', 'metrics_finite'))
            or a.get('checkpoint_verified') is not True or a.get('completed_update') != step
            or a.get('file_sha256') != sha(checkpoint)
            or a.get('model_state_sha256') != ev.get('state_before', {}).get('model_state_sha256')
            or meta.get('config', {}).get('run_id') != cfg['run_id']):
        raise ValueError('Final same-state checkpoint proof differs')
    return {'run_id': cfg['run_id'], 'step': step, 'checkpoint_sha256': a['file_sha256'],
            'evaluation_sha256': sha(evidence)}


def build_completion(root, server, selected_ids):
    configs = legacy_configs(server)
    campaign = legacy_root(root, server)
    state = read(campaign / 'campaign_state.json')
    rows = {r['run_id']: r for r in state['runs']}
    expected = {c['run_id']: c for c in configs}
    if len(rows) != len(state['runs']) or not set(rows) <= set(expected):
        raise ValueError('Legacy registry has extra/duplicate runs')
    failures = {key: support_failure(row, expected[key], root)
                for key, row in rows.items() if row['status'] == 'FAILED'}
    for key, record in state.get('sensor_refine_control', {}).get('preflight_failures', {}).items():
        if key not in expected or key in failures: raise ValueError('Unknown/duplicate legacy preflight failure')
        failures[key] = validate_preflight_failure(root, server, expected[key], record, state)
    failed_pairs = {expected[key]['loss_pair'] for key in failures if expected[key]['phase'] == 'donor'}
    completed, held, skipped = [], [], []
    for cfg in configs:
        key = cfg['run_id']; row = rows.get(key)
        if row and row['status'] == 'COMPLETED' and row['step'] == cfg['planned_horizon']:
            completed.append(final_evidence(campaign, cfg))
        elif key in failures:
            continue
        elif row:
            raise ValueError('Unfinished registered run cannot be concealed: ' + key)
        elif cfg['phase'] == 'stage2' and cfg['loss_pair'] in failed_pairs:
            held.append({'run_id': key, 'reason': 'FAILED_DONOR_NO_SELECTED_REFERENCE'})
        elif key not in selected_ids:
            skipped.append({'run_id': key, 'reason': 'USER_SUPERSEDED_BEFORE_START'})
        else:
            raise ValueError('Retained run has neither completion nor safe failure: ' + key)
    return {'schema': 'SENSOR_REFINE_LEGACY_SCREEN_ACCOUNTING_V1', 'policy': POLICY,
            'server': server, 'status': LEGACY_TERMINAL, 'selected_run_ids': list(selected_ids),
            'completed': completed, 'failed': list(failures.values()),
            'held_students': held, 'excluded_unstarted': skipped,
            'all_original_runs_completed': not (failures or held or skipped)}


def predecessor_receipt(root, server):
    """Deterministic, revalidated admission proof usable from host or /workspace."""
    from panda_s1loss9 import plan
    from panda_s1loss9.selection import validate_receipt
    directory = launcher_root(root, server)
    saved = read(directory / 'legacy_completion.json')
    current = build_completion(root, server, saved['selected_run_ids'])
    # Failed run directories are process-local paths; compare canonical JSON
    # from the immutable saved host record instead of rewriting old evidence.
    if current != saved:
        raise ValueError('Legacy retained-screen evidence changed after completion')
    source_cfg = plan.resolve_case(server, 'S1_A200_E030')
    value = {'schema': 'SENSOR_REFINE_COMPLETED_LEGACY_SCREEN_V1', 'policy': POLICY,
            'server': server, 'legacy_campaign_id': plan.CAMPAIGN_ID,
            'completion_sha256': sha(directory / 'legacy_completion.json'),
            'successor_skips': [], 'wv3_registry_unchanged': True}
    if source_cfg['run_id'] in {p['run_id'] for p in current['completed']}:
        source = legacy_root(root, server) / 'donors/S1_A200_E030/rre_selection.json'
        selected = validate_receipt(read(source))
        value.update(source_selection_sha256=sha(source),
                     source_selected_step=selected['selected']['completed_update'],
                     external_donor={'status': 'COMPLETED'})
    else:
        failures = [p for p in current['failed'] if p['run_id'] == source_cfg['run_id']]
        if len(failures) != 1:
            raise ValueError('External reference has neither completion nor authenticated support failure')
        value['external_donor'] = {'status': 'FAILED_SUPPORT', 'failure': failures[0]}
    return value


def run_legacy(server):
    from panda_s1loss9 import runtime, plan
    from directshift48.__main__ import require_idle_gpu
    require_idle_gpu('cuda')
    directory = launcher_root(ROOT, server)
    request = read(directory / 'takeover.json')
    from .queue import controller_pin
    if request['controller_pin'] != controller_pin():
        raise ValueError('Frozen refinement controller changed before legacy continuation')
    camp = runtime.Campaign(server)
    selection_path = directory / 'legacy_selection.json'
    if selection_path.exists():
        selection = read(selection_path)
        wanted = set(selection['run_ids'])
        registered = {c['run_id']: c for c in legacy_configs(server)}
        chosen = [registered[key] for key in selection['run_ids']]
    else:
        chosen = selected_configs(server, camp.state)
        selection = {'policy': POLICY, 'server': server, 'run_ids': [c['run_id'] for c in chosen]}
        atomic_json(selection_path, selection)
    original_check = camp.check
    next_pin_check = [0.]
    def check(**kwargs):
        original_check(**kwargs)
        if time.monotonic() >= next_pin_check[0]:
            if request['controller_pin'] != controller_pin():
                raise RuntimeError('Refinement adapter changed during legacy continuation')
            next_pin_check[0] = time.monotonic() + 20.
    camp.check = check
    with (camp.root / '.controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = {s: signal.signal(s, lambda *a: setattr(camp, 'stop_requested', True))
               for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            # Original config, code, data, environment and predecessor checks.
            camp.admission(ROOT / 'ablr2/sensor_sources.json')
            camp.state.pop('error', None); camp.state.pop('traceback', None)
            control = camp.state.setdefault('sensor_refine_control', {})
            control.update(policy=POLICY, selected_run_ids=selection['run_ids'], original_training_unchanged=True)
            failed = set()
            failed_ids = set()
            for cfg in legacy_configs(server):
                row = next((r for r in camp.state['runs'] if r['run_id'] == cfg['run_id']), None)
                if row and row['status'] == 'FAILED':
                    support_failure(row, cfg); failed.add(cfg['loss_pair']); failed_ids.add(cfg['run_id'])
                record = control.get('preflight_failures', {}).get(cfg['run_id'])
                if record:
                    validate_preflight_failure(ROOT, server, cfg, record, camp.state)
                    failed_ids.add(cfg['run_id'])
                    if cfg['phase'] == 'donor': failed.add(cfg['loss_pair'])
            for cfg in chosen:
                if cfg['loss_pair'] in failed or cfg['run_id'] in failed_ids:
                    continue
                camp.check()
                if not camp.completed(cfg):
                    camp.admit_group({'name': 'retain_' + cfg['case'], 'runs': [(cfg['case'], cfg['replica'])]})
                    try:
                        camp.validate_preflight(cfg)
                    except ValueError as exc:
                        proof = record_preflight_failure(camp, cfg, exc, directory)
                        if proof is None: raise
                        failed_ids.add(cfg['run_id'])
                        if cfg['phase'] == 'donor': failed.add(cfg['loss_pair'])
                        continue
                    try:
                        camp.run_case(cfg)
                    except ValueError:
                        row = next(r for r in camp.state['runs'] if r['run_id'] == cfg['run_id'])
                        proof = support_failure(row, cfg)
                        failed.add(cfg['loss_pair'])
                        atomic_json(directory / ('support_failure_' + cfg['case'] + '.json'), proof)
                        camp.state['status'] = 'RUNNING'; camp.persist()
                        continue
                if cfg['case'] in plan.DONORS:
                    camp.register_donor(cfg['case'])
            camp.state['status'] = LEGACY_TERMINAL
            camp.persist()
            completion = build_completion(ROOT, server, selection['run_ids'])
            path = directory / 'legacy_completion.json'
            if path.exists() and read(path) != completion:
                raise ValueError('Existing legacy completion receipt changed')
            atomic_json(path, completion)
        except runtime.BudgetExpired as exc:
            camp.state.update(status='STOPPED_SAFE', reason=str(exc))
            raise
        except BaseException as exc:
            camp.state.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
            raise
        finally:
            camp.persist()
            for s, handler in old.items(): signal.signal(s, handler)
    predecessor_receipt(ROOT, server)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True, choices=('s1', 's3'))
    args = parser.parse_args(argv)
    if (ROOT / 'gspread/server.txt').read_text().strip() != args.server:
        raise ValueError('Local server marker mismatch')
    return run_legacy(args.server)


if __name__ == '__main__':
    raise SystemExit(main())
