"""Read-only completion proofs; a safe early stop is never a completed run."""
import hashlib
import json
from pathlib import Path
from . import registry as G


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registry_completion(root, folder, campaign_id, runs, horizons):
    from panda_sensor_refine.orchestration import final_evidence, local
    from panda_sensor_focus import early_stop
    root, folder = Path(root), Path(folder)
    state = read(folder / 'campaign_state.json')
    rows, expected = state.get('runs', []), set(runs)
    if (state.get('status') not in ('FINISHED_REGISTERED_QUEUE', 'FINISHED_WITH_EXPLICIT_EARLY_STOPS')
            or len(rows) != len(expected)
            or {(r['case'], r['replica']) for r in rows} != expected):
        raise ValueError('QB registry is incomplete, duplicated, or failed')
    completed, stopped = [], []
    for row in rows:
        directory = folder / 'runs' / row['run_id'] / 'attempt_01'
        cfg = read(directory / 'config.json')
        horizon = horizons[row['case']]
        if (cfg.get('campaign_id') != campaign_id or cfg.get('planned_horizon') != horizon
                or cfg.get('server') != 's1' or cfg.get('dataset') != 'QB'
                or cfg.get('phase') != 'stage2' or cfg.get('attempt') != 1
                or (cfg.get('case'), cfg.get('replica')) != (row['case'], row['replica'])
                or cfg.get('run_id') != row['run_id']
                or local(row['directory'], root).resolve() != directory.resolve()):
            raise ValueError('QB run identity/horizon/directory differs')
        if row['status'] == 'COMPLETED' and row['step'] == horizon:
            completed.append({**final_evidence(folder, cfg), 'status': 'COMPLETED',
                              'full_horizon_completed': True})
        elif row['status'] == early_stop.STATUS:
            receipt = early_stop.verify(row, cfg, root)
            stopped.append({'run_id': row['run_id'], 'step': row['step'], 'status': early_stop.STATUS,
                            'full_horizon_completed': False, 'planned_horizon': horizon,
                            'early_stop_sha256': sha(directory / 'early_stop.json'),
                            'decision': receipt['decision']})
        else:
            raise ValueError('QB terminal row lacks full completion or authenticated early stop')
    expected_status = 'FINISHED_WITH_EXPLICIT_EARLY_STOPS' if stopped else 'FINISHED_REGISTERED_QUEUE'
    if state['status'] != expected_status:
        raise ValueError('QB campaign status misreports explicit early stops')
    return {'campaign_status': state['status'], 'completed_runs': completed, 'early_stopped_runs': stopped,
            'completed_count': len(completed), 'early_stopped_count': len(stopped),
            'all_registered_accounted': len(completed) + len(stopped) == len(expected),
            'full_queue_completed_without_early_stops': not stopped,
            'state_sha256': sha(folder / 'campaign_state.json')}


def evidence(root, expected_code=None):
    root = Path(root)
    controller = root / 'work_dir/panda_sensor_focus_launchers/s1/status.json'
    try:
        status = read(controller)
    except FileNotFoundError as exc:
        raise ValueError('WAIT: existing focus/WV3 controller status is not available') from exc
    priority = None
    if status.get('server') != 's1' or status.get('stage') != 'COMPLETED' or status.get('status') != 'COMPLETED':
        priority = priority_evidence(root)
    folder = root / 'work_dir/panda_sensor_focus' / G.OLD_CAMPAIGN / 's1/QB'
    proof = registry_completion(root, folder, G.OLD_CAMPAIGN, G.OLD_RUNS,
                                {case: 30000 for case, _ in G.OLD_RUNS})
    admission = read(folder / 'admission.json')
    if expected_code is not None and admission['code'] != expected_code:
        raise ValueError('Previous focus source pin changed')
    return {'schema': 'QB_REFIT_PREDECESSOR_V1', 'server': 's1', 'wv3_unchanged': True,
            'controller': {'stage': status['stage'], 'status': status['status']},
            'priority_boundary': priority,
            'focus_admission_sha256': sha(folder / 'admission.json'), **proof}


def priority_evidence(root):
    """Allow QB after the selected WV3 run, never before its full final evidence.

    Works inside read-only training containers without a Docker socket. The host
    controller records normal container exit; state/final artifact hashes are
    rechecked here. Remaining WV3 stays frozen throughout QB admission.
    """
    from panda_sensor5.transition import campaign_root, completed_run, state_of
    root = Path(root)
    directory = root / G.OUTPUT / 'priority_s1'
    path = directory / 'boundary_complete.json'
    if not path.exists():
        raise ValueError('WAIT: existing focus/WV3 controller has not completed; priority boundary pending')
    request, proof = read(directory / 'request.json'), read(path)
    package, target = request['package'], request['target_run_id']
    if (request.get('policy') != 'USER_20261030_FINISH_CURRENT_THEN_QB_THEN_WV3'
            or request.get('server') != 's1' or package != 'panda_tune30'
            or proof.get('request_sha256') != sha(directory / 'request.json')
            or proof.get('container_exit_code') != 0):
        raise ValueError('Invalid QB priority boundary identity/exit receipt')
    state = state_of(root, package, 's1')
    if (state.get('status') != 'PAUSED_FOR_SENSOR5'
            or any(r['status'] in ('RUNNING', 'STOPPED') for r in state['runs'])
            or sha(campaign_root(root, package, 's1') / 'campaign_state.json') != proof['state_sha256']
            or completed_run(root, package, 's1', target) != proof['completed_run']):
        raise ValueError('QB priority boundary lacks full completion or WV3 changed after the boundary')
    return proof


def terminal(root):
    folder = Path(root) / G.OUTPUT / G.CAMPAIGN_ID / 's1/QB'
    return registry_completion(root, folder, G.CAMPAIGN_ID, G.RUNS,
                                {c: p['total_updates'] for c, p in G.cases().items()})
