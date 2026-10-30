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
                or cfg.get('server') != 's3' or cfg.get('dataset') != 'QB'
                or cfg.get('phase') != ('donor' if row['case'] == G.DONOR_CASE else 'stage2') or cfg.get('attempt') != 1
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




def evidence(root, *, check_runtime=False):
    from panda_gf2_closeout import verify_retirement
    root = Path(root)
    try:
        retired = verify_retirement(root, check_runtime=check_runtime)
    except FileNotFoundError as exc:
        raise ValueError('WAIT: authenticated GF2 retirement required before local QB Stage1') from exc
    return {'schema': 'QB_DEEPEN_OWN_STAGE1_PREDECESSOR_V1', 'server': 's3', 'dataset': 'QB',
            'gf2_retirement': retired, 'donor_origin_server': 's3',
            'fresh_stage1_required': True, 'stage1_seed': G.STAGE1_SEED, 'wv3_unchanged_deferred': True}


def terminal(root):
    # This function is called by the unadapted host launcher. Verify selection
    # in a dedicated adapter process so old module globals cannot contaminate it.
    import subprocess, sys
    root = Path(root)
    folder = root / G.OUTPUT / G.CAMPAIGN_ID / 's3/QB'
    result = registry_completion(root, folder, G.CAMPAIGN_ID, ((G.DONOR_CASE, 'R1'),) + G.RUNS,
        {G.DONOR_CASE: 50000, **{case: p['total_updates'] for case, p in G.cases().items()}})
    proof = subprocess.run([sys.executable, '-m', 'panda_qb_deepen', 'verify-donor'],
                           cwd=root, capture_output=True, text=True, check=True)
    result['local_stage1_selection'] = json.loads(proof.stdout)
    return result
