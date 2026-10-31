"""Own-run completion evidence; no predecessor queue or publication start gate."""
import hashlib
import json
from pathlib import Path
from . import registry as G


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_start(server):
    return {'schema': 'QB_STRUCTURE_LOCAL_START_V1', 'server': server, 'dataset': 'QB',
            'donor_origin_server': server, 'stage1_retrained': False,
            'cross_server_transfer': False, 'predecessor_queue_gate': False}


def terminal(root, server):
    from panda_sensor_refine.orchestration import final_evidence, local
    from panda_sensor_focus import early_stop
    root = Path(root); folder = root / G.OUTPUT / G.CAMPAIGN_ID / server / 'QB'
    state = read(folder / 'campaign_state.json'); rows = state.get('runs', [])
    expected = set(G.RUNS[server])
    if (state.get('status') not in ('FINISHED_REGISTERED_QUEUE', 'FINISHED_WITH_EXPLICIT_EARLY_STOPS')
            or len(rows) != len(expected) or {(r['case'], r['replica']) for r in rows} != expected):
        raise ValueError('QB structure registry is incomplete, duplicated, or failed')
    completed, stopped = [], []
    for row in rows:
        directory = folder / 'runs' / row['run_id'] / 'attempt_01'; cfg = read(directory / 'config.json')
        if (cfg.get('campaign_id') != G.CAMPAIGN_ID or cfg.get('server') != server
                or cfg.get('dataset') != 'QB' or cfg.get('phase') != 'stage2'
                or cfg.get('attempt') != 1 or cfg.get('planned_horizon') != 30000
                or (cfg.get('case'), cfg.get('replica')) != (row['case'], row['replica'])
                or local(row['directory'], root).resolve() != directory.resolve()):
            raise ValueError('QB structure terminal run identity differs')
        if row['status'] == 'COMPLETED' and row['step'] == 30000:
            completed.append({**final_evidence(folder, cfg), 'status': 'COMPLETED'})
        elif row['status'] == early_stop.STATUS:
            receipt = early_stop.verify(row, cfg, root)
            stopped.append({'run_id': row['run_id'], 'step': row['step'], 'status': early_stop.STATUS,
                            'decision': receipt['decision'], 'early_stop_sha256': sha(directory / 'early_stop.json')})
        else:
            raise ValueError('Missing full completion or authenticated QB early stop')
    expected_status = 'FINISHED_WITH_EXPLICIT_EARLY_STOPS' if stopped else 'FINISHED_REGISTERED_QUEUE'
    if state['status'] != expected_status: raise ValueError('Incorrect terminal early-stop status')
    return {'campaign_status': state['status'], 'completed_runs': completed, 'early_stopped_runs': stopped,
            'completed_count': len(completed), 'early_stopped_count': len(stopped),
            'all_registered_accounted': True, 'full_queue_completed_without_early_stops': not stopped,
            'state_sha256': sha(folder / 'campaign_state.json')}
