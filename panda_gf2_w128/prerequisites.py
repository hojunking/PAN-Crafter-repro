"""Authenticate the local GF2 donor and preserved successful W112 reference."""
import json
from pathlib import Path
from ta2.evaluation import file_sha
from . import registry as G


def evidence(root):
    from panda_sensor_focus.donor import validate_source, receipt_path
    root = Path(root)
    receipt = validate_source('s3', root)
    directory = root / G.BASELINE_DIRECTORY
    checkpoint = directory / 'models/step_0047000.pt'
    evaluation_path = directory / 'evaluation_0047000.json'
    if file_sha(checkpoint) != G.BASELINE_SHA or file_sha(evaluation_path) != G.BASELINE_EVALUATION_SHA:
        raise ValueError('Preserved GF2 successful W112 reference is missing or changed')
    ev = json.loads(evaluation_path.read_text())
    meta = json.loads(checkpoint.with_suffix('.json').read_text())
    cfg = json.loads((directory / 'config.json').read_text())
    identity = ev['state_before']
    if not (identity == ev['state_after'] == ev['rr_state'] == ev['fr_state']):
        raise ValueError('W112 reference mixed-state evaluation')
    if (identity['run_id'] != G.BASELINE_RUN or identity['completed_update'] != 47000
            or identity['server'] != 's3' or identity['dataset'] != 'GF2'
            or meta['artifact']['file_sha256'] != G.BASELINE_SHA or meta['evaluation'] != ev
            or cfg['campaign_id'] != G.BASELINE_CAMPAIGN or cfg['phase'] != 'stage2'
            or cfg['lr_S'] != 2e-4 or cfg['planned_horizon'] != 60000
            or cfg['stage2_spec']['student_width'] != 112
            or cfg['stage2_spec']['alpha'] != .25 or cfg['stage2_spec']['gamma'] != .5
            or cfg['fixed_donor_source']['checkpoint_sha256'] != receipt['selected']['file_sha256']):
        raise ValueError('W112 baseline configuration differs')
    h, e, p = ev['fr']['metrics']['HQNR'], ev['rr']['metrics']['ERGAS'], ev['rr']['metrics']['PSNR']
    if not (h >= .9645 and e <= .522 and p >= 45.08):
        raise ValueError('Original GF2 reference no longer proves the three targets')
    return {'schema': 'GF2_W128_LOCAL_SOURCE_V1', 'server': 's3', 'cross_server_transfer': False,
            'donor_receipt_sha256': file_sha(receipt_path('s3', root)),
            'donor_model_sha256': receipt['selected']['file_sha256'], 'donor_step': 42000,
            'baseline_run': G.BASELINE_RUN, 'baseline_step': 47000,
            'baseline_checkpoint_sha256': G.BASELINE_SHA, 'baseline_evaluation_sha256': G.BASELINE_EVALUATION_SHA,
            'baseline_config_sha256': file_sha(directory / 'config.json'),
            'baseline_metrics': {'HQNR': h, 'ERGAS': e, 'PSNR': p}}


def terminal(root, server):
    from .adapter import install
    from panda_sensor_refine.orchestration import final_evidence
    plan = install()
    if server != 's3': raise ValueError('Wrong server')
    folder = Path(root) / G.OUTPUT / G.CAMPAIGN_ID / 's3/GF2'
    state = json.loads((folder / 'campaign_state.json').read_text())
    rows = state.get('runs', [])
    expected = {plan.resolve_case('s3', c, r)['run_id'] for c, r in G.RUNS}
    if state.get('status') != 'FINISHED_REGISTERED_QUEUE' or len(rows) != 12 or {r['run_id'] for r in rows} != expected:
        raise ValueError('GF2 W128 queue not fully completed; preserve failures and pending cases')
    for row in rows:
        directory = folder / 'runs' / row['run_id'] / 'attempt_01'
        cfg = json.loads((directory / 'config.json').read_text())
        planned = plan.resolve_case('s3', row['case'], row['replica'])
        if (row['status'] != 'COMPLETED' or row['step'] != 60000
                or any(cfg[k] != planned[k] for k in ('run_id', 'campaign_id', 'seed', 'planned_horizon', 'stage2_spec'))):
            raise ValueError('Wrong completed GF2 run identity')
        final_evidence(folder, cfg)
    return {'campaign_status': state['status'], 'completed_count': 12, 'pending_count': 0,
            'stage1_training_runs': 0, 'all_registered_accounted': True,
            'state_sha256': file_sha(folder / 'campaign_state.json')}
