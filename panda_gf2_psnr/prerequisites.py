"""Read-only admission: existing GF2 focus and deferred WV3 must finish first."""
import hashlib
import json
from pathlib import Path
from .registry import OLD_CAMPAIGN, OLD_RUNS


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence(root, expected_code=None):
    from panda_sensor_refine.orchestration import final_evidence
    root = Path(root)
    controller = root / 'work_dir/panda_sensor_focus_launchers/s3/status.json'
    status = read(controller)
    if status.get('server') != 's3' or status.get('stage') != 'COMPLETED' or status.get('status') != 'COMPLETED':
        raise ValueError('WAIT: existing focus/WV3 controller has not completed')
    folder = root / 'work_dir/panda_sensor_focus' / OLD_CAMPAIGN / 's3/GF2'
    state = read(folder / 'campaign_state.json')
    rows = state.get('runs', [])
    if (state.get('status') != 'FINISHED_REGISTERED_QUEUE' or len(rows) != len(OLD_RUNS)
            or {(r['case'], r['replica']) for r in rows} != set(OLD_RUNS)):
        raise ValueError('Existing GF2 focus registry is incomplete')
    proofs = []
    for row in rows:
        cfg = read(folder / 'runs' / row['run_id'] / 'attempt_01/config.json')
        if (row['status'] != 'COMPLETED' or row['step'] != 60000
                or cfg['campaign_id'] != OLD_CAMPAIGN or cfg['planned_horizon'] != 60000
                or cfg['server'] != 's3' or cfg['dataset'] != 'GF2'
                or (cfg['case'], cfg['replica']) != (row['case'], row['replica'])
                or cfg['run_id'] != row['run_id']):
            raise ValueError('Previous GF2 run identity/horizon differs')
        final_evidence(folder, cfg)
        proofs.append({'run_id': row['run_id'], 'step': row['step']})
    admission = read(folder / 'admission.json')
    if expected_code is not None and admission['code'] != expected_code:
        raise ValueError('Previous focus source pin changed')
    return {'schema': 'GF2_PSNR4508_PREDECESSOR_V1', 'server': 's3', 'wv3_unchanged': True,
            'controller': {'stage': status['stage'], 'status': status['status']},
            'focus_state_sha256': sha(folder / 'campaign_state.json'),
            'focus_admission_sha256': sha(folder / 'admission.json'), 'runs': proofs}
