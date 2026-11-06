import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/panda_wv2_eval'
PLAN = json.loads((CONFIG / 'plan.json').read_text())
OUT = ROOT / PLAN['output']


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    from ta2.evaluation import atomic_json
    atomic_json(path, value)


def verify_release():
    release = read(CONFIG / 'release.json')
    if release['sha256'] != digest(release['files']):
        raise ValueError('Invalid WV2 source manifest')
    for path, expected in release['files'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('Queued WV2 source changed: ' + path)
    return release['sha256']


def dependency_ready(state):
    d = PLAN['dependency']
    if state['campaign_id'] != d['campaign_id'] or state['queue_sha256'] != d['queue_sha256']:
        raise ValueError('WV3 dependency identity changed')
    rows = state['runs']
    if [r['run_id'] for r in rows] != d['run_ids']:
        raise ValueError('WV3 dependency queue changed')
    complete = sum(r['status'] == 'COMPLETED' and r['step'] == d['required_step'] for r in rows)
    return state['status'] == 'COMPLETED' and complete == len(rows), f'{complete}/{len(rows)} completed; {state["status"]}'


def verify_dependency_completion():
    d = PLAN['dependency']
    statepath = ROOT / d['state_path']
    ready, reason = dependency_ready(read(statepath))
    if not ready:
        raise ValueError('WV3 queue must finish before WV2 evaluation: ' + reason)
    for run_id in d['run_ids']:
        record = read(statepath.parent / 'runs' / run_id / 'evaluation_0030000.json')
        if record['run_id'] != run_id or record['step'] != 30000 or not record['same_state_verified']:
            raise ValueError('Dependency final evaluation missing or mismatched')
    return {'state_sha256': sha(statepath), 'completed_runs': len(d['run_ids'])}


def source_evidence(case):
    for path_key, sha_key in [('checkpoint', 'checkpoint_sha256'), ('evaluation', 'evaluation_sha256'), ('config', 'config_sha256')]:
        if sha(ROOT / case[path_key]) != case[sha_key]:
            raise ValueError('Selected WV3 artifact changed: ' + case[path_key])
    cfg, ev = read(ROOT / case['config']), read(ROOT / case['evaluation'])
    if (cfg['run_id'] != case['source_run_id'] or ev['run_id'] != cfg['run_id']
            or cfg['dataset'] != 'WV3' or cfg['server'] != 's1' or cfg['stage'] != 'stage2'
            or ev['step'] != case['source_step'] or not ev['same_state_verified']
            or ev['checkpoint']['file_sha256'] != case['checkpoint_sha256']
            or cfg['student_input'] != 'CORRECTED_DIFF' or cfg['scale'] != 1
            or cfg['upsampler'] != 'interp23' or cfg['warp'] != 'LANCZOS4_REFLECTION'
            or cfg['architecture'] != f'W{case["width"]}_D123_UBIL'
            or cfg['inference_tta'] or cfg['beta'] != 0):
        raise ValueError('Selected source is not the registered full-scale WV3 model')
    return cfg, ev
