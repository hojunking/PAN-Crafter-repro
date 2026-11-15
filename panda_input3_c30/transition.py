"""Server-local, immutable migration manifest. No remote state or model copying."""
from copy import deepcopy
import json
from pathlib import Path
from directshift48.reporting import atomic_json, utc_now
from . import recipes

TERMINAL = ('COMPLETED', 'FAILED', 'BLOCKED_DONOR', 'EARLY_STOPPED')
REPLACED = 'REPLACED_BY_C30'


def was_started(job, directory):
    if int(job.get('step', 0)) > 0 or job['status'] not in ('QUEUED', REPLACED):
        return True
    directory = Path(directory)
    # A worker may have initialized, or committed updates before its state heartbeat.
    return any((directory / p).exists() for p in ('config.json', 'train.jsonl', 'latest.json')) or any(
        directory.glob('evaluation_*.json')) or any(directory.glob('resume/*.pt')) or any(directory.glob('models/*.pt'))


def classify(state, root):
    root = Path(root)
    pending, started, finish = [], [], []
    for row in state['runs']:
        if was_started(row, root / 'runs' / row['run_id']):
            started.append(row['case'])
            if row['status'] not in TERMINAL:
                finish.append(row['run_id'])
        else:
            pending.append(row['case'])
    return dict(replaced_cases=pending, rerun_cases=started,
                finish_original_run_ids=finish, c30_order=pending + started)


def read_cutover(server):
    p = recipes.campaign_root(server) / 'cutover.json'
    value = json.loads(p.read_text())
    if (value.get('server') != server or value.get('campaign_id') != recipes.CAMPAIGN_ID
            or value.get('original_campaign_id') != recipes.ORIGINAL_CAMPAIGN
            or value.get('manifest_sha256') != recipes.original.digest(
                {k: v for k, v in value.items() if k != 'manifest_sha256'})
            or sorted(value['c30_order']) != sorted(recipes.CASES)
            or value['c30_order'] != value['replaced_cases'] + value['rerun_cases']):
        raise ValueError('Invalid server-local C30 transition manifest')
    return value


def prepare(server):
    """Called only after the authenticated original writer has stopped cleanly."""
    from panda_input3.runtime import read_state
    root = recipes.ORIGINAL_ROOT(server)
    state = read_state(root, server)
    folder = recipes.campaign_root(server); folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'cutover.json'
    if path.exists():
        value = read_cutover(server)
    else:
        value = dict(schema='INPUT3_C30_CUTOVER_V1', server=server, at=utc_now(),
            campaign_id=recipes.CAMPAIGN_ID, original_campaign_id=recipes.ORIGINAL_CAMPAIGN,
            original_state_sha256=recipes.original.digest(state), **classify(state, root))
        value['manifest_sha256'] = recipes.original.digest(value)
        atomic_json(folder / 'original_state_before.json', state)
        atomic_json(path, value)
    for row in state['runs']:
        if row['case'] in value['replaced_cases']:
            if was_started(row, root / 'runs' / row['run_id']):
                raise ValueError('A previously unstarted replacement has since trained; inspect before proceeding')
            row.update(status=REPLACED, replacement_campaign=recipes.CAMPAIGN_ID,
                       replacement_run_id=recipes.resolve(server, row['case'])['run_id'])
    state.update(status='FINISH_STARTED_THEN_C30', replacement_campaign=recipes.CAMPAIGN_ID)
    atomic_json(root / 'campaign_state.json', state)
    return value


def next_original(state, manifest):
    allowed = set(manifest['finish_original_run_ids'])
    return next((r for r in state['runs'] if r['run_id'] in allowed and r['status'] not in TERMINAL), None)


def original_finished(server):
    manifest = read_cutover(server)
    state = json.loads((recipes.ORIGINAL_ROOT(server) / 'campaign_state.json').read_text())
    return next_original(state, manifest) is None
