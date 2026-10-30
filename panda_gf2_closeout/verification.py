"""Read-only winner, archive and retirement verification."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

CAMPAIGN = 'PANDA_S3_GF2_PSNR4508_FIXEDDONOR_20261030_v1'
RUN = 'PSFC_GF2_S3_GF2P_C60LR200_NOSOFT_R2_S261018102'
STEP = 47000
WINNER_SHA = '9027206526d343b5a82aa735057f13de7ed0778ec2ad8fadaec73d27760182d6'
SCHEMA = 'GF2_RETIRED_TARGET_MET_V1'
CONTROL = Path('work_dir/panda_gf2_closeout/s3')
CAMPAIGN_PATH = Path('work_dir/panda_gf2_psnr') / CAMPAIGN / 's3/GF2'
LAUNCHER = Path('work_dir/panda_gf2_psnr/launcher_s3')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def owned(root, raw):
    """Accept this checkout or the recorded Docker /workspace root only."""
    root = Path(root).resolve()
    path = Path(raw)
    if '..' in path.parts:
        raise ValueError('Traversal in artifact path')
    if str(path).startswith('/workspace/'):
        path = root / path.relative_to('/workspace')
    elif not path.is_absolute():
        path = root / path
    if root not in path.parents or path.is_symlink() or path.resolve() != path:
        raise ValueError('Artifact is outside exact checkout or linked')
    return path


def verify_winner(root):
    root = Path(root).resolve()
    directory = root / CAMPAIGN_PATH / 'runs' / RUN / 'attempt_01'
    path = owned(root, directory / 'models' / f'step_{STEP:07d}.pt')
    meta = read(path.with_suffix('.json'))
    cfg = read(directory / 'config.json')
    evpath = directory / f'evaluation_{STEP:07d}.json'
    ev = read(evpath)
    a, state = meta['artifact'], ev['state_before']
    expected = dict(run_id=RUN, campaign_id=CAMPAIGN, server='s3', dataset='GF2',
                    attempt=1, seed=261018102)
    if any(cfg.get(k) != v or state.get(k) != v for k, v in expected.items()):
        raise ValueError('Winner identity differs')
    if (meta['config'] != cfg or meta['evaluation'] != ev
            or state != ev['state_after'] or state != ev['rr_state'] or state != ev['fr_state']
            or state.get('completed_update') != STEP or a.get('completed_update') != STEP
            or a.get('identity') != state or a.get('model_state_sha256') != state.get('model_state_sha256')
            or owned(root, a['checkpoint_path']) != path
            or any(a.get(k) is not True for k in ('weights_saved', 'checkpoint_verified', 'campaign_resume_complete'))
            or any(ev.get('qualification', {}).get(k) is not True for k in
                   ('same_state_verified', 'metrics_complete', 'metrics_finite'))):
        raise ValueError('Winner lacks complete same-state evidence')
    metrics = {'HQNR': ev['fr']['metrics']['HQNR'], 'ERGAS': ev['rr']['metrics']['ERGAS'],
               'PSNR': ev['rr']['metrics']['PSNR']}
    if (any(type(x) not in (int, float) or not math.isfinite(x) for x in metrics.values())
            or metrics['HQNR'] < .9645 or metrics['ERGAS'] > .522 or metrics['PSNR'] < 45.08):
        raise ValueError('Winner fails the simultaneous three-metric target')
    if sha(path) != WINNER_SHA or a.get('file_sha256') != WINNER_SHA:
        raise ValueError('Winner checkpoint SHA differs from verified 47K result')
    return {'run_id': RUN, 'step': STEP, 'path': str(path.relative_to(root)),
            'sha256': WINNER_SHA, 'metrics': metrics, 'config': cfg,
            'evaluation_path': str(evpath.relative_to(root))}


def inspect(name):
    result = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True, check=True)
    values = json.loads(result.stdout)
    if len(values) != 1:
        raise ValueError('Ambiguous container')
    return values[0]


def authenticate_container(receipt, value, root):
    if (receipt.get('container') != 'panda-gf2-psnr4508-s3-v1'
            or value['Id'] != receipt.get('container_id') or value['Image'] != receipt.get('image')
            or receipt.get('args') != ['python', '-m', 'panda_gf2_psnr', 'run']
            or value['Config']['Cmd'] != receipt['args']):
        raise ValueError('GF2 trainer identity collision')
    output = str(Path(root).resolve() / 'work_dir/panda_gf2_psnr')
    mounts = [m for m in value['Mounts'] if m['Destination'] == '/workspace/work_dir/panda_gf2_psnr']
    if len(mounts) != 1 or mounts[0]['Type'] != 'bind' or mounts[0]['Source'] != output or mounts[0]['RW'] is not True:
        raise ValueError('GF2 trainer output mount differs')
    if value['State'].get('OOMKilled'):
        raise ValueError('GF2 trainer was OOM killed; investigate first')
    return value['State']


def verify_retirement(root, *, check_runtime=True):
    """Audit GF2 preservation and stop records independently of future queues."""
    root = Path(root).resolve()
    receipt = read(root / CONTROL / 'retirement.json')
    if (receipt.get('schema') != SCHEMA or receipt.get('server') != 's3'
            or receipt.get('campaign_id') != CAMPAIGN
            or receipt.get('status') != 'RETIRED_TARGET_MET'
            or receipt.get('winner', {}).get('sha256') != WINNER_SHA
            or receipt.get('winner', {}).get('step') != STEP):
        raise ValueError('GF2 retirement is missing or mismatched')
    state = receipt['docker_state']
    if (state.get('Running') is not False or state.get('Status') != 'exited'
            or state.get('OOMKilled') or state.get('ExitCode') not in (0, 1)):
        raise ValueError('Retirement lacks authenticated safe trainer exit')
    if check_runtime:
        current = inspect(receipt['trainer']['container'])
        current_state = authenticate_container(receipt['trainer'], current, root)
        if current_state != state:
            raise ValueError('Retired GF2 trainer was restarted or changed')
    if sha(root / CONTROL / 'intent.json') != receipt['intent_sha256']:
        raise ValueError('Retirement intent changed')
    for row in receipt['archive']:
        path = owned(root, row['archive_path'])
        if CONTROL not in path.relative_to(root).parents or sha(path) != row['sha256']:
            raise ValueError('Retirement archive changed')
    archived = {r['source_path']: r['sha256'] for r in receipt['archive']}
    for model in receipt['qualified_models']:
        if archived.get(model['path']) != model['sha256']:
            raise ValueError('A target-qualified model is absent from the archive')
    if not any(m['sha256'] == WINNER_SHA and m['step'] == STEP for m in receipt['qualified_models']):
        raise ValueError('Verified 47K model is not explicitly protected')
    campaign_state = root / CAMPAIGN_PATH / 'campaign_state.json'
    if sha(campaign_state) != receipt['campaign_state_sha256']:
        raise ValueError('Retired GF2 campaign changed')
    sheet = read(root / CONTROL / 'final_sheet_readback.json')
    if (sha(root / CONTROL / 'final_sheet_readback.json') != receipt['sheet_receipt_sha256']
            or sheet.get('readback_verified') is not True or sheet.get('winner_sha256') != WINNER_SHA
            or sheet.get('campaign_state_sha256') != receipt['campaign_state_sha256']
            or sheet.get('projection_sha256') != sha(root / CONTROL / 'sheet_projection.json')):
        raise ValueError('GF2 final Sheet readback is missing or changed')
    return receipt
