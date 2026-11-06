import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def local(path):
    path = Path(path)
    if path.is_absolute() and path.parts[:2] == ('/', 'workspace'):
        return ROOT / path.relative_to('/workspace')
    return path if path.is_absolute() else ROOT / path


def source_namespace(cfg):
    """Install the original registry in this dedicated process before inference/training."""
    campaign = cfg['campaign_id']
    if campaign == 'PANDA_S3_GF2_PSNR4508_FIXEDDONOR_20261030_v1':
        from panda_gf2_psnr.adapter import install
        install()
        return 'panda_sensor_focus'
    if campaign == 'PANDA_S3_GF2_W128_FIXEDDONOR_20261106_v1':
        from panda_gf2_w128.adapter import install
        install()
        return 'panda_sensor_focus'
    for package, prefix in [('panda_sensor_focus', 'PANDA_S13_SENSOR_FOCUS_'),
                            ('panda_sensor_refine', 'PANDA_S13_SENSOR_REFINE_'),
                            ('panda_sensor5', 'PANDA_S13_SENSOR5_RGB_L4_')]:
        if campaign.startswith(prefix):
            return package
    raise ValueError('Unsupported original campaign: ' + campaign)


def checkpoint_evidence(run, step):
    run = Path(run).resolve(strict=True)
    cfg = read(run / 'config.json')
    ev = read(run / f'evaluation_{step:07d}.json')
    ck = run / 'models' / f'step_{step:07d}.pt'
    side = read(ck.with_suffix('.json'))
    identity = ev['state_before']
    if cfg['phase'] != 'stage2' or cfg['dataset'] not in ('QB', 'GF2'):
        raise ValueError('Only four-band QB/GF2 Stage2 checkpoints are supported')
    if (identity != ev['state_after'] or identity != ev['rr_state'] or identity != ev['fr_state']
            or identity != side['artifact']['identity'] or identity['run_id'] != cfg['run_id']
            or identity['completed_update'] != step or identity['dataset'] != cfg['dataset']
            or identity['server'] != cfg['server'] or side['config'] != cfg
            or side['artifact']['completed_update'] != step):
        raise ValueError('Mixed checkpoint/config/evaluation identity')
    if sha(ck) != side['artifact']['file_sha256']:
        raise ValueError('Checkpoint file SHA differs from its original sidecar')
    for split in ('rr', 'fr'):
        if len(ev[split]['per_scene']) != 20:
            raise ValueError('The original complete RR20/FR20 evaluation is required')
        for i, row in enumerate(ev[split]['per_scene']):
            if row['scene_id'] != f'{cfg["dataset"]}:{split}:{i}' or not row['raw_prediction_sha256']:
                raise ValueError('Wrong scene order or missing original output SHA')
    return cfg, ev, ck, side
