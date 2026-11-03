"""Fresh s3-only RGB Stage1 components, round-major, with a new full control."""
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/panda_wv3_ablation_rgb'
PLAN = json.loads((CONFIG / 'plan.json').read_text())
REFERENCES = json.loads((ROOT / 'config/panda_wv3_ablation/references.json').read_text())
CAMPAIGN_ID = PLAN['campaign_id']
REVISION = PLAN['method_revision']
CASES = {c['case_id']: c for c in PLAN['cases']}
HQNR_MIN, ERGAS_MAX = .9585, 2.040


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def local_path(value):
    p = Path(value)
    return ROOT / p.relative_to('/workspace') if str(p).startswith('/workspace/') else (
        p if p.is_absolute() else ROOT / p)


def campaign_root(server):
    if server != 's3':
        raise ValueError('RGB campaign is registered for s3 Stage1 only')
    return ROOT / 'work_dir/panda_wv3_ablation_rgb' / CAMPAIGN_ID / server / 'WV3'


def seeds(seed):
    # Same hash labels as donor30.plan.stream_seeds; no case/server in the seed.
    labels = ('A_init', 'T_init', 'S_init', 'rr_data', 'rr_augmentation',
              'rr_epsilon', 'rr_band_augmentation')
    return {label: 1 + int.from_bytes(hashlib.sha256(
        f'donor30:{seed}:{label}'.encode()).digest()[:8], 'big') % (2**63-1)
        for label in labels}


def resolve(server, case, repeat):
    campaign_root(server)
    row = deepcopy(CASES[case])
    if row['server'] != server or type(repeat) is not int or repeat not in (1, 2, 3):
        raise ValueError('Case belongs to another server or unregistered repetition')
    stage1 = row['stage'] == 'stage1'
    seed = (261016100 if stage1 else 261018100) + repeat
    row.update(campaign_id=CAMPAIGN_ID, method_revision=REVISION, dataset='WV3',
        repeat=repeat, replica=f'R{repeat}', seed=seed, stream_seeds=seeds(seed),
        experiment=f"WV3-ABL-{case.replace('_', '-')} R{repeat} (RGB)",
        run_id=f'WABLRGB_WV3_{server.upper()}_{case}_R{repeat}_S{seed}',
        lr=1e-4, lr_aligner=1e-5 if stage1 and row['alignment']=='LEARNED' else 0.,
        weight_decay=.01, betas=[.9, .999], adam_eps=1e-8,
        batch=48, microbatch=12, warmup=100, min_lr=0.,
        upsampler='interp23', epsilon_radius=3., precision='FP32',
        scharr_search_axis_bound=5., scharr_fine_step=.25, scharr_roi_margin=8,
        augmentation={'hflip_probability':1., 'vflip_probability':1., 'rot90':True},
        source_registry=REFERENCES['baseline']['source_registry'],
        source_registry_sha256=REFERENCES['baseline']['source_registry_sha256'],
        evaluation_steps=list(range(1000, row['horizon']+1, 1000)),
        validation_steps=list(range(1000, 50001, 1000)) if stage1 else [],
        checkpoint_selection='VALIDATION_RRE50_RRE90_ERGAS_STEP' if stage1 else
                             'HQNR_QUALIFIED_MIN_ERGAS_ELSE_MAX_HQNR',
        inference_tta=False, beta=0., rotation_consistency=0., attempt=1)
    if not stage1:raise ValueError('Only Stage1 is registered in the RGB campaign')
    row['config_sha256'] = digest(row)
    return row


def queue(server):
    campaign_root(server)
    # ALL case types at R1, then ALL at R2, then ALL at R3, including diagnostics.
    return [resolve(server, c, r) for r in (1, 2, 3)
            for c, row in CASES.items() if row['server'] == server]


def validate(cfg):
    if cfg != resolve(cfg.get('server'), cfg.get('case_id'), cfg.get('repeat')):
        raise ValueError('Configuration differs from the registered scientific recipe')


def catalog(server):
    stream = io.StringIO()
    keys = ['order','server','stage','case','repeat','seed','horizon','primary_step','lr',
            'lambda_abs','lambda_epsilon','alpha','gamma','beta','input','experiment']
    writer = csv.DictWriter(stream, fieldnames=keys, lineterminator='\n'); writer.writeheader()
    for i,c in enumerate(queue(server),1):
        writer.writerow(dict(order=i,server=server,stage=c['stage'],case=c['case_id'],
            repeat=c['repeat'],seed=c['seed'],horizon=c['horizon'],primary_step=c['primary_step'],
            lr=c['lr'],lambda_abs=c.get('lambda_abs',''),lambda_epsilon=c.get('lambda_epsilon',''),
            alpha=c.get('alpha',''),gamma=c.get('gamma',''),beta=0,
            input=c.get('student_input','PAN+MS'),experiment=c['experiment']))
    return stream.getvalue()
