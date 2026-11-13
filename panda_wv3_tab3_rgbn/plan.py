"""Fixed-50K Table 3: five paired configurations, three round-major repeats."""
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/panda_wv3_tab3_rgbn'
PLAN = json.loads((CONFIG / 'plan.json').read_text())
CAMPAIGN_ID, REVISION = PLAN['campaign_id'], PLAN['method_revision']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def seeds(seed):
    return {key: 1 + int.from_bytes(hashlib.sha256(f'donor30:{seed}:{key}'.encode()).digest()[:8], 'big') % (2**63 - 1)
            for key in ('A_init', 'T_init', 'S_init', 'rr_data', 'rr_augmentation', 'rr_epsilon', 'rr_band_augmentation')}


def local_path(value):
    p = Path(value)
    return ROOT / p.relative_to('/workspace') if str(p).startswith('/workspace/') else p if p.is_absolute() else ROOT / p


def sensor_for(server):
    if server != 's3':
        raise ValueError('This campaign is registered only for s3 WV3')
    return 'WV3'


def targets(sensor):
    if sensor != 'WV3':
        raise ValueError('WV3 only')
    return {'HQNR_min': .9585, 'ERGAS_max': 2.040}  # Diagnostic only; never selects or drops a run.


def reference_bands(sensor, mode):
    if (sensor, mode) != ('WV3', 'RGBN'):
        raise ValueError('WV3 RGBN is required throughout this campaign')
    return [4, 2, 1, 6]


def campaign_root(server):
    return ROOT / 'work_dir/panda_wv3_tab3_rgbn' / CAMPAIGN_ID / server / sensor_for(server)


def stage1_cases(server):
    sensor_for(server)
    return [c['id'] for c in PLAN['cases']]


def resolve(server, case, repeat):
    sensor = sensor_for(server)
    if type(repeat) is not int or repeat not in PLAN['repeats']:
        raise ValueError('Unregistered repeat')
    cases = {c['id']: c for c in PLAN['cases']}
    if case not in cases:
        raise ValueError('Unregistered Table 3 case')
    row = cases[case]
    seed = 261016100 + repeat
    from .contract import MASK2_POLICY, WARP_CONTRACT, warp_manifest
    cfg = dict(case_id=case, table_row=row['table_row'], server=server, sensor=sensor, dataset=sensor, bands=8,
        campaign_id=CAMPAIGN_ID, method_revision=REVISION, stage='stage1',
        repeat=repeat, replica=f'R{repeat}', seed=seed, stream_seeds=seeds(seed),
        run_id=f'TAB3_WV3_S3_{case}_R{repeat}_S{seed}',
        experiment=f'WV3-TAB3-{case.replace("_", "-")} R{repeat} (RGBN)',
        alignment=row['alignment'], alignment_target='SCHARR_OF_RGBN_MEAN_GT',
        reference_mode='RGBN', reference_bands=[4,2,1,6],
        stage1_spatial_bands=[4,2,1,6], stage1_spatial_weights=[.25]*4,
        stage2_q_bands=[4,2,1,6], stage2_executed=False,
        validation_reference='RGBN_MEAN_GT', test_rre_reference='RGBN_GT_RR_RGBN_STORED_LMS_FR',
        lambda_abs=row['lambda_abs'], lambda_epsilon=row['lambda_epsilon'],
        alpha=0., gamma=0., beta=0., horizon=50000, primary_step=50000, common_comparison_step=50000,
        lr=1e-4, lr_aligner=1e-5 if row['alignment']=='LEARNED' else 0.,
        width=112, depth=[1,2,3], architecture='W112_D123_UBIL', stage1_backbone='UBIL',
        sampling_variant='UBIL', student_input='CORRECTED', initialization='FRESH',
        weight_decay=.01, betas=[.9,.999], adam_eps=1e-8, batch=48, microbatch=12, warmup=100, min_lr=0.,
        upsampler='interp23', epsilon_radius=3., precision='FP32', scale=1.,
        augmentation={'hflip_probability':1., 'vflip_probability':1., 'rot90':True},
        warp='LANCZOS4_REFLECTION_ALL_PATHS', teacher_warp='LANCZOS4_REFLECTION',
        warp_contract=WARP_CONTRACT, warp_manifest=warp_manifest(), mask_policy=MASK2_POLICY,
        stage1_and_teacher_warp='LANCZOS4', epsilon_injection_warp='LANCZOS4',
        validation_rr_margin=19, test_rr_rre_margin=19, test_fr_rre_margin=24,
        stage1_reuses_existing_weights=False, inference_tta=False, rotation_consistency=0.,
        weight_averaging=False, attempt=1, weight='UNWEIGHTED_RECONSTRUCTION',
        baseline_variant='FRESH_UBIL_RGBN_L4_COMMON_SCHARR_VALID_MASK',
        source_registry='ablr2/sensor_sources.json',
        source_registry_sha256=hashlib.sha256((ROOT/'ablr2/sensor_sources.json').read_bytes()).hexdigest(),
        checkpoint_selection=PLAN['checkpoint_selection'],
        evaluation_steps=list(PLAN['evaluation_steps']), validation_steps=list(PLAN['evaluation_steps']),
        publication_steps=list(PLAN['publication_steps']), early_stop={'enabled':False}, donor_run_id=None,
        scharr_matcher={'bound_per_axis':5., 'coarse_step':1., 'fine_step':.25, 'fine_radius':1.,
            'reference':'OBSERVED_STORED_LMS_RGBN_MEAN_NO_GT', 'score':'MEAN_ABS_SCHARR_COMMON_VALID_MASK',
            'cache':'EXACT_PAN_AND_LMS_CONTENT_ONLY_NO_APPROXIMATION'})
    cfg['config_sha256'] = digest(cfg)
    return cfg


def queue(server):
    return [resolve(server, c, r) for r in PLAN['repeats'] for c in stage1_cases(server)]


def validate(cfg):
    if cfg != resolve(cfg.get('server'), cfg.get('case_id'), cfg.get('repeat')):
        raise ValueError('Unregistered Table 3 configuration')


def catalog(server):
    stream = io.StringIO()
    keys = ['order','server','table_row','case_id','repeat','seed','alignment','lambda_abs','lambda_epsilon',
            'width','horizon','lr','lr_aligner','reference_mode','experiment']
    writer = csv.DictWriter(stream, fieldnames=keys, lineterminator='\n'); writer.writeheader()
    for i,cfg in enumerate(queue(server),1):
        writer.writerow(dict(order=i, **{k:cfg[k] for k in keys if k!='order'}))
    return stream.getvalue()
