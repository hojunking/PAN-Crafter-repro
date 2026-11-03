"""Three local RGB donors and six full-method Student recipes per repetition."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
from panda_wv3_ablation.plan import digest, seeds

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/panda_wv3_main_rgb'
PLAN = json.loads((CONFIG / 'plan.json').read_text())
REFERENCES = json.loads((ROOT / 'config/panda_wv3_ablation/references.json').read_text())
CAMPAIGN_ID = PLAN['campaign_id']
REVISION = PLAN['method_revision']
CASES = {c['case_id']: c for c in PLAN['stage2_cases']}
HQNR_MIN, ERGAS_MAX = .9585, 2.040


def local_path(value):
    p = Path(value)
    return ROOT / p.relative_to('/workspace') if str(p).startswith('/workspace/') else (
        p if p.is_absolute() else ROOT / p)


def campaign_root(server):
    if server != 's1':
        raise ValueError('This independent main campaign belongs to s1 only')
    return ROOT / 'work_dir/panda_wv3_main_rgb' / CAMPAIGN_ID / server / 'WV3'


def resolve(server, case, repeat):
    campaign_root(server)
    if type(repeat) is not int or repeat not in (1, 2, 3):
        raise ValueError('Unregistered repetition')
    stage1 = case == 'S1_RGB_FULL'
    if stage1:
        row = dict(case_id=case, stage='stage1', alignment='LEARNED',
            lambda_abs=2., lambda_epsilon=.003, alpha=0., gamma=0.,
            horizon=50000, lr=1e-4, lr_aligner=1e-5, primary_step=50000,
            architecture='W112_D123_BASE', warp='BICUBIC_BORDER',
            alignment_target='SCHARR_OF_RGB_MEAN_GT', reference_bands=[4,2,1],
            validation_reference='RGB_MEAN_GT', donor_run_id=None)
    else:
        c = deepcopy(CASES[case])
        row = dict(case_id=case, stage='stage2', alignment='LEARNED_FROZEN',
            lambda_abs=0., lambda_epsilon=0., alpha=c['alpha'], gamma=c['gamma'],
            horizon=c['horizon'], lr=c['peak_lr'], lr_aligner=0.,
            primary_step=c['horizon']-1000, common_comparison_step=25000,
            architecture='W112_D123_UBIL', student_input='CORRECTED_DIFF',
            warp='LANCZOS4_REFLECTION', teacher_warp='BICUBIC_BORDER', scale=1.,
            weight='1+alpha*d_e+gamma*d_e*r_q',
            donor_run_id=resolve(server, 'S1_RGB_FULL', repeat)['run_id'])
    seed = (261016100 if stage1 else 261018100) + repeat
    row.update(server=server, campaign_id=CAMPAIGN_ID, method_revision=REVISION,
        dataset='WV3', repeat=repeat, replica=f'R{repeat}', seed=seed, stream_seeds=seeds(seed),
        run_id=f'WMRGB_WV3_S1_{case}_R{repeat}_S{seed}',
        experiment=f'WV3-MAIN-{case.replace("_", "-")} R{repeat} (RGB)',
        initialization='FRESH', weight_decay=.01, betas=[.9,.999], adam_eps=1e-8,
        batch=48, microbatch=12, warmup=100, min_lr=0., upsampler='interp23',
        epsilon_radius=3., precision='FP32',
        augmentation={'hflip_probability':1., 'vflip_probability':1., 'rot90':True},
        source_registry=REFERENCES['baseline']['source_registry'],
        source_registry_sha256=REFERENCES['baseline']['source_registry_sha256'],
        evaluation_steps=list(range(1000,row['horizon']+1,1000)),
        validation_steps=list(range(1000,50001,1000)) if stage1 else [],
        checkpoint_selection='VALIDATION_RRE50_RRE90_ERGAS_STEP' if stage1 else
                             'HQNR_QUALIFIED_MIN_ERGAS_ELSE_MAX_HQNR',
        inference_tta=False, beta=0., rotation_consistency=0., weight_averaging=False, attempt=1)
    row['config_sha256'] = digest(row)
    return row


def queue(server):
    campaign_root(server)
    return [resolve(server,case,repeat) for repeat in (1,2,3)
            for case in ('S1_RGB_FULL',*CASES)]


def validate(cfg):
    if cfg != resolve(cfg.get('server'),cfg.get('case_id'),cfg.get('repeat')):
        raise ValueError('Configuration differs from the registered main recipe')


def catalog(server):
    stream=io.StringIO()
    fields=['order','server','stage','case_id','repeat','seed','horizon','lr','lr_aligner',
            'alpha','gamma','beta','lambda_abs','lambda_epsilon','donor_run_id','experiment']
    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
    for i,cfg in enumerate(queue(server),1):
        writer.writerow(dict(order=i,**{k:cfg[k] for k in fields if k!='order'}))
    return stream.getvalue()
