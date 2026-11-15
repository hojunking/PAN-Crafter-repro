"""One existing local RGBN donor selected from 40/48/50K; one Student/server."""
import csv
import io
from panda_qb_s3_parallel import plan as parallel
from panda_unified_campaign import plan as unified

ROOT = unified.ROOT
CONFIG = ROOT / 'config/panda_qb_lateval'
CAMPAIGN_ID = 'PANDA_QB_S23_LATEVAL_20261115_v1'
REVISION = 'QB_RGBN_L4_COMMON_W128_UBIL_LATEVAL_PRODUCT_FULL50_V1'
CASE = 'LV404850_LR100_G025'
CANDIDATE_STEPS = (40000, 48000, 50000)
SELECTOR = 'VAL_RRE50_RRE90_VAL_ERGAS_EARLIER_OF_40K_48K_50K'
digest, local_path, seeds = unified.digest, unified.local_path, unified.seeds


def sensor_for(server):
    if server not in ('s2', 's3'):
        raise ValueError('QB on physical s2 or s3 only')
    return 'QB'


def targets(sensor):
    if sensor != 'QB':
        raise ValueError('QB only')
    return {'HQNR_min': .9315, 'ERGAS_max': 3.5, 'PSNR_min': None}


def campaign_root(server):
    return ROOT / 'work_dir/panda_qb_lateval' / CAMPAIGN_ID / server / sensor_for(server)


def donor_config(server):
    sensor_for(server)
    return (unified.resolve('s2', 'S1_RGBN_FULL', 1) if server == 's2'
            else parallel.donor_config('s3'))


def donor_root(server):
    sensor_for(server)
    return unified.campaign_root('s2') if server == 's2' else parallel.campaign_root('s3')


def baseline(server):
    sensor_for(server)
    if server == 's2':
        from panda_sensor_main50 import plan as old
        return old, old.resolve('s2', 'Q1_BASE_LR100_G025')
    return parallel, parallel.resolve('s3', 'P1_LR100_G025')


def resolve(server, case=CASE, repeat=1):
    sensor_for(server)
    if case != CASE or type(repeat) is not int or repeat != 1:
        raise ValueError('Exactly one paired-seed Stage2 case per server')
    # The same method/optimizer/data stream as Q1/P1; only donor selection changes.
    _, cfg = baseline(server)
    dc = donor_config(server)
    cfg.update(campaign_id=CAMPAIGN_ID, method_revision=REVISION, case_id=case,
        run_id=f'QBLV_QB_{server.upper()}_{case}_R1_S261018101',
        experiment=f'QB-LATEVAL-40-48-50K-LR100-G025 {server} R1 (RGBN)',
        seed=261018101, stream_seeds=seeds(261018101),
        stage='stage2', donor_run_id=dc['run_id'], donor_campaign_id=dc['campaign_id'],
        donor_selection={'policy': SELECTOR, 'candidate_steps': list(CANDIDATE_STEPS),
                         'uses_test_for_selection': False, 'performance_admission_gate': False},
        horizon=50000, primary_step=50000, common_comparison_step=50000,
        evaluation_steps=list(range(1000, 50001, 1000)), validation_steps=[],
        early_stop={'enabled': False, 'reason': 'FULL50K_NO_METRIC_STOP'},
        reuse_existing=True, experiment_kind='MAIN_DONOR_SELECTION_COMPARISON',
        reference_repeats=1, input_alignment=True, student_input='CORRECTED',
        weight_variant='PRODUCT', weight='1+0.25*d_e*r_q; r_q=0 if q=0',
        training_support=cfg['mask_policy'], evaluation_reference='SELECTED_LOCAL_DONOR_CORRECTED_PAN',
        official_selection_reference='CORRECTED_PAN',
        checkpoint_selection='CORRECTED_H_QUALIFIED_MIN_E_ELSE_MAX_H',
        baseline_variant='LOCAL_LATE_VALIDATION_SELECTION_ONLY',
        lr=1e-4, gamma=.25, alpha=0., beta=0.)
    cfg.pop('fixed_donor', None)
    cfg.pop('config_sha256')
    cfg['config_sha256'] = digest(cfg)
    return cfg


def queue(server):
    return [resolve(server)]


def validate(cfg):
    if cfg != resolve(cfg.get('server'), cfg.get('case_id'), cfg.get('repeat')):
        raise ValueError('Registered QB LATEVAL config required')


def assert_method(cfg):
    expected = dict(sensor='QB', bands=4, width=128, depth=[1,2,3],
        sampling_variant='UBIL', stage1_backbone='UBIL', reference_mode='RGBN',
        reference_bands=[2,1,0,3], stage1_spatial_bands=[2,1,0,3], stage2_q_bands=[2,1,0,3],
        scale=1., upsampler='interp23', warp='LANCZOS4_REFLECTION_ALL_PATHS',
        teacher_warp='LANCZOS4_REFLECTION', stage1_and_teacher_warp='LANCZOS4',
        epsilon_injection_warp='LANCZOS4', epsilon_radius=3.)
    from panda_sensor_eq40.contract import MASK2_POLICY, WARP_CONTRACT
    expected.update(mask_policy=MASK2_POLICY, warp_contract=WARP_CONTRACT)
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError('Method differs: '+key)
    if cfg['stage'] == 'stage1' and (cfg['lambda_abs'], cfg['lambda_epsilon'], cfg['lr'], cfg['lr_aligner']) != (4.,.03,1e-4,1e-5):
        raise ValueError('Wrong Stage1 coefficients/LRs')
    if cfg['stage'] == 'stage2' and (cfg['student_input'],cfg['alpha'],cfg['beta'],cfg['gamma']) != ('CORRECTED',0.,0.,.25):
        raise ValueError('Wrong Stage2 input/loss')


def catalog(server):
    c=resolve(server); d=donor_config(server)
    out=io.StringIO(); keys=['server','sensor','stage1_new_runs','donor_run_id','candidate_steps',
        'case_id','horizon','width','lr','gamma','seed']
    w=csv.DictWriter(out,fieldnames=keys,lineterminator='\n');w.writeheader()
    w.writerow({**{k:c[k] for k in keys if k in c},'stage1_new_runs':0,
        'donor_run_id':d['run_id'],'candidate_steps':'40000/48000/50000'})
    return out.getvalue()
