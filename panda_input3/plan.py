"""Existing local RGBN donors, three input layouts, one paired seed per sensor."""
import csv
import io
import json
from panda_eq40_zeroq import plan as main
from panda_sensor_main50 import plan as sensor_main
from panda_wv3_s2abl5 import plan as ablation
from panda_unified_campaign.plan import seeds

ROOT = main.ROOT
CONFIG = ROOT / 'config/panda_input3'
CAMPAIGN_ID = 'PANDA_S123_INPUT3_20261115_v1'
REVISION = 'RGBN_FIXED_DONOR_RAW_DUAL_DIFF_PRODUCT_ONLY_V1'
digest, local_path = main.digest, main.local_path
sensor_for, targets = main.sensor_for, main.targets
CASES = {'A_RAW': 'RAW', 'B_RAW_CORRECTED': 'RAW_CORRECTED',
         'C_CORRECTED_DIFF': 'CORRECTED_DIFF'}
FIXED = json.loads((CONFIG / 'donors.json').read_text())


def campaign_root(server):
    return ROOT / 'work_dir/panda_input3' / CAMPAIGN_ID / server / sensor_for(server)


def donor_config(server):
    return main.donor_config('s1',1) if server=='s1' else sensor_main.donor_config(server)


def donor_root(server):
    return main.donor_root(server)


def source_config(server):
    if server=='s1':return main.queue(server)[0]
    return sensor_main.resolve(server,'Q1_BASE_LR100_G025' if server=='s2' else 'G3_LR250_G025')


def resolve(server,case,repeat=1):
    sensor=sensor_for(server)
    if case not in CASES or type(repeat) is not int or repeat!=1:
        raise ValueError('Three input layouts, one paired Student seed each')
    cfg=source_config(server);fixed=FIXED[sensor];variant=CASES[case]
    seed=261018101;horizon=40000 if server=='s1' else 50000
    cfg.update(campaign_id=CAMPAIGN_ID,method_revision=REVISION,case_id=case,
        run_id=f'INPUT3_{sensor}_{server.upper()}_{case}_R1_S{seed}',
        experiment=f'{sensor}-INPUT3-{case.replace("_","-")} R1 (RGBN)',
        stage='stage2',repeat=1,replica='R1',seed=seed,stream_seeds=seeds(seed),
        horizon=horizon,primary_step=horizon,common_comparison_step=horizon,
        evaluation_steps=list(range(1000,horizon+1,1000)),validation_steps=[],
        schedule={'name':'WARMUP_COSINE_HOLD','decay_end':horizon,'min_lr':0.},
        alpha=0.,beta=0.,gamma=.25,weight_variant='PRODUCT',
        weight='1+0.25*d_e*r_q; r_q=0 if q=0',
        q_zero_policy='EXACT_ZERO_RANK_ZERO_POSITIVE_RIGHT_ECDF_V1',
        student_input=variant,student_pan_channels=1 if variant=='RAW' else 2,
        input_alignment=variant!='RAW',alignment='REFERENCE_ONLY_FROZEN' if variant=='RAW' else 'LEARNED_FROZEN',
        donor_run_id=fixed['run_id'],donor_step=fixed['step'],fixed_donor=fixed,
        donor_campaign_id=donor_config(server)['campaign_id'],
        donor_selection={'policy':'EXISTING_MAIN_FIXED_DONOR_ALL_THREE_INPUTS','uses_test_for_selection':True},
        early_stop={'enabled':False,'reason':'FIXED_HORIZON_NO_METRIC_STOP'},
        official_selection_reference='CORRECTED_PAN',
        checkpoint_selection='CORRECTED_H_QUALIFIED_MIN_E_ELSE_MAX_H',
        fr_references=['ORIGINAL_PAN','CORRECTED_PAN'],
        evaluation_reference='SAME_FIXED_LOCAL_MAIN_DONOR_FOR_ALL_THREE_INPUTS',
        training_support=cfg['mask_policy'],reuse_existing=False,reference_repeats=1,
        baseline_variant='INPUT_ONLY_FIXED_TEACHER_Q_MASK_AND_OPTIMIZER',
        experiment_kind='STUDENT_INPUT_COMPARISON',
        guidance_pan='CORRECTED_PAN_FOR_ALL_CASES')
    cfg.pop('config_sha256')
    cfg['config_sha256']=digest(cfg)
    return cfg


def queue(server):return [resolve(server,c) for c in CASES]


def validate(cfg):
    if cfg!=resolve(cfg.get('server'),cfg.get('case_id'),cfg.get('repeat')):
        raise ValueError('Registered INPUT3 recipe required')


def catalog(server):
    out=io.StringIO();keys=['case_id','sensor','student_input','horizon','lr','width','gamma','seed','donor_run_id','donor_step']
    w=csv.DictWriter(out,fieldnames=keys,lineterminator='\n');w.writeheader()
    for cfg in queue(server):w.writerow({k:cfg[k] for k in keys})
    return out.getvalue()
