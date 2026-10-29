"""Registered fixed-donor Student followups; source identities remain original."""
from copy import deepcopy
from panda_sensor_refine.contract import (LANES, BUDGET, POLICY, SOURCE_FIELDS,
    TRAINING_SCHEDULE_FIELDS, SCALE_POLICIES, SAMPLING_VARIANTS, INPUT_VARIANTS,
    file_identity, STAGE1_SELECTION, lane_for_sensor, policy, target_gap,
    input_manifest as _input, sampling_manifest as _sampling, WARP_MANIFEST as _warp)
CAMPAIGN_ID = 'PANDA_S13_SENSOR_FOCUS_FIXEDDONOR_20261029_v1'
METHOD_REVISION = 'PANDA-SENSOR-FOCUS-FIXEDDONOR-RGB-L4-DIFFUBIL-HARDONLY-v1'
RUN_PREFIX = 'PSFC'
NEXT_CAMPAIGN_ID = CAMPAIGN_ID
WARP_MANIFEST = {**deepcopy(_warp), 'schema':'PANDA_SENSOR_FOCUS_UNIFIED_WARP_V1'}
SOURCE_DONORS = {
 's1': {'package':'panda_s1loss9','campaign_id':'PANDA_S13_S1LOSS9_RGB_L4_STAGE12_20261026_v1',
        'case':'S1_A200_E030','selected_step':44000,'lambda_abs':2.,'lambda_epsilon':.03,
        'epsilon_radius_rr':3.,'checkpoint_sha256':'7fa4491f0c0e7b8c5d559263c50fff18f84abd7a820852a0d24666928191ed34',
        'receipt_sha256':'bea177873b2611822c60f9952c86f3a9f4982832142a7295c1053974e50e748b'},
 's3': {'package':'panda_sensor5','campaign_id':'PANDA_S13_SENSOR5_RGB_L4_STAGE12_20261026_v2',
        'case':'RGB_MATURE_U23','selected_step':42000,'lambda_abs':2.,'lambda_epsilon':.003,
        'epsilon_radius_rr':3.,'checkpoint_sha256':'937ac56b559b76a151df661bbb853119512d6f9cab283346c61186db27ff7478',
        'receipt_sha256':'52626a232557267d6852bc2b49bde8fc05aa6ccd5c1fc759a79fbe1101bedf95'},
}

def input_manifest(variant,width=112,bands=4):
    if type(width) is not int or width not in (112,128):
        raise ValueError('Only registered W112/W128 Students are allowed')
    value=_input(variant,112,bands)
    value.update(schema='PANDA_SENSOR_FOCUS_STUDENT_INPUT_V1',additional_parameters=9*width if variant=='CORRECTED_DIFF' else 0)
    return value

def sampling_manifest(variant,input_variant='CORRECTED',width=112,bands=4):
    value=_sampling(variant,input_variant,112,bands)
    up=variant in ('UBIL','DAVG_UBIL')
    value.update(schema='PANDA_SENSOR_FOCUS_STUDENT_SAMPLING_V1',student_width=width,
        student_input=input_manifest(input_variant,width,bands),
        student_parameters=(220 if up else 210)*width*width+148*width+4+input_manifest(input_variant,width,bands)['additional_parameters'])
    return value

def _fresh(sensor,updates=30000,lr=1e-4,alpha=.25,gamma=.5,width=112):
    server='s1' if sensor=='QB' else 's3'; donor=SOURCE_DONORS[server]
    return dict(sensor=sensor,policy_id='S100',scale=1.,alpha=alpha,gamma=gamma,
        lr_S=lr,total_updates=updates,student_width=width,warmup_updates=100,min_lr=0.,
        inherited_updates=0,schedule_kind='WARMUP_COSINE',sampling_variant='UBIL',
        input_variant='CORRECTED_DIFF',**{k:None for k in SOURCE_FIELDS},
        donor_case=donor['case'],lambda_abs=donor['lambda_abs'],lambda_epsilon=donor['lambda_epsilon'],
        epsilon_radius_rr=3.)
CASES={
 'QB_W128':_fresh('QB',width=128),
 'QB_A0125':_fresh('QB',alpha=.125),
 'QB_H050':_fresh('QB',alpha=.125,gamma=.25),
 'QB_G000':_fresh('QB',gamma=0.),
 'GF2_C60':_fresh('GF2',updates=60000),
 'GF2_C60LR150':_fresh('GF2',updates=60000,lr=1.5e-4),
 'GF2_C60W128':_fresh('GF2',updates=60000,width=128),
}
REGISTERED_RUNS={
 's1':(('QB_W128','R1'),('QB_W128','R2'),('QB_A0125','R1'),('QB_A0125','R2'),('QB_H050','R2'),('QB_G000','R1')),
 's3':(('GF2_C60','R2'),('GF2_C60LR150','R1'),('GF2_C60LR150','R2'),('GF2_C60W128','R1'),('GF2_C60W128','R2')),
}
DONORS=()
ALL_DONORS=tuple(d['case'] for d in SOURCE_DONORS.values())
LOSS_PAIRS=tuple((d['case'],d['lambda_abs'],d['lambda_epsilon']) for d in SOURCE_DONORS.values())
CONTROL_CAMPAIGN=CAMPAIGN_ID
CONTROL_CASE='FIXED_ORIGINAL_DONOR'

def validate_loss_pair_config(cfg):
    server=cfg.get('server'); source=SOURCE_DONORS.get(server)
    if (source is None or cfg.get('phase')!='stage2' or cfg.get('dataset')!=LANES[server]['dataset']
            or (cfg.get('case'),cfg.get('replica')) not in REGISTERED_RUNS[server]):
        raise ValueError('Only registered sensor-specific fresh Students are allowed')
    expected={k:source[k] for k in ('lambda_abs','lambda_epsilon')}
    if (cfg.get('donor_version')!=source['case'] or cfg.get('stage1_loss_parameters')!=expected
            or cfg.get('epsilon_radius_rr')!=3. or cfg.get('epsilon_distribution')!='UNIFORM_AREA_DISK'
            or any(cfg.get(k)!=0. for k in ('lambda_abs','lambda_epsilon','lambda_eps_RR'))):
        raise ValueError('Frozen original donor parameters or Stage2-only loss changed')
    return expected

def validate_source_teacher(binding):
    source=next((s for s in SOURCE_DONORS.values() if s['campaign_id']==binding.get('source_campaign_id')),None)
    teacher=binding.get('band30_teacher',{})
    if (source is None or teacher!=binding.get('mature30_teacher')
            or teacher.get('campaign_id')!=source['campaign_id']
            or binding.get('donor_step')!=source['selected_step']
            or binding.get('donor_checkpoint_sha256')!=source['checkpoint_sha256']):
        raise ValueError('Only exact selected original source checkpoints are allowed')
    sensor='QB' if source is SOURCE_DONORS['s1'] else 'GF2'
    if binding.get('sensor')!=sensor:
        raise ValueError('Frozen source sensor mismatch')
    required={'upsampler':'interp23','aligner_outputs':2,'pan_channels':1,'alignment_target':'SCHARR_OF_RGB_MEAN_GT'}
    if any(teacher.get(k)!=v for k,v in required.items()):
        raise ValueError('RGB/L4 global2 source required')
    # Sensor5 v2's authentic Teacher manifest predates coefficient fields.
    # Its immutable checkpoint SHA + original loader validates the source config;
    # never add fields to impersonate a newer campaign binding.
    if sensor=='QB' and any(teacher.get(k)!=source[k] for k in ('lambda_abs','lambda_epsilon')):
        raise ValueError('Original QB donor coefficient binding differs')
    return teacher

def validate_donor_loss_pair(cfg,binding):
    expected=validate_loss_pair_config(cfg); validate_source_teacher(binding)
    if binding['source_campaign_id']!=SOURCE_DONORS[cfg['server']]['campaign_id']:
        raise ValueError('Wrong donor source for Student lane')
    return expected
