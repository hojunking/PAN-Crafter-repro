"""Finite QB_S2 recipes; unchanged scientific operators and selection rules."""
from copy import deepcopy
from panda_s1loss9.contract import (file_identity, STAGE1_SELECTION, POLICY,
    SOURCE_FIELDS, TRAINING_SCHEDULE_FIELDS, SCALE_POLICIES, SAMPLING_VARIANTS,
    INPUT_VARIANTS, WARP_MANIFEST, input_manifest as _input, sampling_manifest as _sampling)
from . import registry as G
CAMPAIGN_ID, METHOD_REVISION = G.CAMPAIGN_ID, G.METHOD_REVISION
NEXT_CAMPAIGN_ID = CONTROL_CAMPAIGN = CAMPAIGN_ID
RUN_PREFIX='PQS2'
CONTROL_CASE='QB6_S2_A200_E030_D101'
LANES={s:dict(dataset='QB',hqnr_min=.9315,ergas_max=3.5,peak=2047,
              train_count=17139,val_count=1905,long_updates=60000) for s in G.RUNS}
BUDGET=dict(total_seconds=None,closeout_seconds=600,safety_factor=1.08,
            calibration_reserve_seconds=600,scope='COMPLETE_REGISTERED_FOCUS_QUEUE_NO_DEADLINE',
            restart='EXACT_CHECKPOINT_RESUME_NO_REINITIALIZATION')
DONORS=tuple(dict.fromkeys(r['case'] for rows in G.PLAN['queues'].values() for r in rows if r['phase']=='stage1'))
ALL_DONORS=DONORS
LOSS_PAIRS=(('A200_E030',2.,.03),('A400_E030',4.,.03),('A300_E030',3.,.03),('A400_E060',4.,.06))
PAIR_PARAMETERS={k:dict(lambda_abs=a,lambda_epsilon=e) for k,a,e in LOSS_PAIRS}

def stage1_pair(case):
    if case not in ALL_DONORS: raise ValueError('Unregistered Stage1 branch')
    return G.job('s2', case)['loss_pair']

def lane_for_sensor(sensor):
    if sensor!='QB': raise ValueError('QB_S2 only trains QB')
    return LANES['s2']

def policy(sensor):
    lane_for_sensor(sensor); return POLICY

def target_gap(sensor,hqnr,ergas):
    lane=lane_for_sensor(sensor)
    return max(0.,lane['hqnr_min']-hqnr),max(0.,ergas-lane['ergas_max'])

def input_manifest(variant,width=112,bands=4):
    if width not in (112,128): raise ValueError('Unregistered width')
    value=_input(variant,112,bands)
    value['additional_parameters']=9*width if variant=='CORRECTED_DIFF' else 0
    return value

def sampling_manifest(variant,input_variant='CORRECTED',width=112,bands=4):
    value=_sampling(variant,input_variant,112,bands)
    value.update(student_width=width,student_input=input_manifest(input_variant,width,bands),
        student_parameters=(220 if variant in ('UBIL','DAVG_UBIL') else 210)*width*width+148*width+4+input_manifest(input_variant,width,bands)['additional_parameters'])
    return value

CASES={}
for rows in G.PLAN['queues'].values():
    for r in rows:
        if r['phase']!='stage2': continue
        p=dict(sensor='QB',policy_id='S100',scale=1.,alpha=r['alpha'],gamma=r['gamma'],
            lr_S=r['lr_student'],total_updates=r['updates'],student_width=r['width'],
            warmup_updates=r['warmup'],min_lr=0.,inherited_updates=0,schedule_kind='WARMUP_COSINE',
            sampling_variant='UBIL',input_variant='CORRECTED_DIFF',weight_decay=.01,
            effective_batch_size=r['batch'],microbatch_size=r['microbatch'],
            donor_case=G.donor_case(r),loss_pair=stage1_pair(G.donor_case(r)),
            lambda_abs=r['stage1_lambda_abs'],lambda_epsilon=r['stage1_lambda_epsilon'],
            **{k:None for k in SOURCE_FIELDS})
        if r['case'] in CASES and CASES[r['case']]!=p: raise ValueError('Conflicting server recipe')
        CASES[r['case']]=p

def validate_loss_pair_config(cfg):
    try: row=G.job(cfg['server'],cfg['case'],cfg['replica'])
    except (KeyError,StopIteration) as exc: raise ValueError('Unregistered QB_S2 job') from exc
    expected=PAIR_PARAMETERS[stage1_pair(G.donor_case(row))]
    donor=row['phase']=='stage1'
    if (cfg.get('campaign_id')!=CAMPAIGN_ID or cfg.get('dataset')!='QB'
        or cfg.get('phase')!=('donor' if donor else 'stage2') or cfg.get('seed')!=row['seed']
        or cfg.get('donor_version')!=G.donor_case(row) or cfg.get('stage1_loss_parameters')!=expected
        or cfg.get('loss_pair')!=stage1_pair(G.donor_case(row))
        or cfg.get('epsilon_radius_rr')!=3. or cfg.get('epsilon_distribution')!='UNIFORM_AREA_DISK'
        or cfg.get('planned_horizon')!=row['updates']
        or any(cfg.get(k)!=(v if donor else 0.) for k,v in expected.items())
        or cfg.get('lambda_eps_RR')!=(expected['lambda_epsilon'] if donor else 0.)):
        raise ValueError('QB_S2 donor/loss/seed/horizon registration mismatch')
    t=cfg.get('training_spec',{})
    if t and (t.get('effective_batch_size')!=row['batch'] or t.get('microbatch_size')!=row['microbatch']
              or t.get('warmup_updates')!=row['warmup'] or t.get('weight_decay')!=.01):
        raise ValueError('QB_S2 batch/warmup/weight decay differs from plan')
    return expected

def validate_source_teacher(binding):
    teacher=binding.get('band30_teacher',{})
    origin=binding.get('source_campaign_id')
    known={CAMPAIGN_ID}
    if (origin not in known or teacher!=binding.get('mature30_teacher') or teacher.get('campaign_id')!=origin
        or binding.get('sensor')!='QB'
        or any(teacher.get(k)!=v for k,v in dict(upsampler='interp23',aligner_outputs=2,pan_channels=1,
            alignment_target='SCHARR_OF_RGB_MEAN_GT').items())
        or (teacher.get('lambda_abs'),teacher.get('lambda_epsilon')) not in {(a,e) for _,a,e in LOSS_PAIRS}):
        raise ValueError('Authentic RGB/L4 QB same-checkpoint A/T required')
    return teacher

def validate_donor_loss_pair(cfg,binding):
    expected=validate_loss_pair_config(cfg); teacher=validate_source_teacher(binding)
    donor=G.job(cfg['server'],cfg['donor_version'])
    run_id=f"{RUN_PREFIX}_QB_S2_{donor['case']}_NOSOFT_{donor['replica']}_S{donor['seed']}"
    if (binding['source_campaign_id']!=CAMPAIGN_ID or binding.get('donor_run_id')!=run_id
            or any(teacher.get(k)!=v for k,v in expected.items())):
        raise ValueError('Student uses a different Stage1 branch/server')
    # Actual loader additionally verifies local seed, source receipt and full50K history.
    return expected
