"""Finite fresh-Student recipes using only the two pinned original s2 donors."""
from copy import deepcopy
from panda_qb_s2.contract import (file_identity, STAGE1_SELECTION, POLICY, SOURCE_FIELDS,
    TRAINING_SCHEDULE_FIELDS, SCALE_POLICIES, SAMPLING_VARIANTS, INPUT_VARIANTS,
    WARP_MANIFEST, input_manifest, sampling_manifest, validate_source_teacher,
    LANES, LOSS_PAIRS, PAIR_PARAMETERS, lane_for_sensor, policy, target_gap)
from . import registry as G

CAMPAIGN_ID, METHOD_REVISION = G.CAMPAIGN_ID, G.METHOD_REVISION
RUN_PREFIX = 'QFS2'
DONORS = ()  # No trainable Stage1 case exists in this consumer.
BUDGET = dict(total_seconds=None, closeout_seconds=600, safety_factor=1.08,
              calibration_reserve_seconds=0, scope='FROZEN_S2_NO_DEADLINE',
              restart='EXACT_CHECKPOINT_RESUME_NO_REINITIALIZATION')

def stage1_pair(case):
    d=G.donor(case)
    return 'A400_E030' if d['source_case']=='QB6_S1_A400_E030_D301' else 'A200_E030'

CASES = {}
for r in G.PLAN['queues']['s2']:
    p=dict(sensor='QB',policy_id='S100',scale=1.,alpha=r['alpha'],gamma=r['gamma'],
        lr_S=r['lr_student'],total_updates=r['updates'],student_width=r['width'],
        warmup_updates=r['warmup'],min_lr=0.,inherited_updates=0,schedule_kind='WARMUP_COSINE',
        sampling_variant='UBIL',input_variant='CORRECTED_DIFF',weight_decay=.01,
        effective_batch_size=r['batch'],microbatch_size=r['microbatch'],donor_case=r['donor_case'],
        loss_pair=r['loss_pair'],**PAIR_PARAMETERS[r['loss_pair']],**{k:None for k in SOURCE_FIELDS})
    if r['case'] in CASES and CASES[r['case']]!=p:
        raise ValueError('Conflicting frozen Student recipe')
    CASES[r['case']]=p

def validate_loss_pair_config(cfg):
    try: row=G.job(cfg['server'],cfg['case'],cfg['replica'])
    except (KeyError,StopIteration) as exc: raise ValueError('Unregistered frozen Student') from exc
    expected=PAIR_PARAMETERS[row['loss_pair']]
    if (cfg.get('server')!='s2' or cfg.get('campaign_id')!=CAMPAIGN_ID or cfg.get('dataset')!='QB'
        or cfg.get('phase')!='stage2' or cfg.get('seed')!=row['seed']
        or cfg.get('donor_version')!=row['donor_case'] or cfg.get('stage1_loss_parameters')!=expected
        or cfg.get('loss_pair')!=row['loss_pair'] or cfg.get('stage1_retrained') is not False
        or cfg.get('epsilon_radius_rr')!=3. or cfg.get('epsilon_distribution')!='UNIFORM_AREA_DISK'
        or cfg.get('planned_horizon')!=row['updates'] or cfg.get('shift_scale')!=1.
        or any(cfg.get(k)!=0. for k in ('lr_A','lambda_abs','lambda_epsilon','lambda_eps_RR'))):
        raise ValueError('Frozen s2 donor/Student identity or training contract differs')
    t=cfg.get('training_spec',{})
    if (t.get('effective_batch_size')!=row['batch'] or t.get('microbatch_size')!=row['microbatch']
        or t.get('warmup_updates')!=row['warmup'] or t.get('weight_decay')!=.01):
        raise ValueError('Registered Student batch/warmup/weight decay differs')
    return expected

def validate_donor_loss_pair(cfg,binding):
    expected=validate_loss_pair_config(cfg)
    teacher=validate_source_teacher(binding)
    pin=G.donor(cfg['donor_version'])
    if (binding.get('source_campaign_id')!=pin['source_campaign_id']
        or binding.get('donor_run_id')!=pin['source_run_id']
        or binding.get('donor_step')!=pin['selected_step']
        or binding.get('donor_checkpoint_sha256')!=pin['checkpoint_sha256']
        or binding.get('mature30_selector_sha256')!=pin['selector_receipt_sha256']
        or any(teacher.get(k)!=v for k,v in expected.items())):
        raise ValueError('Only the exact selected s2 checkpoint A/T is registered')
    return expected
