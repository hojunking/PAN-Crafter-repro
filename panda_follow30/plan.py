"""Finite s1/s3 follow-up queues with explicit same-seed continuation parents."""
from copy import deepcopy
from dataclasses import asdict
import csv
import hashlib
import io
from pathlib import Path

from panda_mature30 import plan as previous
from donor30.plan import canonical_sha256, stream_seeds
from .contract import (CAMPAIGN_ID, METHOD_REVISION, NEXT_CAMPAIGN_ID, RUN_PREFIX,
    BUDGET, POLICY, STAGE1_SELECTION, CASES, HQNR_MIN, ERGAS_MAX,
    sampling_manifest, input_manifest, WARP_MANIFEST, CONTROL_CAMPAIGN,
    CONTROL_CASE, SOURCE_FIELDS)

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = Path(__file__).resolve()
HORIZON = 50000
DONORS, SOURCE_SEED = previous.DONORS, previous.SOURCE_SEED
SEEDS = {**previous.SEEDS, 'R3':261018103}
_QUEUES = {
    's1': [('S100DIFFC50','R1'),('S100DIFFC50','R2'),
           ('S100DIFFUBILC30','R1'),('S100DIFFUBILC30','R2'),
           ('S100DIFFF15C30','R1'),('S100DIFFF15C30','R2'),
           ('S100DIFFC30','R3'),('S100C30','R3'),
           ('S100UBILC30','R1'),('S100UBILC30','R2')],
    's3': [('S100DIFFC50','R1'),('S100DIFFC50','R2'),
           ('S100DIFFLR150C50','R1'),('S100DIFFLR150C50','R2'),
           ('S100DIFFH050C50','R1'),('S100DIFFH050C50','R2'),
           ('S100DIFFUBILC30','R1'),('S100DIFFUBILC30','R2'),
           ('S100DIFFF15C30','R1'),('S100DIFFF15C30','R2'),
           ('S100DIFFC30','R3'),('S100C30','R3')],
}


def baseline(server):
    return previous.baseline(server)


def donor_receipt_path(server):
    baseline(server)
    return ('work_dir/panda_mature30/PANDA_S13_MATURE_ALIGN_HARDONLY_30H_20261020_v2/'
            + server + '/WV3/donors/rre_selection.json')


def groups(server):
    baseline(server)
    return [{'name':f'{i:02d}_{case.lower()}_{replica.lower()}', 'optional':False,
             'runs':[(case,replica)], 'admission':'REGISTERED_NO_WALLTIME_LIMIT'}
            for i,(case,replica) in enumerate(_QUEUES[server],1)]


def evaluation_steps(horizon, inherited_updates=0):
    # Same checkpoints across configurations; no target-driven cadence or stop.
    return list(range(inherited_updates+1000,horizon+1,1000))


def default_config(server):
    return {
        'campaign_id':CAMPAIGN_ID, 'method_revision':METHOD_REVISION,
        'plan_revision':'FULL_SCALE_NO_WEIGHT_AVERAGING_20261024',
        'server':server, 'dataset':'WV3', 'baseline_resolved':baseline(server),
        'budget':deepcopy(BUDGET), 'donors':list(DONORS), 'groups':groups(server),
        'cases':deepcopy(CASES), 'seeds':deepcopy(SEEDS),
        'donor_receipt':donor_receipt_path(server),
        'donor_policy':'REUSE_LOCAL_MATURE30_ORIGINAL_VAL_RRE_RECEIPT_NO_RESELECTION',
        'allow_cross_server_donor':False,
        'checkpoint_selection':POLICY.manifest(), 'stage1_selection':deepcopy(STAGE1_SELECTION),
        'sheet_selection':'ALL_H9585_E2040_PASSES_PLUS_H9585_MIN_ERGAS_OR_BEST_HQNR_REPRESENTATIVE',
        'evaluation':'ORIGINAL_UNSHIFTED_UNMASKED_RR20_FR20_SINGLE_FORWARD_NO_TTA',
        'fixed_loss':{'beta':0., 'delta':1e-6, 'rotation_consistency':0.},
        'existing_hard_formula':'(1+alpha*d_e+gamma*d_e*r_q_bar)*mean_bands(abs(S-GT))',
        'tunable_parameters':['schedule','student_lr','hard_coefficients',
                              'student_sampling','student_input'],
        'new_method_terms':'EXISTING_FULL_SCALE_DIFF_INPUT_AND_APPROVED_UBIL_NO_NEW_LOSS',
        'fixed_aligner_scale':1.,
        'weight_averaging':'EXCLUDED_USER_DECISION_20261024',
        'band_weighted_loss':'EXCLUDED_USER_DECISION_20261023',
        'warp':deepcopy(WARP_MANIFEST),
        'comparison_control':{'campaign':CONTROL_CAMPAIGN, 'case':CONTROL_CASE,
                              'replicas':['R1','R2']},
        'difference_control':{'campaign':NEXT_CAMPAIGN_ID,'case':'S100DIFFC30'},
        'migration_policy':'KEEP_STARTED_AND_FULL_SCALE_NEXT30; SKIP_ONLY_UNSTARTED_LOW_SCALE; THEN_FOLLOW30',
        'rotation_consistency':'EXCLUDED_USER_DECISION_20261023',
        'soft_distillation':'REMOVED_USER_DECISION_20261020',
        'time_prior_hours':{'stage1':0.,'stage2':3.65},
        'priority_policy':'FINITE_REGISTERED_QUEUE_NO_WALLTIME_LIMIT_NO_ADAPTIVE_RETUNING',
        'joint_target':{'HQNR':HQNR_MIN,'ERGAS':ERGAS_MAX},
        'evidence_level':'EXPLORATORY_TEST_AWARE_NOT_CONFIRMATORY',
        'comparison_scope':'SAME_SERVER_SAME_DONOR_SAME_SEED_CROSS_SERVER_DONORS_DIFFER',
        'source_campaign':NEXT_CAMPAIGN_ID,
        'priority_policy_detail':'FULL_SCALE_DIFF_MAIN; CORRECTED_ONLY_SMALL_MATCHED_CONTROLS',
    }


def validate_config(cfg):
    if canonical_sha256(cfg) != canonical_sha256(default_config(cfg.get('server'))):
        raise ValueError('Unregistered FOLLOW30 configuration')


def _source_identity(server, replica, row):
    if row['source_case'] is None:
        return {'source_run_id':None,'source_replica':None,'source_attempt':None,
                'source_completed_update':0,'source_checkpoint_relative_path':None}
    prefix = 'PNX30' if row['source_package'] == 'panda_next30' else RUN_PREFIX
    run = f"{prefix}_WV3_{server.upper()}_{row['source_case']}_NOSOFT_{replica}_S{SEEDS[replica]}"
    rel = (Path('work_dir')/row['source_package']/row['source_campaign_id']/server/
           'WV3/runs'/run/'attempt_01/models'/f"step_{row['inherited_updates']:07d}.pt")
    return {'source_run_id':run,'source_replica':replica,'source_attempt':1,
            'source_completed_update':row['inherited_updates'],
            'source_checkpoint_relative_path':str(rel)}


def source_checkpoint_path(cfg):
    row = CASES.get(cfg.get('case'))
    if (row is None or row['source_case'] is None
            or cfg.get('replica') not in SEEDS
            or cfg.get('seed') != SEEDS[cfg['replica']]
            or cfg.get('inherited_updates') != row['inherited_updates']
            or any(cfg.get(k) != row[k] for k in SOURCE_FIELDS)):
        raise ValueError('Exact registered same-seed continuation source required')
    baseline(cfg.get('server'))
    if (cfg['case'],cfg['replica']) not in _QUEUES[cfg['server']]:
        raise ValueError('Continuation case/replica is not registered on this server')
    expected = _source_identity(cfg['server'],cfg['replica'],row)
    if any(cfg.get(k) != value for k,value in expected.items()):
        raise ValueError('Continuation source run/attempt/path differs from registration')
    return ROOT/expected['source_checkpoint_relative_path']


def _comparison(server, case, replica, seed):
    row = CASES[case]
    if row['source_case']:
        return {'campaign_id':row['source_campaign_id'],'case':row['source_case'],
                'server':server,'replica':replica,'seed':seed,
                'comparison':'SAME_PARENT_PREFIX_OR_MATCHED_CONTINUATION'}
    campaign, control = NEXT_CAMPAIGN_ID, 'S100DIFFC30'
    if case in ('S100DIFFLR150C50','S100DIFFH050C50'):
        campaign,control = CAMPAIGN_ID,'S100DIFFC50'
    elif replica == 'R3' and case == 'S100DIFFC30':
        campaign,control = CAMPAIGN_ID,'S100C30'
    elif case == 'S100UBILC30':
        campaign,control = NEXT_CAMPAIGN_ID,'S100C30'
    elif case == 'S100C30':
        campaign,control = CAMPAIGN_ID,'S100DIFFC30'
    return {'campaign_id':campaign,'case':control,'server':server,'replica':replica,
            'seed':seed,'fresh_student_no_weights_reused':True,
            'single_factor_comparison':case in ('S100DIFFUBILC30','S100DIFFC50',
                'S100DIFFLR150C50','S100DIFFH050C50','S100UBILC30','S100DIFFC30','S100C30'),
            'comparison_kind':'MATCHED_SETTINGS'}


def resolve_case(server, case, replica='R1', *, attempt=1):
    from .spec import MatureSpec, policy_manifest
    from .engine import ScheduleTrainingSpec
    registered = [r for g in groups(server) for r in g['runs']]
    donor = case in DONORS
    if ((case,replica) not in registered and not (donor and replica=='R1')
            or type(attempt) is not int or attempt<1):
        raise ValueError('Unregistered FOLLOW30 case/replica/attempt')
    # Mature30 does not register R3. Rebuild all seed-derived fields explicitly.
    cfg = previous.resolve_case(server,DONORS[0] if donor else 'HALF',
                                'R1' if replica=='R3' else replica,attempt=attempt)
    seed = SOURCE_SEED if donor else SEEDS[replica]
    seeds = stream_seeds(seed)
    seeds['rr_band_augmentation'] = 1+int.from_bytes(hashlib.sha256(
        f'donor30:{seed}:rr_band_augmentation'.encode()).digest()[:8],'big')%(2**63-1)
    cfg.update(campaign_id=CAMPAIGN_ID,method_revision=METHOD_REVISION,case=case,loss_case=case,
        replica=replica,seed=seed,stream_seeds=seeds,
        run_id=f'{RUN_PREFIX}_WV3_{server.upper()}_{case}_NOSOFT_{replica}_S{seed}',
        run_display=f'FOLLOW30 L4 {case} {replica}',
        config_digest=canonical_sha256(default_config(server)),hqnr_min=HQNR_MIN,ergas_max=ERGAS_MAX,
        question='Full-scale Aligner with existing difference input, matched schedule/Hard and UBIL follow-up',
        prior_donor_receipt=donor_receipt_path(server),stage1_retrained=False,inference_tta=False,
        model_variant='RAW',weight_averaging='EXCLUDED_USER_DECISION_20261024')
    if donor:
        return cfg
    p = CASES[case]
    spec = MatureSpec('WV3','panda_condq',seeds['S_init'],p['alpha'],0.,p['gamma'],1e-6,
        mask_policy=cfg['loss_mask_policy'],policy_id=p['policy_id'],
        total_updates=p['total_updates'],sampling_variant=p['sampling_variant'],input_variant=p['input_variant'])
    training = ScheduleTrainingSpec(lr=p['lr_S'],total_updates=p['total_updates'],
        warmup_updates=p['warmup_updates'],min_lr=p['min_lr'],full_audit_interval=100,
        inherited_updates=p['inherited_updates'],schedule_kind=p['schedule_kind'],
        source_lr=p['source_lr'],source_horizon=p['source_horizon'],
        source_warmup_updates=p['source_warmup_updates'],source_min_lr=p['source_min_lr'])
    retention = {p['total_updates']}
    cfg.update(sampling_variant=p['sampling_variant'],input_variant=p['input_variant'],
        student_sampling=sampling_manifest(p['sampling_variant'],p['input_variant']),
        student_input=input_manifest(p['input_variant']),student_warp=deepcopy(WARP_MANIFEST),
        stage2_spec=asdict(spec),policy_id=spec.policy_id,geometry=policy_manifest(spec),
        shift_scale=p['scale'],student_views=1,rotation_consistency_weight=0.,pair_transform='NONE',
        paired_weights='NOT_APPLICABLE_SINGLE_VIEW',lr_U=p['lr_S'],lr_S=p['lr_S'],
        training_spec=asdict(training),tuning_parameters=deepcopy(p),
        schedule=p['schedule_kind'],planned_horizon=p['total_updates'],
        inherited_updates=p['inherited_updates'],start_update=0,
        continuation_updates=p['total_updates']-p['inherited_updates'],
        evaluation_steps=evaluation_steps(p['total_updates'],p['inherited_updates']),
        retention_steps=sorted(retention),
        initialization='FORK_PRESERVE_COMPLETE_STATE' if p['source_case'] else 'FRESH',
        counter_policy='ABSOLUTE_STUDENT_UPDATE; INHERITED_UPDATES_NOT_CHARGED_TO_LOCAL_TRAINING',
        comparison_control=_comparison(server,case,replica,seed),
        **{key:p[key] for key in SOURCE_FIELDS}, **_source_identity(server,replica,p))
    return cfg


def rank(record):
    return POLICY.rank(record['HQNR'],record['ERGAS'],record['completed_update'])


def catalog_csv(server):
    output = io.StringIO(); writer = csv.writer(output)
    writer.writerow(['order','server','case','replica','seed','scale','alpha','beta','gamma',
        'sampling_variant','input_variant','warp','lr_S','min_lr','schedule','views','lambda_cons',
        'updates','inherited_updates','new_updates','source_package','source_campaign_id',
        'source_case','source_step','source_lr','source_horizon','weight_averaging','optional'])
    for order,g in enumerate(groups(server),1):
        for case,replica in g['runs']:
            cfg = resolve_case(server,case,replica); p = CASES[case]
            writer.writerow([order,server,case,replica,cfg['seed'],p['scale'],p['alpha'],0.,p['gamma'],
                p['sampling_variant'],p['input_variant'],'lanczos4',p['lr_S'],p['min_lr'],p['schedule_kind'],
                1,0.,p['total_updates'],p['inherited_updates'],cfg['continuation_updates'],
                p['source_package'],p['source_campaign_id'],p['source_case'],p['inherited_updates'],
                p['source_lr'],p['source_horizon'],cfg['weight_averaging'],False])
    return output.getvalue()
