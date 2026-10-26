"""Fixed 36-hour s1/s3 fresh-Student queues with matched prior controls."""
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
HORIZON = max(row['total_updates'] for row in CASES.values())
DONORS, SOURCE_SEED = previous.DONORS, previous.SOURCE_SEED
SEEDS = {**previous.SEEDS, 'R3':261018103}
_QUEUES = {server: [(case,replica) for case in CASES for replica in ('R1','R2')]
           for server in ('s1','s3')}



def baseline(server):
    return previous.baseline(server)


def donor_receipt_path(server):
    baseline(server)
    return ('work_dir/panda_mature30/PANDA_S13_MATURE_ALIGN_HARDONLY_30H_20261020_v2/'
            + server + '/WV3/donors/rre_selection.json')


def groups(server):
    baseline(server)
    result = []
    for case, replica in _QUEUES[server]:
        if not result or result[-1]['case'] != case:
            result.append({'name':f'{len(result)+1:02d}_{case.lower()}', 'case':case,
                'optional':False, 'runs':[],
                'admission':'COMPLETE_SEED_PAIR_WITHIN_DURABLE_36H'})
        result[-1]['runs'].append((case,replica))
    return result


def evaluation_steps(horizon, inherited_updates=0):
    # Same checkpoints across configurations; no target-driven cadence or stop.
    return list(range(inherited_updates+1000,horizon+1,1000))


def default_config(server):
    return {
        'campaign_id':CAMPAIGN_ID, 'method_revision':METHOD_REVISION,
        'plan_revision':'ALPHA_GAMMA_ONLY_PAIRED_36H_20261026',
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
        'tunable_parameters':['alpha','gamma'],
        'new_method_terms':'EXISTING_FULL_SCALE_DIFF_INPUT_AND_APPROVED_UBIL_NO_NEW_LOSS',
        'fixed_aligner_scale':1.,
        'weight_averaging':'EXCLUDED_USER_DECISION_20261024',
        'band_weighted_loss':'EXCLUDED_USER_DECISION_20261023',
        'warp':deepcopy(WARP_MANIFEST),
        'comparison_control':{'campaign':CONTROL_CAMPAIGN, 'case':CONTROL_CASE,
                              'replicas':['R1','R2']},
        'difference_control':{'campaign':CONTROL_CAMPAIGN,'case':'S100DIFFC30'},
        'migration_policy':'WAIT_FOR_TERMINAL_TUNE30_NO_PREDECESSOR_MUTATION',
        'rotation_consistency':'EXCLUDED_USER_DECISION_20261023',
        'soft_distillation':'REMOVED_USER_DECISION_20261020',
        'time_prior_hours':{'stage1':0.,'stage2':2.5 if server=='s1' else 1.667},
        'time_prior_seconds_per_update':{'s1':.30,'s3':.20},
        'time_prior_scope':'ALL_IN_WALL_TIME_TRAIN_EVAL_SAVE_FROM_FOLLOW30_W112_UBIL',
        'priority_policy':'PAIRED_FIXED_QUEUE_36H_NO_ADAPTIVE_RETUNING',
        'joint_target':{'HQNR':HQNR_MIN,'ERGAS':ERGAS_MAX},
        'evidence_level':'EXPLORATORY_TEST_AWARE_NOT_CONFIRMATORY',
        'comparison_scope':'SAME_SERVER_SAME_DONOR_SAME_SEED_CROSS_SERVER_DONORS_DIFFER',
        'source_campaign':CONTROL_CAMPAIGN,
        'priority_policy_detail':'UNIFORM_GT_THEN_E_ONLY_Q_ONLY_HALF_BOTH_ALPHA_UP_GAMMA_UP',
    }


def validate_config(cfg):
    if canonical_sha256(cfg) != canonical_sha256(default_config(cfg.get('server'))):
        raise ValueError('Unregistered HARD36 configuration')


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
    control = CONTROL_CASE
    changed = [name for name,value in (('alpha',.25),('gamma',.5))
               if CASES[case][name] != value]
    return {'campaign_id':CONTROL_CAMPAIGN, 'case':control,
            'run_id':f'PF30_WV3_{server.upper()}_{control}_NOSOFT_{replica}_S{seed}',
            'server':server,'replica':replica,'seed':seed,
            'fresh_student_no_weights_reused':True,'single_factor_comparison':len(changed)==1,
            'changed_factors':changed,
            'comparison_kind':'MATCHED_SERVER_DONOR_SEED_UBIL_C30_LR100'}


def resolve_case(server, case, replica='R1', *, attempt=1):
    from .spec import MatureSpec, policy_manifest
    from .engine import ScheduleTrainingSpec
    registered = [r for g in groups(server) for r in g['runs']]
    donor = case in DONORS
    if ((case,replica) not in registered and not (donor and replica=='R1')
            or type(attempt) is not int or attempt<1):
        raise ValueError('Unregistered HARD36 case/replica/attempt')
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
        run_display=f'HARD36 L4 {case} {replica}',
        config_digest=canonical_sha256(default_config(server)),hqnr_min=HQNR_MIN,ergas_max=ERGAS_MAX,
        question='Only alpha/gamma vary; beta=0; matched full-scale DIFF UBIL C30' ,
        prior_donor_receipt=donor_receipt_path(server),stage1_retrained=False,inference_tta=False,
        model_variant='RAW',weight_averaging='EXCLUDED_USER_DECISION_20261024')
    if donor:
        return cfg
    p = CASES[case]
    spec = MatureSpec('WV3','panda_condq' if p['gamma'] > 0 else 'panda_e',seeds['S_init'],p['alpha'],0.,p['gamma'],1e-6,
        mask_policy=cfg['loss_mask_policy'],policy_id=p['policy_id'],
        total_updates=p['total_updates'],sampling_variant=p['sampling_variant'],input_variant=p['input_variant'],student_width=p['student_width'])
    training = ScheduleTrainingSpec(lr=p['lr_S'],total_updates=p['total_updates'],
        warmup_updates=p['warmup_updates'],min_lr=p['min_lr'],full_audit_interval=100,
        inherited_updates=p['inherited_updates'],schedule_kind=p['schedule_kind'],
        source_lr=p['source_lr'],source_horizon=p['source_horizon'],
        source_warmup_updates=p['source_warmup_updates'],source_min_lr=p['source_min_lr'])
    retention = {p['total_updates']}
    cfg.update(variant=spec.variant, sampling_variant=p['sampling_variant'],input_variant=p['input_variant'],
        student_sampling=sampling_manifest(p['sampling_variant'],p['input_variant'],p['student_width']),
        student_input=input_manifest(p['input_variant'],p['student_width']),student_warp=deepcopy(WARP_MANIFEST),
        student_architecture={'bands':8,'width':p['student_width'],'depth':[1,2,3],
            'input_layout':'P0','base_reference':'BICUBIC_MS'},
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
    output = io.StringIO(); writer = csv.writer(output, lineterminator='\n')
    writer.writerow(['order','server','case','replica','seed','scale','alpha','beta','gamma',
        'sampling_variant','input_variant','student_width','warp','lr_S','min_lr','schedule','views','lambda_cons',
        'updates','inherited_updates','new_updates','source_package','source_campaign_id',
        'source_case','source_step','source_lr','source_horizon','weight_averaging','optional'])
    for order,g in enumerate(groups(server),1):
        for case,replica in g['runs']:
            cfg = resolve_case(server,case,replica); p = CASES[case]
            writer.writerow([order,server,case,replica,cfg['seed'],p['scale'],p['alpha'],0.,p['gamma'],
                p['sampling_variant'],p['input_variant'],p['student_width'],'lanczos4',p['lr_S'],p['min_lr'],p['schedule_kind'],
                1,0.,p['total_updates'],p['inherited_updates'],cfg['continuation_updates'],
                p['source_package'],p['source_campaign_id'],p['source_case'],p['inherited_updates'],
                p['source_lr'],p['source_horizon'],cfg['weight_averaging'],g['optional']])
    return output.getvalue()
