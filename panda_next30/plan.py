"""Fresh Student sampling queues using immutable same-server donors and controls."""
from copy import deepcopy
from dataclasses import asdict
import csv
import io
from pathlib import Path

from panda_mature30 import plan as previous
from donor30.plan import canonical_sha256
from .contract import CAMPAIGN_ID, METHOD_REVISION, BUDGET, POLICY, STAGE1_SELECTION, CASES, HQNR_MIN, ERGAS_MAX, sampling_manifest, input_manifest, WARP_MANIFEST, CONTROL_CAMPAIGN, CONTROL_CASE

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = Path(__file__).resolve()
HORIZON = 50000  # Historical donor horizon, not a Student run default.
DONORS, SEEDS, SOURCE_SEED = previous.DONORS, previous.SEEDS, previous.SOURCE_SEED
_QUEUES = {
    's1': [('S020C30','R1'),('S100C30','R1'),('S100DIFFC30','R1'),
           ('S010C30','R1'),('S020C35','R1'),('S020C30LR150','R1'),
           ('DAVG','R1'),('UBIL','R1'),('DAVG_UBIL','R1'),
           ('S020C30','R2'),('S010C30','R2'),('S020BASE','R2'),('S010BASE','R2'),
           ('S100C30','R2'),('S100DIFFC30','R2'),('DAVG','R2'),('UBIL','R2'),('DAVG_UBIL','R2')],
    's3': [('S020C30','R1'),('S100C30','R1'),('S100DIFFC30','R1'),
           ('S020C30H050','R1'),('S020C35H050','R1'),('S020C30LR150','R1'),
           ('DAVG','R1'),('UBIL','R1'),('DAVG_UBIL','R1'),
           ('S020C30','R2'),('S020C30H050','R2'),('S020H050','R2'),
           ('S100C30','R2'),('S100DIFFC30','R2'),('DAVG','R2'),('UBIL','R2'),('DAVG_UBIL','R2')],
}


def baseline(server):
    return previous.baseline(server)


def donor_receipt_path(server):
    baseline(server)
    return ('work_dir/panda_mature30/PANDA_S13_MATURE_ALIGN_HARDONLY_30H_20261020_v2/'
            + server + '/WV3/donors/rre_selection.json')


def groups(server):
    baseline(server)
    return [{'name': f'{i:02d}_{case.lower()}_{replica.lower()}', 'optional': False,
             'runs': [(case,replica)], 'admission': 'REGISTERED_NO_WALLTIME_LIMIT'}
            for i,(case,replica) in enumerate(_QUEUES[server],1)]


def evaluation_steps(horizon, inherited_updates=0):
    # Preserve the original early checkpoints, then resolve the target crossing
    # every 1K through 35K. A continuation evaluates only newly learned states.
    original = previous.resolve_case('s1','HALF')['evaluation_steps']
    return sorted({int(s) for s in original if inherited_updates < s <= horizon}
                  | {s for s in range(20000, min(horizon,35000)+1,1000)
                     if s > inherited_updates} | {horizon})


def default_config(server):
    return {
        'campaign_id': CAMPAIGN_ID, 'method_revision': METHOD_REVISION,
        'server': server, 'dataset': 'WV3', 'baseline_resolved': baseline(server),
        'budget': deepcopy(BUDGET), 'donors': list(DONORS), 'groups': groups(server),
        'cases': deepcopy(CASES), 'donor_receipt': donor_receipt_path(server),
        'donor_policy': 'REUSE_LOCAL_MATURE30_ORIGINAL_VAL_RRE_RECEIPT_NO_RESELECTION',
        'allow_cross_server_donor': False,
        'checkpoint_selection': POLICY.manifest(), 'stage1_selection': deepcopy(STAGE1_SELECTION),
        'sheet_selection': 'ALL_H9585_E2040_PASSES_PLUS_H9585_MIN_ERGAS_OR_BEST_HQNR_REPRESENTATIVE',
        'evaluation': 'ORIGINAL_UNSHIFTED_UNMASKED_RR20_FR20_SINGLE_FORWARD_NO_TTA',
        'fixed_loss': {'beta':0.,'delta':1e-6,'rotation_consistency':0.},
        'existing_hard_formula': '(1+alpha*d_e+gamma*d_e*r_q_bar)*mean_bands(abs(S-GT))',
        'tunable_parameters': ['schedule','student_lr','hard_coefficients','student_sampling','student_input'],
        'new_method_terms': 'ONE_LANCZOS4_WARP_PLUS_REGISTERED_SAMPLING_OR_PAN_DIFFERENCE_NO_NEW_LOSS',
        'band_weighted_loss': 'EXCLUDED_USER_DECISION_20261023',
        'warp': deepcopy(WARP_MANIFEST),
        'comparison_control': {'campaign':CAMPAIGN_ID,'case':CONTROL_CASE,'replicas':['R1','R2']},
        'difference_control': {'case':'S100C30','scale':1.,'sampling_variant':'BASE'},
        'migration_policy':'PRESERVE_STARTED_BICUBIC_RUNS_REPLACE_UNSTARTED_ONLY_ALWAYS_RUN_L4_S020C30_ANCHOR',
        'rotation_consistency': 'EXCLUDED_USER_DECISION_20261023',
        'soft_distillation': 'REMOVED_USER_DECISION_20261020',
        'time_prior_hours': {'stage1':0.,'stage2':3.65},
        'priority_policy': 'FINITE_REGISTERED_QUEUE_NO_WALLTIME_LIMIT_NO_ADAPTIVE_RETUNING',
        'joint_target': {'HQNR':HQNR_MIN,'ERGAS':ERGAS_MAX},
        'evidence_level': 'EXPLORATORY_TEST_AWARE_NOT_CONFIRMATORY',
        'comparison_scope': 'SAME_SERVER_SAME_DONOR_SAME_SEED_CROSS_SERVER_DONORS_DIFFER',
        'source_campaign': CONTROL_CAMPAIGN,
    }


def validate_config(cfg):
    if canonical_sha256(cfg) != canonical_sha256(default_config(cfg.get('server'))):
        raise ValueError('Unregistered sampling extension configuration')


def resolve_case(server, case, replica='R1', *, attempt=1):
    from .spec import MatureSpec, policy_manifest
    from .engine import ScheduleTrainingSpec
    registered = [r for g in groups(server) for r in g['runs']]
    donor = case in DONORS
    if ((case,replica) not in registered and not (donor and replica=='R1')
            or type(attempt) is not int or attempt<1):
        raise ValueError('Unregistered sampling case/replica/attempt')
    cfg=previous.resolve_case(server,DONORS[0] if donor else 'HALF',replica,attempt=attempt)
    cfg.update(campaign_id=CAMPAIGN_ID,method_revision=METHOD_REVISION,case=case,loss_case=case,
        run_id=f"PNX30_WV3_{server.upper()}_{case}_NOSOFT_{replica}_S{cfg['seed']}",
        run_display=f'NEXT30 L4 {case} {replica}',
        config_digest=canonical_sha256(default_config(server)),hqnr_min=HQNR_MIN,ergas_max=ERGAS_MAX,
        question='Use one Lanczos4 warp for subsequent schedules, Student sampling and full-u PAN-difference inputs',
        prior_donor_receipt=donor_receipt_path(server),stage1_retrained=False,inference_tta=False)
    if not donor:
        p=CASES[case]
        spec=MatureSpec('WV3','panda_condq',cfg['stream_seeds']['S_init'],p['alpha'],0.,p['gamma'],1e-6,
             mask_policy=cfg['loss_mask_policy'],policy_id=p['policy_id'],total_updates=p['total_updates'],sampling_variant=p['sampling_variant'],input_variant=p['input_variant'])
        training=ScheduleTrainingSpec(lr=p['lr_S'], total_updates=p['total_updates'],
             warmup_updates=p['warmup_updates'],min_lr=p['min_lr'],full_audit_interval=100,
             inherited_updates=p['inherited_updates'])
        cfg.update(sampling_variant=p['sampling_variant'],input_variant=p['input_variant'],
            student_sampling=sampling_manifest(p['sampling_variant'],p['input_variant']),
            student_input=input_manifest(p['input_variant']),student_warp=deepcopy(WARP_MANIFEST),
            stage2_spec=asdict(spec),policy_id=spec.policy_id,geometry=policy_manifest(spec),
            shift_scale=p['scale'],student_views=1,rotation_consistency_weight=0.,pair_transform='NONE',
            paired_weights='NOT_APPLICABLE_SINGLE_VIEW',lr_U=p['lr_S'],lr_S=p['lr_S'],
            training_spec=asdict(training),tuning_parameters=deepcopy(p),
            schedule=p['schedule_kind'], planned_horizon=p['total_updates'],
            inherited_updates=p['inherited_updates'], start_update=0,
            continuation_updates=p['total_updates']-p['inherited_updates'],
            evaluation_steps=evaluation_steps(p['total_updates'],p['inherited_updates']),
            retention_steps=[p['total_updates']],
            initialization='FORK_PRESERVE_COMPLETE_STATE' if p['source_case'] else 'FRESH',
            source_case=p['source_case'], source_completed_update=p['inherited_updates'],
            counter_policy='ABSOLUTE_STUDENT_UPDATE; INHERITED_UPDATES_NOT_CHARGED_TO_LOCAL_TRAINING')
        control='S100C30' if p['input_variant']=='CORRECTED_DIFF' else CONTROL_CASE
        cfg['comparison_control'] = {'campaign_id':CAMPAIGN_ID,'case':control,
            'run_id':f"PNX30_WV3_{server.upper()}_{control}_NOSOFT_{replica}_S{cfg['seed']}",
            'server':server,'replica':replica,'seed':cfg['seed'],'fresh_student_no_weights_reused':True}
        from panda_schedule30 import plan as schedule
        old_runs={tuple(run) for group in schedule.groups(server) for run in group['runs']}
        cfg['migration_predecessor_case'] = case if (case,replica) in old_runs else None
        cfg['migration_always_run'] = case == CONTROL_CASE
    return cfg


def rank(record):
    return POLICY.rank(record['HQNR'],record['ERGAS'],record['completed_update'])


def catalog_csv(server):
    output=io.StringIO(); writer=csv.writer(output)
    writer.writerow(['order','server','case','replica','seed','scale','alpha','beta','gamma',
                     'sampling_variant','input_variant','warp','lr_S','min_lr','views','lambda_cons','updates','inherited_updates','new_updates','optional'])
    for order,g in enumerate(groups(server),1):
        for case,replica in g['runs']:
            cfg=resolve_case(server,case,replica); p=CASES[case]
            writer.writerow([order,server,case,replica,cfg['seed'],p['scale'],p['alpha'],0.,p['gamma'],
                p['sampling_variant'],p['input_variant'],'lanczos4',p['lr_S'],p['min_lr'],1,0.,p['total_updates'],p['inherited_updates'],cfg['continuation_updates'],False])
    return output.getvalue()
