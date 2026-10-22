"""Registered positive-scale and existing-Hard tuning queues with local mature donors."""
from copy import deepcopy
from dataclasses import asdict
import csv
import io
from pathlib import Path

from panda_mature30 import plan as previous
from donor30.plan import canonical_sha256
from .contract import CAMPAIGN_ID, METHOD_REVISION, BUDGET, POLICY, STAGE1_SELECTION, CASES

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = Path(__file__).resolve()
HORIZON = 50000
DONORS = previous.DONORS
SEEDS = previous.SEEDS
SOURCE_SEED = previous.SOURCE_SEED
_QUEUES = {
    's1': [('S020BASE', 'R1'), ('S010BASE', 'R1'), ('S020LR075', 'R1'),
           ('S030BASE', 'R1'), ('S040BASE', 'R1'), ('S050BASE', 'R1'),
           ('S020BASE', 'R2'), ('S010BASE', 'R2')],
    's3': [('S020BASE', 'R1'), ('S020H050', 'R1'), ('S020LR150', 'R1'),
           ('S020H200', 'R1'), ('S020LR150H050', 'R1'), ('S010LR150H050', 'R1'),
           ('S020BASE', 'R2'), ('S020LR150H050', 'R2')],
}


def donor_receipt_path(server):
    baseline(server)
    return ('work_dir/panda_mature30/PANDA_S13_MATURE_ALIGN_HARDONLY_30H_20261020_v2/'
            + server + '/WV3/donors/rre_selection.json')


def baseline(server):
    return previous.baseline(server)


def groups(server):
    baseline(server)
    return [{'name': f'{order:02d}_{case.lower()}_{replica.lower()}',
             'optional': order == 8, 'runs': [(case, replica)],
             'admission': 'WHOLE_GROUP_OR_SKIP'}
            for order, (case, replica) in enumerate(_QUEUES[server], 1)]


def default_config(server):
    return {
        'campaign_id': CAMPAIGN_ID, 'method_revision': METHOD_REVISION,
        'server': server, 'dataset': 'WV3', 'baseline_resolved': baseline(server),
        'budget': deepcopy(BUDGET), 'donors': list(DONORS), 'groups': groups(server),
        'cases': deepcopy(CASES),
        'donor_policy': 'REUSE_LOCAL_MATURE30_ORIGINAL_VAL_RRE_RECEIPT_NO_RESELECTION',
        'donor_receipt': donor_receipt_path(server), 'allow_cross_server_donor': False,
        'checkpoint_selection': POLICY.manifest(), 'stage1_selection': deepcopy(STAGE1_SELECTION),
        'sheet_selection': 'STAGE2_RUNNING_BEST_HQNR_THEN_ERGAS_ONE_ROW_PER_RUN',
        'evaluation': 'ORIGINAL_UNSHIFTED_UNMASKED_RR20_FR20_SINGLE_FORWARD_NO_TTA',
        'fixed_loss': {'beta': 0., 'delta': 1e-6, 'rotation_consistency': 0.},
        'existing_hard_formula': '(1+alpha*d_e+gamma*d_e*r_q_bar)*mean_bands(abs(S-GT))',
        'tunable_parameters': ['positive_pan_scale', 'existing_hard_alpha_gamma', 'student_peak_lr'],
        'new_method_terms': 'NONE_SINGLE_VIEW_EXISTING_C4_AUGMENTATION',
        'soft_distillation': 'REMOVED_USER_DECISION_20261020',
        'rotation_consistency': 'EXCLUDED_USER_DECISION_20261022',
        'time_prior_hours': {'stage1': 0., 'stage2': 3.65},
        'timing_estimator': 'RECENT_SINGLE_VIEW_UPDATE_P90_PLUS_EVAL_SAVE_12PCT_MARGIN',
        'priority_policy': 'ORDERED_FIXED_50K_ATTEMPTS_LOCAL_CONTROLS_AND_R2_NO_MIDRUN_RETUNING',
        'joint_target': {'HQNR': .959, 'ERGAS': 2.045},
        'evidence_level': 'EXPLORATORY_TEST_AWARE_NOT_CONFIRMATORY',
        'comparison_scope': 'SAME_SERVER_SAME_DONOR_SAME_SEED_CROSS_SERVER_DONORS_DIFFER',
    }


def validate_config(cfg):
    if canonical_sha256(cfg) != canonical_sha256(default_config(cfg.get('server'))):
        raise ValueError('Unregistered SCALE30 configuration')


def resolve_case(server, case, replica='R1', *, attempt=1):
    from .spec import MatureSpec, policy_manifest
    registered = [r for g in groups(server) for r in g['runs']]
    donor = case in DONORS
    if ((case, replica) not in registered and not (donor and replica == 'R1')
            or type(attempt) is not int or attempt < 1):
        raise ValueError('Unregistered SCALE30 case/replica/attempt')
    # Read-only reuse preserves source, seed streams, C4 augmentation and official evaluation.
    cfg = previous.resolve_case(server, DONORS[0] if donor else 'HALF', replica, attempt=attempt)
    cfg.update(campaign_id=CAMPAIGN_ID, method_revision=METHOD_REVISION,
        case=case, loss_case=case,
        run_id=f"PS30_WV3_{server.upper()}_{case}_NOSOFT_{replica}_S{cfg['seed']}",
        run_display=f'SCALE30 NOSOFT {case} {replica}',
        config_digest=canonical_sha256(default_config(server)), hqnr_min=.959, ergas_max=2.045,
        question='Positive frozen PAN Aligner scale and existing Hard/LR tuning; no new loss',
        prior_donor_receipt=donor_receipt_path(server),
        stage1_retrained=False, inference_tta=False)
    if not donor:
        params = CASES[case]
        spec = MatureSpec('WV3', 'panda_condq', cfg['stream_seeds']['S_init'],
                          params['alpha'], 0., params['gamma'], 1e-6,
                          mask_policy=cfg['loss_mask_policy'], policy_id=params['policy_id'])
        cfg.update(stage2_spec=asdict(spec), policy_id=spec.policy_id, geometry=policy_manifest(spec),
                   shift_scale=params['scale'], student_views=1, rotation_consistency_weight=0.,
                   pair_transform='NONE', paired_weights='NOT_APPLICABLE_SINGLE_VIEW',
                   lr_U=params['lr_S'], lr_S=params['lr_S'],
                   tuning_parameters=deepcopy(params))
        cfg['training_spec']['lr'] = params['lr_S']
    return cfg


def rank(record):
    return POLICY.rank(record['HQNR'], record['ERGAS'], record['completed_update'])


def catalog_csv(server):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['order','server','case','replica','seed','scale','alpha','beta','gamma',
                     'lr_S','views','lambda_cons','updates','optional'])
    order = 0
    for group in groups(server):
        for case, replica in group['runs']:
            order += 1
            cfg = resolve_case(server, case, replica)
            spec = cfg['stage2_spec']
            writer.writerow([order,server,case,replica,cfg['seed'],cfg['shift_scale'],
                spec['alpha'],spec['beta'],spec['gamma'],cfg['lr_S'],cfg['student_views'],
                cfg['rotation_consistency_weight'],HORIZON,group['optional']])
    return output.getvalue()
