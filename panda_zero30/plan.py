"""Pre-registered ZERO then compute-matched C4 consistency, local 30h queue."""
from copy import deepcopy
from dataclasses import asdict
import csv
import io
from pathlib import Path

from panda_mature30 import plan as previous
from donor30.plan import canonical_sha256
from .contract import CAMPAIGN_ID, METHOD_REVISION, BUDGET, POLICY, STAGE1_SELECTION

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = Path(__file__).resolve()
HORIZON = 50000
DONORS = previous.DONORS
SEEDS = previous.SEEDS
SOURCE_SEED = previous.SOURCE_SEED


def donor_receipt_path(server):
    baseline(server)
    return ('work_dir/panda_mature30/PANDA_S13_MATURE_ALIGN_HARDONLY_30H_20261020_v2/'
            + server + '/WV3/donors/rre_selection.json')


def baseline(server):
    return previous.baseline(server)


def groups(server):
    baseline(server)
    return [
        {'name': 'zero_replicates', 'optional': False,
         'runs': [('ZERO', 'R1'), ('ZERO', 'R2')], 'admission': 'WHOLE_GROUP_OR_SKIP'},
        {'name': 'rotation_pair_r1', 'optional': False,
         'runs': [('PAIR0', 'R1'), ('ROTCONS', 'R1')], 'admission': 'WHOLE_GROUP_OR_SKIP'},
        {'name': 'rotation_pair_r2', 'optional': True,
         'runs': [('PAIR0', 'R2'), ('ROTCONS', 'R2')], 'admission': 'WHOLE_GROUP_OR_SKIP'},
    ]


def default_config(server):
    return {
        'campaign_id': CAMPAIGN_ID, 'method_revision': METHOD_REVISION,
        'server': server, 'dataset': 'WV3', 'baseline_resolved': baseline(server),
        'budget': deepcopy(BUDGET), 'donors': list(DONORS), 'groups': groups(server),
        'donor_policy': 'REUSE_LOCAL_MATURE30_ORIGINAL_VAL_RRE_RECEIPT_NO_RESELECTION',
        'donor_receipt': donor_receipt_path(server), 'allow_cross_server_donor': False,
        'checkpoint_selection': POLICY.manifest(), 'stage1_selection': deepcopy(STAGE1_SELECTION),
        'sheet_selection': 'STAGE2_RUNNING_BEST_HQNR_THEN_ERGAS_ONE_ROW_PER_RUN',
        'evaluation': 'ORIGINAL_UNSHIFTED_UNMASKED_RR20_FR20_SINGLE_FORWARD_NO_TTA',
        'fixed_loss': {'alpha': .25, 'beta': 0., 'gamma': .5},
        'soft_distillation': 'REMOVED_USER_DECISION_20261020',
        'time_prior_hours': {'stage1': 0., 'stage2': 3.65, 'paired': 5.5},
        'timing_estimator': 'RECENT_UPDATE_P90_DISTINCT_SINGLE_OR_PAIRED_PLUS_EVAL_SAVE_12PCT_MARGIN',
        'priority_policy': 'ZERO_2SEED_THEN_COMPUTE_MATCHED_PAIR_THEN_OPTIONAL_PAIR_REPEAT',
        'joint_target': {'HQNR': .959, 'ERGAS': 2.045},
        'evidence_level': 'EXPLORATORY_TEST_AWARE_NOT_CONFIRMATORY',
    }


def validate_config(cfg):
    if canonical_sha256(cfg) != canonical_sha256(default_config(cfg.get('server'))):
        raise ValueError('Unregistered ZERO30 configuration')


def resolve_case(server, case, replica='R1', *, attempt=1):
    from .spec import MatureSpec, policy_manifest
    registered = [r for g in groups(server) for r in g['runs']]
    donor = case in DONORS
    if ((case, replica) not in registered and not (donor and replica == 'R1')
            or type(attempt) is not int or attempt < 1):
        raise ValueError('Unregistered ZERO30 case/replica/attempt')
    # Inherit unchanged baseline, seed streams, training/evaluation schedule.
    cfg = previous.resolve_case(server, DONORS[0] if donor else 'HALF', replica, attempt=attempt)
    cfg.update(campaign_id=CAMPAIGN_ID, method_revision=METHOD_REVISION,
        case=case, loss_case=case,
        run_id=f"PZ30_WV3_{server.upper()}_{case}_NOSOFT_{replica}_S{cfg['seed']}",
        run_display=f'ZERO30 NOSOFT {case} {replica}',
        config_digest=canonical_sha256(default_config(server)), hqnr_min=.959, ergas_max=2.045,
        question='ZERO mature donor; compute-matched C4 two-view consistency; single inference',
        prior_donor_receipt=donor_receipt_path(server),
        stage1_retrained=False, inference_tta=False)
    if not donor:
        spec = MatureSpec('WV3', 'panda_condq', cfg['stream_seeds']['S_init'],
                          .25, 0., .5, 1e-6, mask_policy=cfg['loss_mask_policy'], policy_id=case)
        cfg.update(stage2_spec=asdict(spec), policy_id=case, geometry=policy_manifest(spec),
                   shift_scale=0., student_views=1 if case == 'ZERO' else 2,
                   rotation_consistency_weight=.05 if case == 'ROTCONS' else 0.,
                   pair_transform='NONE' if case == 'ZERO' else 'EXACT_ROT90_CCW',
                   paired_weights='CANONICAL_FROZEN_HARD_MAP_ROTATED_EXACTLY')
    return cfg


def rank(record):
    return POLICY.rank(record['HQNR'], record['ERGAS'], record['completed_update'])


def catalog_csv(server):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['order','server','case','replica','seed','views','lambda_cons','updates','optional'])
    order = 0
    for group in groups(server):
        for case, replica in group['runs']:
            order += 1
            cfg = resolve_case(server, case, replica)
            writer.writerow([order,server,case,replica,cfg['seed'],cfg['student_views'],
                cfg['rotation_consistency_weight'],HORIZON,group['optional']])
    return output.getvalue()
