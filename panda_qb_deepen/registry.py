"""Eight fresh Students; no change to Aligner, loss definition or data inputs."""
from copy import deepcopy

CAMPAIGN_ID = 'PANDA_S3_QB_DEEPEN_OWNSTAGE1_20261030_v1'
METHOD_REVISION = 'PANDA-QB-DEEPEN-OWN-STAGE1-S100-DIFFUBIL-RGB-L4-HARDONLY-v1'
DONOR_CASE = 'S1_A200_E030'
STAGE1_SEED = 261016301
OUTPUT = 'work_dir/panda_qb_deepen'
TARGET = {'HQNR': .9315, 'ERGAS': 3.5}
RUNS = tuple((case, rep) for case in ('QB3_CONTROL', 'QB3_WD000', 'QB3_WD100', 'QB3_A075')
             for rep in ('R1', 'R2'))
CHANGES = {
    'QB3_CONTROL': {'axes': [], 'purpose': 'Local s3 independently trained Stage1 baseline; controls share this selected A/T pair'},
    'QB3_WD000': {'axes': ['weight_decay'], 'purpose': 'Remove existing AdamW decay: .01 to 0'},
    'QB3_WD100': {'axes': ['weight_decay'], 'purpose': 'Increase existing AdamW decay: .01 to .1'},
    'QB3_A075': {'axes': ['alpha'], 'purpose': 'Increase existing Teacher-error weighting: .25 to .75, gamma .5 unchanged'},
}
CONTROLS = {
    'R1': {'package': 'panda_s1loss9', 'campaign_id': 'PANDA_S13_S1LOSS9_RGB_L4_STAGE12_20261026_v1',
           'case': 'QB_A200_E030', 'replica': 'R1', 'seed': 261018101, 'server': 's1'},
    'R2': {'package': 'panda_sensor_refine', 'campaign_id': 'PANDA_S13_SENSOR_REFINE_R34_20261028_v1',
           'case': 'QB_BASE', 'replica': 'R2', 'seed': 261018102, 'server': 's1'},
}


def cases():
    from panda_s1loss9.contract import _fresh
    result = {name: {**_fresh('QB'), 'weight_decay': .01, 'loss_pair': 'A200_E030',
                     'donor_case': DONOR_CASE, 'lambda_abs': 2., 'lambda_epsilon': .03}
              for name in CHANGES}
    result['QB3_WD000']['weight_decay'] = 0.
    result['QB3_WD100']['weight_decay'] = .1
    result['QB3_A075']['alpha'] = .75
    return result


def manifest():
    return {'campaign_id': CAMPAIGN_ID, 'server': 's3', 'dataset': 'QB',
        'donor_origin_server': 's3', 'joint_target': deepcopy(TARGET), 'cases': cases(), 'runs': RUNS,
        'ordered_runs': ((DONOR_CASE, 'R1'),) + RUNS,
        'comparisons': deepcopy(CHANGES), 'historical_controls': deepcopy(CONTROLS),
        'stage1': {'case': DONOR_CASE, 'seed': STAGE1_SEED, 'updates': 50000,
                   'lambda_abs': 2., 'lambda_epsilon': .03, 'epsilon_radius_rr': 3.,
                   'selection': 'FULL50K_VAL_RRE50_RRE90_VAL_ERGAS_EARLIER_STEP',
                   'independent_local_training': True, 'external_donor_reuse': False},
        'initialization': 'FRESH_STUDENT', 'time_cap_seconds': None,
        'after': 'AUTHENTICATED_GF2_TARGET_RETIREMENT', 'after_campaign': 'STOP_NO_AUTOMATIC_WV3_RESUME',
        'early_stop': {'scope': 'QB_STAGE2_ONLY', 'steps': [8000, 9000, 10000],
                       'all_hqnr_strictly_below': .90, 'authenticated_safe_checkpoint_required': True},
        'selection': 'ALL_HE_JOINT_PASSES_PLUS_HQUALIFIED_MIN_E_REP_ELSE_MAX_H_REP',
        'prohibited': ['SWA', 'EMA', 'TTA', 'rotation_loss', 'soft_loss', 'band_weighted_loss']}
