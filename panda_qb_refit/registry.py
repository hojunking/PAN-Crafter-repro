"""Five registered QB comparisons, two matched fresh-Student seeds each."""
from copy import deepcopy

CAMPAIGN_ID = 'PANDA_S1_QB_REFIT_FIXEDDONOR_20261030_v1'
METHOD_REVISION = 'PANDA-QB-REFIT-S100-DIFFUBIL-RGB-L4-HARDONLY-v1'
OLD_CAMPAIGN = 'PANDA_S13_SENSOR_FOCUS_FIXEDDONOR_20261029_v1'
TARGET = {'HQNR': .9315, 'ERGAS': 3.5}
OUTPUT = 'work_dir/panda_qb_refit'
OLD_RUNS = (('QB_W128', 'R1'), ('QB_W128', 'R2'), ('QB_A0125', 'R1'),
            ('QB_A0125', 'R2'), ('QB_H050', 'R2'), ('QB_G000', 'R1'))
RUNS = tuple((case, rep) for case in (
    'QB2_C30LR080', 'QB2_C30LR050', 'QB2_C25', 'QB2_C30AG000', 'QB2_C30G025')
    for rep in ('R1', 'R2'))
CHANGES = {
    'QB2_C30LR080': {'axes': ['lr_S'], 'purpose': 'Moderately lower LR: 1e-4 to 8e-5'},
    'QB2_C30LR050': {'axes': ['lr_S'], 'purpose': 'Lower LR bracket: 1e-4 to 5e-5'},
    'QB2_C25': {'axes': ['total_updates'], 'purpose': 'Earlier cosine endpoint: 30K to 25K'},
    'QB2_C30AG000': {'axes': ['alpha', 'gamma'], 'purpose': 'Joint existing Hard-weight ablation: both extra weights zero; same valid mask'},
    'QB2_C30G025': {'axes': ['gamma'], 'purpose': 'Intermediate conditional weight: .5 to .25; alpha .25 unchanged'},
}
CONTROLS = {
    'R1': {'package': 'panda_s1loss9',
           'campaign_id': 'PANDA_S13_S1LOSS9_RGB_L4_STAGE12_20261026_v1',
           'case': 'QB_A200_E030', 'replica': 'R1', 'seed': 261018101},
    'R2': {'package': 'panda_sensor_refine',
           'campaign_id': 'PANDA_S13_SENSOR_REFINE_R34_20261028_v1',
           'case': 'QB_BASE', 'replica': 'R2', 'seed': 261018102},
}


def cases():
    from panda_sensor_focus.contract import _fresh
    return {
        'QB2_C30LR080': _fresh('QB', lr=8e-5),
        'QB2_C30LR050': _fresh('QB', lr=5e-5),
        'QB2_C25': _fresh('QB', updates=25000),
        'QB2_C30AG000': _fresh('QB', alpha=0., gamma=0.),
        'QB2_C30G025': _fresh('QB', gamma=.25),
    }


def manifest():
    return {'campaign_id': CAMPAIGN_ID, 'server': 's1', 'dataset': 'QB',
            'joint_target': deepcopy(TARGET), 'cases': cases(), 'runs': RUNS,
            'comparisons': deepcopy(CHANGES), 'matched_seed_controls': deepcopy(CONTROLS),
            'stage1': 'REUSE_EXACT_AUTHENTICATED_S1_A200_E030_44000',
            'initialization': 'FRESH_STUDENT', 'time_cap_seconds': None,
            'after': 'CURRENT_WV3_RUN_COMPLETED_THEN_QB_BEFORE_REMAINING_WV3',
            'queue_change_policy': 'USER_REQUESTED_REVIEW_THEN_VERSIONED_CHECKPOINT_BOUNDARY_MIGRATION',
            'early_stop': {'scope': 'QB_STAGE2_ONLY', 'steps': [8000, 9000, 10000],
                           'all_hqnr_strictly_below': .90, 'authenticated_safe_checkpoint_required': True},
            'selection': 'ALL_HE_JOINT_PASSES_PLUS_HQUALIFIED_MIN_E_REP_ELSE_MAX_H_REP',
            'prohibited': ['SWA', 'EMA', 'TTA', 'rotation_loss', 'soft_loss', 'band_weighted_loss']}
