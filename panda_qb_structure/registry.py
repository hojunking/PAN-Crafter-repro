"""Finite, open-ended QB queue; existing local Stage1s remain independent."""
from copy import deepcopy

CAMPAIGN_ID = 'PANDA_S13_QB_STRUCTURE_LOCALSTAGE1_20261031_v1'
METHOD_REVISION = 'PANDA-QB-STRUCTURE-LOCALSTAGE1-S100-RGB-L4-HARDONLY-v1'
OUTPUT = 'work_dir/panda_qb_structure'
TARGET = {'HQNR': .9315, 'ERGAS': 3.5}
SEEDS = {'R1': 261018101, 'R2': 261018102, 'R3': 261018103}
DONOR_CASE = 'S1_A200_E030'
SOURCES = {
    's1': {'package': 'panda_s1loss9', 'campaign_id': 'PANDA_S13_S1LOSS9_RGB_L4_STAGE12_20261026_v1',
           'case': DONOR_CASE, 'seed': 261016101},
    's3': {'package': 'panda_qb_deepen', 'campaign_id': 'PANDA_S3_QB_DEEPEN_OWNSTAGE1_20261030_v1',
           'case': DONOR_CASE, 'seed': 261016301},
}
RUNS = {
    's1': tuple((c, r) for c in ('QB4_DIFFBASE', 'QB4_P0UBIL', 'QB4_P0BASE', 'QB4_WD000')
                for r in ('R1', 'R2')) + (('QB4_CONTROL', 'R3'), ('QB4_WD000', 'R3')),
    's3': tuple((c, r) for c in ('QB4_P0UBIL', 'QB4_DIFFBASE', 'QB4_P0BASE')
                for r in ('R1', 'R2')) + (('QB4_CONTROL', 'R3'), ('QB4_WD000', 'R3')),
}
CHANGES = {
    'QB4_DIFFBASE': {'axes': ['sampling_variant'], 'purpose': 'Restore transposed-convolution upsampling; keep PAN difference input'},
    'QB4_P0UBIL': {'axes': ['input_variant'], 'purpose': 'Remove only the PAN difference input; retain corrected PAN and bilinear upsampling'},
    'QB4_P0BASE': {'axes': ['input_variant', 'sampling_variant'], 'purpose': 'Complete input-by-upsampling factorial: corrected PAN only and original upsampling'},
    'QB4_CONTROL': {'axes': [], 'purpose': 'Third matched Student seed for the existing local-donor baseline'},
    'QB4_WD000': {'axes': ['weight_decay'], 'purpose': 'AdamW weight decay zero; same local donor and matched Student seed'},
}


def cases():
    # Explicit recipe avoids importing and contaminating a source campaign.
    from panda_sensor_refine.contract import SOURCE_FIELDS
    base = dict(sensor='QB', policy_id='S100', scale=1., alpha=.25, gamma=.5,
        lr_S=1e-4, total_updates=30000, student_width=112, warmup_updates=100,
        min_lr=0., inherited_updates=0, schedule_kind='WARMUP_COSINE',
        sampling_variant='UBIL', input_variant='CORRECTED_DIFF', weight_decay=.01,
        donor_case=DONOR_CASE, lambda_abs=2., lambda_epsilon=.03, epsilon_radius_rr=3.,
        **{key: None for key in SOURCE_FIELDS})
    return {
        'QB4_DIFFBASE': {**base, 'sampling_variant': 'BASE'},
        'QB4_P0UBIL': {**base, 'input_variant': 'CORRECTED'},
        'QB4_P0BASE': {**base, 'sampling_variant': 'BASE', 'input_variant': 'CORRECTED'},
        'QB4_CONTROL': deepcopy(base),
        'QB4_WD000': {**base, 'weight_decay': 0.},
    }


def controls(server, replica):
    if server not in RUNS or replica not in SEEDS:
        raise ValueError('Registered server and replica required')
    if replica == 'R3':
        return {'package': 'panda_qb_structure', 'campaign_id': CAMPAIGN_ID,
                'case': 'QB4_CONTROL', 'replica': replica, 'server': server, 'seed': SEEDS[replica]}
    if server == 's3':
        return {'package': 'panda_qb_deepen', 'campaign_id': SOURCES[server]['campaign_id'],
                'case': 'QB3_CONTROL', 'replica': replica, 'server': server, 'seed': SEEDS[replica]}
    return {'package': 'panda_s1loss9' if replica == 'R1' else 'panda_sensor_refine',
            'campaign_id': SOURCES[server]['campaign_id'] if replica == 'R1' else 'PANDA_S13_SENSOR_REFINE_R34_20261028_v1',
            'case': 'QB_A200_E030' if replica == 'R1' else 'QB_BASE',
            'replica': replica, 'server': server, 'seed': SEEDS[replica]}


def manifest():
    return {'campaign_id': CAMPAIGN_ID, 'method_revision': METHOD_REVISION, 'dataset': 'QB',
            'joint_target': deepcopy(TARGET), 'ordered_runs': deepcopy(RUNS),
            'cases': cases(), 'seeds': deepcopy(SEEDS), 'comparisons': deepcopy(CHANGES),
            'stage1_sources': deepcopy(SOURCES), 'stage1_runs': 0,
            'stage1_policy': 'REUSE_EACH_SERVERS_COMPLETED_VALIDATION_SELECTED_LOCAL_STAGE1',
            'cross_server_donor_transfer': False, 'time_cap_seconds': None,
            'initialization': 'FRESH_STUDENT_NO_TEACHER_WEIGHT_COPY',
            'historical_s3_wd000': {'package': 'panda_qb_deepen', 'case': 'QB3_WD000', 'replicas': ['R1', 'R2']},
            'early_stop': {'scope': 'QB_STAGE2_ONLY', 'steps': [8000, 9000, 10000], 'all_hqnr_strictly_below': .90},
            'selection': 'ALL_HE_JOINT_PASSES_PLUS_HQUALIFIED_MIN_E_REP_ELSE_MAX_H_REP',
            'prohibited': ['SWA', 'EMA', 'TTA', 'rotation_loss', 'soft_loss', 'band_weighted_loss']}
