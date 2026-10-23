"""Isolated Student sampling ablations; live schedule and original warp stay immutable."""
from copy import deepcopy
from panda_mature30.contract import file_identity, POLICY, STAGE1_SELECTION

CAMPAIGN_ID = 'PANDA_S13_SAMPLING_HARDONLY_20261023_v1'
METHOD_REVISION = 'PANDA-SAMPLING30-POSITIVE-HARDONLY-v1'
HQNR_MIN, ERGAS_MAX = .9585, 2.040
BUDGET = {'total_seconds': None, 'closeout_seconds': 0, 'safety_factor': 1.12,
          'calibration_reserve_seconds': 600, 'scope': 'FINITE_REGISTERED_QUEUE_NO_WALL_TIME_LIMIT',
          'restart': 'PRESERVE_ELAPSED_AND_COMPLETE_STATE'}
SCALE_POLICIES = {'S020': .2}
SAMPLING_VARIANTS = ('BASE', 'DAVG', 'UBIL', 'DAVG_UBIL')


def sampling_manifest(variant):
    if variant not in SAMPLING_VARIANTS:
        raise ValueError('Unregistered Student sampling variant')
    down = variant in ('DAVG', 'DAVG_UBIL')
    up = variant in ('UBIL', 'DAVG_UBIL')
    return {'schema': 'PANDA_STUDENT_SAMPLING_V1', 'variant': variant,
            'down_count': 2, 'up_count': 2,
            'down': 'AVGPOOL2_STRIDE2_THEN_CONV3_STRIDE1_PAD1' if down else 'CONV3_STRIDE2_PAD1',
            'up': 'BILINEAR_X2_ALIGN_CORNERS_FALSE_THEN_CONV3_PAD1' if up else 'CONVTRANSPOSE2_STRIDE2',
            'student_parameters': 2784328 if up else 2658888,
            'shared_parameter_initialization': 'ORIGINAL_NAMED_SEED_ROLE_VALUES',
            'teacher_sampling': 'ORIGINAL_BASE_UNCHANGED',
            'rotation_equivariance_claimed': False}


def _case(variant):
    return dict(policy_id='S020', scale=.2, alpha=.25, gamma=.5, lr_S=1e-4,
                total_updates=30000, warmup_updates=100, min_lr=0.,
                source_case=None, inherited_updates=0, schedule_kind='WARMUP_COSINE',
                sampling_variant=variant)


CASES = {variant: _case(variant) for variant in SAMPLING_VARIANTS if variant != 'BASE'}
CONTROL_CAMPAIGN = 'PANDA_S13_SCHEDULE_HARDONLY_20261023_v1'
CONTROL_CASE = 'S020C30'
