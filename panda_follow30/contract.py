"""Finite follow-up registry; NEXT30 physical schemas stay compatible for exact forks."""
from copy import deepcopy
from panda_mature30.contract import file_identity, POLICY, STAGE1_SELECTION

CAMPAIGN_ID = 'PANDA_S13_FOLLOW_HARDONLY_20261024_v1'
METHOD_REVISION = 'PANDA-FOLLOW30-LANCZOS4-HARDONLY-v1'
NEXT_CAMPAIGN_ID = 'PANDA_S13_NEXT_HARDONLY_20261023_v1'
RUN_PREFIX = 'PF30'
HQNR_MIN, ERGAS_MAX = .9585, 2.040
BUDGET = {'total_seconds': None, 'closeout_seconds': 0, 'safety_factor': 1.12,
          'calibration_reserve_seconds': 600, 'scope': 'FINITE_REGISTERED_QUEUE_NO_WALL_TIME_LIMIT',
          'restart': 'PRESERVE_ELAPSED_AND_COMPLETE_STATE'}
SCALE_POLICIES = {'S100': 1.}
SAMPLING_VARIANTS = ('BASE', 'DAVG', 'UBIL', 'DAVG_UBIL')
INPUT_VARIANTS = ('CORRECTED', 'CORRECTED_DIFF')
WARP_MANIFEST = {
    'schema':'PANDA_NEXT30_STUDENT_WARP_V1', 'method':'lanczos4',
    'radius':4, 'taps_per_axis':8, 'normalization':'SUM_TO_ONE',
    'boundary':'WHOLE_SAMPLE_REFLECTION',
    'convention':'OUTPUT_YX_READ_Y_PLUS_DY_X_PLUS_DX', 'dtype':'FP32',
    'teacher_warp':'ORIGINAL_BICUBIC_BORDER', 'shift_unit':'CURRENT_PAN_PIXEL',
}


def input_manifest(variant):
    if variant not in INPUT_VARIANTS:
        raise ValueError('Unregistered Student input variant')
    diff = variant == 'CORRECTED_DIFF'
    return {'schema':'PANDA_NEXT30_STUDENT_INPUT_V1', 'variant':variant,
            'channels':10 if diff else 9, 'pan_channels':2 if diff else 1,
            'order':['CORRECTED_PAN','CORRECTED_MINUS_ORIGINAL_PAN','MS_BASE_8'] if diff else ['CORRECTED_PAN','MS_BASE_8'],
            'delta_sign':'CORRECTED_MINUS_ORIGINAL' if diff else None,
            'delta_stem_initialization':'ZERO_SHARED_COLUMNS_PRESERVED' if diff else None,
            'additional_parameters':1008 if diff else 0, 'output_channels':8,
            'loss':'UNCHANGED_GT_HARD', 'extra_supervision':False}


def sampling_manifest(variant, input_variant='CORRECTED'):
    if variant not in SAMPLING_VARIANTS:
        raise ValueError('Unregistered Student sampling variant')
    down = variant in ('DAVG', 'DAVG_UBIL')
    up = variant in ('UBIL', 'DAVG_UBIL')
    return {'schema': 'PANDA_STUDENT_SAMPLING_V1', 'variant': variant,
            'down_count': 2, 'up_count': 2,
            'down': 'AVGPOOL2_STRIDE2_THEN_CONV3_STRIDE1_PAD1' if down else 'CONV3_STRIDE2_PAD1',
            'up': 'BILINEAR_X2_ALIGN_CORNERS_FALSE_THEN_CONV3_PAD1' if up else 'CONVTRANSPOSE2_STRIDE2',
            'student_parameters': (2784328 if up else 2658888) + input_manifest(input_variant)['additional_parameters'],
            'shared_parameter_initialization': 'ORIGINAL_NAMED_SEED_ROLE_VALUES',
            'teacher_sampling': 'ORIGINAL_BASE_UNCHANGED',
            'rotation_equivariance_claimed': False}


def _fresh(scale, updates=50000, *, lr=1e-4, alpha=.25, gamma=.5,
           sampling='BASE', difference=False):
    return dict(policy_id=f'S{round(scale*100):03d}', scale=scale,
        alpha=alpha, gamma=gamma, lr_S=lr, total_updates=updates,
        warmup_updates=100, min_lr=0., inherited_updates=0,
        schedule_kind='WARMUP_COSINE', sampling_variant=sampling,
        input_variant='CORRECTED_DIFF' if difference else 'CORRECTED',
        source_package=None, source_campaign=None, source_campaign_id=None,
        source_case=None, source_lr=None, source_horizon=None,
        source_warmup_updates=None, source_min_lr=None)


def _fork(base, source_case, inherited, horizon, *, source_package='panda_next30',
          source_lr=1e-4, source_horizon=30000):
    campaign = NEXT_CAMPAIGN_ID if source_package == 'panda_next30' else CAMPAIGN_ID
    return {**deepcopy(base), 'lr_S':2e-5, 'warmup_updates':0,
        'min_lr':1e-6,
        'total_updates':horizon, 'inherited_updates':inherited,
        'schedule_kind':'SOURCE_THEN_COSINE',
        'source_package':source_package, 'source_campaign':campaign,
        'source_campaign_id':campaign, 'source_case':source_case,
        'source_lr':source_lr, 'source_horizon':source_horizon,
        'source_warmup_updates':100, 'source_min_lr':0.}


CASES = {
    'S100DIFFUBILC30':_fresh(1., 30000, sampling='UBIL', difference=True),
    'S100UBILC30':_fresh(1., 30000, sampling='UBIL'),
    'S100DIFFC50':_fresh(1., difference=True),
    'S100DIFFLR150C50':_fresh(1., lr=1.5e-4, difference=True),
    'S100DIFFH050C50':_fresh(1., alpha=.125, gamma=.25, difference=True),
    'S100DIFFC30':_fresh(1., 30000, difference=True),
    'S100C30':_fresh(1., 30000),
}
CASES['S100DIFFF15C30'] = _fork(CASES['S100DIFFC30'], 'S100DIFFC30', 15000, 30000)

SOURCE_FIELDS = ('source_package','source_campaign','source_campaign_id','source_case',
                 'source_lr','source_horizon','source_warmup_updates','source_min_lr')
TRAINING_SCHEDULE_FIELDS = ('lr_S','total_updates','warmup_updates','min_lr',
    'inherited_updates','schedule_kind','source_lr','source_horizon',
    'source_warmup_updates','source_min_lr')


CONTROL_CAMPAIGN = NEXT_CAMPAIGN_ID
CONTROL_CASE = 'S100DIFFC30'
