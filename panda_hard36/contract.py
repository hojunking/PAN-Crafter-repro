"""Frozen full-scale DIFF/UBIL tuning registry under a durable 36-hour budget."""
from copy import deepcopy
from panda_mature30.contract import file_identity, POLICY, STAGE1_SELECTION

CAMPAIGN_ID = 'PANDA_S13_ALPHA_GAMMA_HARDONLY_36H_20261026_v1'
METHOD_REVISION = 'PANDA-HARD36-LANCZOS4-HARDONLY-v1'
NEXT_CAMPAIGN_ID = 'PANDA_S13_NEXT_HARDONLY_20261023_v1'
RUN_PREFIX = 'PHG36'
HQNR_MIN, ERGAS_MAX = .9585, 2.040
BUDGET = {'total_seconds': 36*3600, 'closeout_seconds': 1800, 'safety_factor': 1.08,
          'calibration_reserve_seconds': 600, 'scope': 'PER_SERVER_36H_FROM_FIRST_COST_PROFILE_INCLUDING_PREFLIGHT',
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


def input_manifest(variant, width=112):
    if variant not in INPUT_VARIANTS:
        raise ValueError('Unregistered Student input variant')
    if type(width) is not int or width not in (112,128):
        raise ValueError('Registered W112/W128 Student width required')
    diff = variant == 'CORRECTED_DIFF'
    return {'schema':'PANDA_NEXT30_STUDENT_INPUT_V1', 'variant':variant,
            'channels':10 if diff else 9, 'pan_channels':2 if diff else 1,
            'order':['CORRECTED_PAN','CORRECTED_MINUS_ORIGINAL_PAN','MS_BASE_8'] if diff else ['CORRECTED_PAN','MS_BASE_8'],
            'delta_sign':'CORRECTED_MINUS_ORIGINAL' if diff else None,
            'delta_stem_initialization':'ZERO_SHARED_COLUMNS_PRESERVED' if diff else None,
            'additional_parameters':9*width if diff else 0, 'output_channels':8,
            'loss':'UNCHANGED_GT_HARD', 'extra_supervision':False}


def sampling_manifest(variant, input_variant='CORRECTED', width=112):
    if variant not in SAMPLING_VARIANTS:
        raise ValueError('Unregistered Student sampling variant')
    down = variant in ('DAVG', 'DAVG_UBIL')
    up = variant in ('UBIL', 'DAVG_UBIL')
    return {'schema': 'PANDA_STUDENT_SAMPLING_V1', 'variant': variant,
            'down_count': 2, 'up_count': 2,
            'down': 'AVGPOOL2_STRIDE2_THEN_CONV3_STRIDE1_PAD1' if down else 'CONV3_STRIDE2_PAD1',
            'up': 'BILINEAR_X2_ALIGN_CORNERS_FALSE_THEN_CONV3_PAD1' if up else 'CONVTRANSPOSE2_STRIDE2',
            'student_width':width, 'student_depth':[1,2,3],
            'student_parameters': ((220 if up else 210)*width*width + 220*width + 8
                                   + input_manifest(input_variant,width)['additional_parameters']),
            'shared_parameter_initialization': 'ORIGINAL_NAMED_SEED_ROLE_VALUES',
            'teacher_sampling': 'ORIGINAL_BASE_UNCHANGED',
            'rotation_equivariance_claimed': False}


def _fresh(scale=1., updates=30000, *, lr=1e-4, alpha=.25, gamma=.5,
           sampling='UBIL', difference=True, width=112):
    return dict(policy_id=f'S{round(scale*100):03d}', scale=scale,
        alpha=alpha, gamma=gamma, lr_S=lr, total_updates=updates, student_width=width,
        warmup_updates=100, min_lr=0., inherited_updates=0,
        schedule_kind='WARMUP_COSINE', sampling_variant=sampling,
        input_variant='CORRECTED_DIFF' if difference else 'CORRECTED',
        source_package=None, source_campaign=None, source_campaign_id=None,
        source_case=None, source_lr=None, source_horizon=None,
        source_warmup_updates=None, source_min_lr=None)




CASES = {
    'S100DIFFUBILC30AG000':_fresh(alpha=0., gamma=0.),
    'S100DIFFUBILC30G000':_fresh(gamma=0.),
    'S100DIFFUBILC30A000':_fresh(alpha=0.),
    'S100DIFFUBILC30H050':_fresh(alpha=.125, gamma=.25),
    'S100DIFFUBILC30A050':_fresh(alpha=.5),
    'S100DIFFUBILC30G100':_fresh(gamma=1.),
}

SOURCE_FIELDS = ('source_package','source_campaign','source_campaign_id','source_case',
                 'source_lr','source_horizon','source_warmup_updates','source_min_lr')
TRAINING_SCHEDULE_FIELDS = ('lr_S','total_updates','warmup_updates','min_lr',
    'inherited_updates','schedule_kind','source_lr','source_horizon',
    'source_warmup_updates','source_min_lr')


CONTROL_CAMPAIGN = 'PANDA_S13_FOLLOW_HARDONLY_20261024_v1'
CONTROL_CASE = 'S100DIFFUBILC30'
