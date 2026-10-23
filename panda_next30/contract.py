"""One Lanczos4 Student warp for all subsequent schedules, samplers and PAN-difference inputs."""
from copy import deepcopy
from panda_mature30.contract import file_identity, POLICY, STAGE1_SELECTION

CAMPAIGN_ID = 'PANDA_S13_NEXT_HARDONLY_20261023_v1'
METHOD_REVISION = 'PANDA-NEXT30-LANCZOS4-HARDONLY-v1'
HQNR_MIN, ERGAS_MAX = .9585, 2.040
BUDGET = {'total_seconds': None, 'closeout_seconds': 0, 'safety_factor': 1.12,
          'calibration_reserve_seconds': 600, 'scope': 'FINITE_REGISTERED_QUEUE_NO_WALL_TIME_LIMIT',
          'restart': 'PRESERVE_ELAPSED_AND_COMPLETE_STATE'}
SCALE_POLICIES = {'S010': .1, 'S020': .2, 'S100': 1.}
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


from panda_schedule30.contract import CASES as _SCHEDULE_CASES
CASES = {name:{**deepcopy(value),'sampling_variant':'BASE','input_variant':'CORRECTED'}
         for name,value in _SCHEDULE_CASES.items() if value['source_case'] is None}
for _variant in ('DAVG','UBIL','DAVG_UBIL'):
    CASES[_variant] = {**deepcopy(CASES['S020C30']),'sampling_variant':_variant}
CASES['S100C30'] = {**deepcopy(CASES['S020C30']),'policy_id':'S100','scale':1.}
CASES['S100DIFFC30'] = {**deepcopy(CASES['S100C30']),'input_variant':'CORRECTED_DIFF'}
CONTROL_CAMPAIGN = 'PANDA_S13_SCHEDULE_HARDONLY_20261023_v1'
CONTROL_CASE = 'S020C30'
