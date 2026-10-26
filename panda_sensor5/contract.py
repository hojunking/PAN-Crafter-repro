"""Frozen sensor-specific experimental registration; no implicit launch."""
from copy import deepcopy
from panda_mature30.contract import file_identity, STAGE1_SELECTION
from panda_band30.contract import HQNRPolicy

CAMPAIGN_ID = 'PANDA_S13_SENSOR5_STAGE12_20261026_v1'
METHOD_REVISION = 'PANDA-SENSOR5-MEAN4-DIFFUBIL-HARDONLY-v1'
RUN_PREFIX = 'PSN5'
NEXT_CAMPAIGN_ID = CAMPAIGN_ID
LANES = {
    's1': {'dataset':'QB', 'hqnr_min':.9315, 'ergas_max':3.5,
           'peak':2047, 'train_count':17139, 'val_count':1905, 'long_updates':50000},
    's3': {'dataset':'GF2', 'hqnr_min':.9645, 'ergas_max':.548,
           'peak':1023, 'train_count':19809, 'val_count':2201, 'long_updates':60000},
}
POLICY = HQNRPolicy()  # All evaluated files retained; publication selects by sensor target.

def lane_for_sensor(sensor):
    return next(v for v in LANES.values() if v['dataset'] == sensor)

def policy(sensor):
    lane_for_sensor(sensor)
    return POLICY

def target_gap(sensor, hqnr, ergas):
    row = lane_for_sensor(sensor)
    return max(0., row['hqnr_min']-hqnr), max(0., ergas-row['ergas_max'])

BUDGET = {'total_seconds':None, 'closeout_seconds':0, 'safety_factor':1.08,
          'calibration_reserve_seconds':600, 'scope':'COMPLETE_REGISTERED_QUEUE_NO_TIME_CAP',
          'restart':'EXACT_CHECKPOINT_RESUME_NO_REINITIALIZATION'}
SCALE_POLICIES = {'S100':1.}
SAMPLING_VARIANTS = ('BASE','DAVG','UBIL','DAVG_UBIL')
INPUT_VARIANTS = ('CORRECTED','CORRECTED_DIFF')
WARP_MANIFEST = {
    'schema':'PANDA_SENSOR5_STUDENT_WARP_V1', 'method':'lanczos4',
    'radius':4, 'taps_per_axis':8, 'normalization':'SUM_TO_ONE',
    'boundary':'WHOLE_SAMPLE_REFLECTION',
    'convention':'OUTPUT_YX_READ_Y_PLUS_DY_X_PLUS_DX', 'dtype':'FP32',
    'teacher_warp':'ORIGINAL_BICUBIC_BORDER', 'shift_unit':'CURRENT_PAN_PIXEL',
}

def input_manifest(variant, width=112, bands=4):
    if variant not in INPUT_VARIANTS or width != 112 or bands != 4:
        raise ValueError('Sensor5 requires registered W112 four-band inputs')
    diff = variant == 'CORRECTED_DIFF'
    return {'schema':'PANDA_SENSOR5_STUDENT_INPUT_V1', 'variant':variant,
        'channels':bands+(2 if diff else 1), 'pan_channels':2 if diff else 1,
        'order':['CORRECTED_PAN','CORRECTED_MINUS_ORIGINAL_PAN','MS_BASE_4'] if diff else ['CORRECTED_PAN','MS_BASE_4'],
        'delta_sign':'CORRECTED_MINUS_ORIGINAL' if diff else None,
        'delta_stem_initialization':'ZERO_SHARED_COLUMNS_PRESERVED' if diff else None,
        'additional_parameters':9*width if diff else 0, 'output_channels':bands,
        'loss':'UNCHANGED_GT_HARD', 'extra_supervision':False}

def sampling_manifest(variant, input_variant='CORRECTED', width=112, bands=4):
    if variant not in SAMPLING_VARIANTS:
        raise ValueError('Unregistered sampling variant')
    down = variant in ('DAVG','DAVG_UBIL'); up = variant in ('UBIL','DAVG_UBIL')
    return {'schema':'PANDA_SENSOR5_STUDENT_SAMPLING_V1', 'variant':variant,
        'down_count':2, 'up_count':2,
        'down':'AVGPOOL2_STRIDE2_THEN_CONV3_STRIDE1_PAD1' if down else 'CONV3_STRIDE2_PAD1',
        'up':'BILINEAR_X2_ALIGN_CORNERS_FALSE_THEN_CONV3_PAD1' if up else 'CONVTRANSPOSE2_STRIDE2',
        'student_width':width, 'student_depth':[1,2,3], 'bands':bands,
        'student_parameters':((220 if up else 210)*width*width + 148*width + 4
                              + input_manifest(input_variant,width,bands)['additional_parameters']),
        'shared_parameter_initialization':'ORIGINAL_NAMED_SEED_ROLE_VALUES',
        'teacher_sampling':'ORIGINAL_BASE_UNCHANGED', 'rotation_equivariance_claimed':False,
        'student_input':input_manifest(input_variant,width,bands)}

def _fresh(sensor, *, updates=30000, lr=1e-4, alpha=.25, gamma=.5):
    return dict(sensor=sensor, policy_id='S100', scale=1., alpha=alpha, gamma=gamma,
        lr_S=lr,total_updates=updates,student_width=112,warmup_updates=100,min_lr=0.,
        inherited_updates=0,schedule_kind='WARMUP_COSINE',sampling_variant='UBIL',
        input_variant='CORRECTED_DIFF',source_package=None,source_campaign=None,
        source_campaign_id=None,source_case=None,source_lr=None,source_horizon=None,
        source_warmup_updates=None,source_min_lr=None)

CASES = {}
for _lane in LANES.values():
    _sensor = _lane['dataset']
    CASES.update({
        _sensor+'_BASE':_fresh(_sensor),
        _sensor+'_LR050':_fresh(_sensor,lr=5e-5),
        _sensor+'_LR200':_fresh(_sensor,lr=2e-4),
        _sensor+'_LONG':_fresh(_sensor,updates=_lane['long_updates']),
        _sensor+'_HARD4':_fresh(_sensor,alpha=1.,gamma=2.),
    })
SOURCE_FIELDS = ('source_package','source_campaign','source_campaign_id','source_case',
                 'source_lr','source_horizon','source_warmup_updates','source_min_lr')
TRAINING_SCHEDULE_FIELDS = ('lr_S','total_updates','warmup_updates','min_lr',
    'inherited_updates','schedule_kind','source_lr','source_horizon','source_warmup_updates','source_min_lr')
CONTROL_CAMPAIGN = CAMPAIGN_ID
CONTROL_CASE = 'BASE'
