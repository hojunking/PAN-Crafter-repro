"""Frozen sensor-specific experimental registration; no implicit launch."""
from copy import deepcopy
from panda_mature30.contract import file_identity, STAGE1_SELECTION
from panda_band30.contract import HQNRPolicy

CAMPAIGN_ID = 'PANDA_S13_SENSOR_REFINE_R34_20261028_v1'
METHOD_REVISION = 'PANDA-SENSOR-REFINE-R34-RGB-L4-DIFFUBIL-HARDONLY-v1'
RUN_PREFIX = 'PSRF'
NEXT_CAMPAIGN_ID = CAMPAIGN_ID
LANES = {
    's1': {'dataset':'QB', 'hqnr_min':.9315, 'ergas_max':3.5,
           'peak':2047, 'train_count':17139, 'val_count':1905, 'long_updates':40000},
    's3': {'dataset':'GF2', 'hqnr_min':.9645, 'ergas_max':.522,
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
    'schema':'PANDA_SENSOR_REFINE_UNIFIED_WARP_V2', 'method':'lanczos4',
    'radius':4, 'taps_per_axis':8, 'normalization':'SUM_TO_ONE',
    'boundary':'WHOLE_SAMPLE_REFLECTION',
    'convention':'OUTPUT_YX_READ_Y_PLUS_DY_X_PLUS_DX', 'dtype':'FP32',
    'teacher_warp':'SAME_LANCZOS4_REFLECTION', 'epsilon_warp':'SAME_LANCZOS4_REFLECTION', 'shift_unit':'CURRENT_PAN_PIXEL',
}

def input_manifest(variant, width=112, bands=4):
    if variant not in INPUT_VARIANTS or width != 112 or bands != 4:
        raise ValueError('SensorRefine requires registered W112 four-band inputs')
    diff = variant == 'CORRECTED_DIFF'
    return {'schema':'PANDA_SENSOR_REFINE_STUDENT_INPUT_V1', 'variant':variant,
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
    return {'schema':'PANDA_SENSOR_REFINE_STUDENT_SAMPLING_V1', 'variant':variant,
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

# All refinements preserve full Aligner use. GF2 is an explicit transfer
# hypothesis; its center is not claimed to be a sensor-specific optimum.
LEGACY_CAMPAIGN_ID = 'PANDA_S13_S1LOSS9_RGB_L4_STAGE12_20261026_v1'
EXTERNAL_DONOR = 'S1_A200_E030'
LOSS_PAIRS = (
    ('A200_E030', 2., .03),
    ('A200_E030_R4', 2., .03),
    ('A200_E010_R3', 2., .01),
    ('A100_E030_R3', 1., .03),
)
PAIR_PARAMETERS = {key: {'lambda_abs': absolute, 'lambda_epsilon': epsilon}
                   for key, absolute, epsilon in LOSS_PAIRS}
PAIR_RADII = {key: 4. if key.endswith('_R4') else 3. for key in PAIR_PARAMETERS}
DONORS = tuple('S1_' + key for key, _, _ in LOSS_PAIRS if key != 'A200_E030')
ALL_DONORS = (EXTERNAL_DONOR,) + DONORS
STUDENT_ORDER = ('BASE', 'LR120', 'C40', 'H050', 'A200_E030_R4',
                 'A200_E010_R3', 'A100_E030_R3')
CASES = {}
for lane in LANES.values():
    sensor = lane['dataset']
    for original_name in STUDENT_ORDER:
        name = ({'LR120': 'LR150', 'C40': 'C60'}.get(original_name, original_name)
                if sensor == 'GF2' else original_name)
        pair = name if name in PAIR_PARAMETERS else 'A200_E030'
        knobs = _fresh(sensor, updates=60000 if name == 'C60' else 40000 if name == 'C40' else 30000,
                       lr=1.5e-4 if name == 'LR150' else 1.2e-4 if name == 'LR120' else 1e-4,
                       alpha=.125 if name == 'H050' else .25,
                       gamma=.25 if name == 'H050' else .5)
        CASES[sensor + '_' + name] = {**knobs, 'loss_pair': pair,
            'donor_case': 'S1_' + pair, **PAIR_PARAMETERS[pair],
            'epsilon_radius_rr': PAIR_RADII[pair],
            'replica': 'R2' if name == 'BASE' else 'R1',
            'external_donor': pair == 'A200_E030'}


def stage1_pair(case):
    if case not in ALL_DONORS:
        raise ValueError('Registered Stage1 coefficient/radius case required')
    return case.removeprefix('S1_')


def validate_loss_pair_config(cfg):
    pair = cfg.get('loss_pair')
    expected = PAIR_PARAMETERS.get(pair)
    if expected is None or cfg.get('stage1_loss_parameters') != expected:
        raise ValueError('Registered Stage1 loss pair and exact parameters required')
    if (cfg.get('epsilon_radius_rr') != PAIR_RADII[pair]
            or cfg.get('epsilon_distribution') != 'UNIFORM_AREA_DISK'):
        raise ValueError('Registered epsilon radius/distribution required')
    if cfg.get('donor_version') != 'S1_' + pair:
        raise ValueError('Student and its Stage1 coefficient/radius pair differ')
    if cfg.get('phase') == 'donor':
        if (cfg.get('case') not in DONORS or cfg['case'] != 'S1_' + pair
                or any(cfg.get(k) != v for k, v in expected.items())
                or cfg.get('lambda_eps_RR') != expected['lambda_epsilon']):
            raise ValueError('Stage1 case/coefficient values differ from registration')
    elif cfg.get('phase') == 'stage2':
        case = CASES.get(cfg.get('case'))
        if (case is None or case['loss_pair'] != pair
                or cfg.get('replica') != case['replica']
                or cfg.get('lambda_abs') != 0. or cfg.get('lambda_epsilon') != 0.
                or cfg.get('lambda_eps_RR') != 0.):
            raise ValueError('Student case differs or extra Stage1 loss entered Stage2')
    else:
        raise ValueError('Registered donor or stage2 phase required')
    return expected


def teacher_radius(teacher, binding=None):
    if teacher.get('campaign_id') == LEGACY_CAMPAIGN_ID:
        # Original authenticated loader verifies the stored R3 pipeline. Do
        # not rewrite its original binding to impersonate this campaign.
        if teacher.get('epsilon_radius_rr', 3.) != 3.:
            raise ValueError('Legacy Stage1 donor is exclusively radius3')
        return 3.
    return teacher.get('epsilon_radius_rr')


def validate_source_teacher(binding):
    teacher = binding.get('band30_teacher', {})
    campaign = teacher.get('campaign_id')
    if (campaign not in (CAMPAIGN_ID, LEGACY_CAMPAIGN_ID)
            or binding.get('source_campaign_id') != campaign
            or binding.get('mature30_teacher') != teacher):
        raise ValueError('Registered actual donor source campaign required')
    values = (teacher.get('lambda_abs'), teacher.get('lambda_epsilon'), teacher_radius(teacher))
    allowed = {(2., .03, 3.)} if campaign == LEGACY_CAMPAIGN_ID else {
        (PAIR_PARAMETERS[p]['lambda_abs'], PAIR_PARAMETERS[p]['lambda_epsilon'], PAIR_RADII[p])
        for p in PAIR_PARAMETERS if p != 'A200_E030'}
    if values not in allowed:
        raise ValueError('Unregistered source coefficients/radius')
    if any(teacher.get(k) != v for k,v in {
            'upsampler':'interp23','aligner_outputs':2,'pan_channels':1,
            'alignment_target':'SCHARR_OF_RGB_MEAN_GT'}.items()):
        raise ValueError('Registered RGB/L4/global2 source required')
    return teacher


def validate_donor_loss_pair(cfg, binding):
    expected = validate_loss_pair_config(cfg)
    teacher = validate_source_teacher(binding)
    external = cfg['donor_version'] == EXTERNAL_DONOR
    source_campaign = LEGACY_CAMPAIGN_ID if external else CAMPAIGN_ID
    if (teacher.get('campaign_id') != source_campaign
            or binding.get('source_campaign_id') != source_campaign
            or any(teacher.get(key) != value for key, value in expected.items())
            or teacher_radius(teacher) != cfg['epsilon_radius_rr']):
        raise ValueError('Frozen A/T coefficients, radius or source differ from Student')
    if binding.get('mature30_teacher') != teacher:
        raise ValueError('Frozen teacher coefficient/radius bindings disagree')
    return expected

SOURCE_FIELDS = ('source_package','source_campaign','source_campaign_id','source_case',
                 'source_lr','source_horizon','source_warmup_updates','source_min_lr')
TRAINING_SCHEDULE_FIELDS = ('lr_S','total_updates','warmup_updates','min_lr',
    'inherited_updates','schedule_kind','source_lr','source_horizon','source_warmup_updates','source_min_lr')
CONTROL_CAMPAIGN = CAMPAIGN_ID
CONTROL_CASE = 'A200_E030'
