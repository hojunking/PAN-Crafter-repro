"""Stdlib-only campaign identity for the host-side sheet publisher."""
from copy import deepcopy

CAMPAIGN_ID = 'PANDA_S3_MIDPOINT_STAGE2_20261019_v2'
METHOD_REVISION = 'PANDA-MIDPOINT-DIRECT-MASK2-U23-v2'
UPSAMPLING = {
    'schema': 'DLPAN_INTERP23TAP_X4_POST_AUG_PHASE_CORRECTED_V1',
    'base_reference': 'INTERP23_MS_POST_AUG_PHASE_CORRECTED',
    'ratio': 4,
    'kernel': 'CDF_23_TAP_SEPARABLE',
    'padding': 'circular',
    'hr_sample_phase': [2, 2],
    'augmentation_order': 'LRMS_HFLIP_THEN_VFLIP_THEN_CCW_ROT90_THEN_INTERP23TAP',
    'phase_correction': 'INTEGER_ROLL_R_V_MINUS_V_WITH_V_DY_DX_HALF_HALF',
    'aligner_reference': 'PROVIDED_STORED_LMS_UNCHANGED',
    'frozen_teacher_base_reference': 'BICUBIC_MS',
    'translation_warp': 'EXISTING_BICUBIC_UNCHANGED',
}
GEOMETRY = {
    'policy': 'SYMMETRIC_HALF_SHIFT_V1', 'pan_shift_factor': .5,
    'ms_shift_factor': -.5, 'gt_shift_factor': -.5,
    'teacher_policy': 'ORIGINAL_STAGE1_FORWARD_THEN_MINUS_HALF_SHIFT',
    'inference_coordinates': 'MIDPOINT_NO_OUTPUT_RETURN_WARP',
    'metric_references': 'ORIGINAL_UNSHIFTED_UNMASKED_RR20_FR20',
    'shift_unit': 'CURRENT_PAN_PIXEL_NO_X4',
    'student_upsampling': deepcopy(UPSAMPLING),
}
