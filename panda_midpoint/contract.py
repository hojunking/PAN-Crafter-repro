"""Stdlib-only campaign identity for the host-side sheet publisher."""
CAMPAIGN_ID = 'PANDA_S3_MIDPOINT_STAGE2_20261019_v1'
METHOD_REVISION = 'PANDA-MIDPOINT-DIRECT-MASK2-v1'
GEOMETRY = {
    'policy': 'SYMMETRIC_HALF_SHIFT_V1', 'pan_shift_factor': .5,
    'ms_shift_factor': -.5, 'gt_shift_factor': -.5,
    'teacher_policy': 'ORIGINAL_STAGE1_FORWARD_THEN_MINUS_HALF_SHIFT',
    'inference_coordinates': 'MIDPOINT_NO_OUTPUT_RETURN_WARP',
    'metric_references': 'ORIGINAL_UNSHIFTED_UNMASKED_RR20_FR20',
    'shift_unit': 'CURRENT_PAN_PIXEL_NO_X4',
}
