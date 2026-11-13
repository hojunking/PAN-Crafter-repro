"""Host-safe unified scientific protocol."""
WARP_CONTRACT='lanczos4_whole_sample_reflection_FP32_current_pan_px'
MASK2_POLICY='DETACHED_LANCZOS4_INTEGER_TAP_SUPPORT_ERODE3X3_COMMON_V2'
def warp_manifest():
    return {'schema':'UNIFIED_L4_COMMON_SUPPORT_V1','method':'lanczos4','radius':4,'taps_per_axis':8,
        'normalization':'SUM_TO_ONE','boundary':'WHOLE_SAMPLE_REFLECTION','dtype':'FP32',
        'convention':'OUTPUT_YX_READ_Y_PLUS_DY_X_PLUS_DX','shift_unit':'CURRENT_PAN_PIXEL',
        'paths':['stage1_native','stage1_epsilon_injection','validation_native','validation_epsilon_injection',
                 'stage2_teacher','stage2_student','calibration_teacher','calibration_student','RR_FR_inference'],
        'aligner_input_crop':8,'loss_mask':'erode3x3(full_Lanczos4_tap_support)',
        'mask_policy':MASK2_POLICY,'shift_clamping':False,'support_indices':'pixel_int64+floor(shift).int64'}
