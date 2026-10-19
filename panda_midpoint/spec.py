"""Geometry-only revision of the existing conditional-q Stage 2 loss."""
from dataclasses import dataclass

from panda_stage2_condq.spec import Stage2Spec, MASK2_POLICY

METHOD_VERSION = "PANDA_STAGE2_MIDPOINT_TRANSPORT_V1"
GEOMETRY_VERSION = "FROZEN_A_PAN_PLUS_HALF_MS_GT_MINUS_HALF_V1"


def geometry_manifest():
    return {
        "schema": GEOMETRY_VERSION,
        "aligner_input": "original_PAN_and_provided_stored_LMS",
        "shift_unit": "CURRENT_PAN_PIXEL",
        "pan_shift_multiplier": .5,
        "ms_base_shift_multiplier": -.5,
        "gt_shift_multiplier": -.5,
        "ms_base": "bicubic_LRMS_x4_align_corners_false_then_one_warp",
        "teacher_target": "original_Stage1_T_warpPAN_plus_u_then_output_minus_half_u",
        "loss_frame": "MIDPOINT",
        "output_frame": "MIDPOINT_no_inverse_output_warp",
        "loss_support": "intersection_PAN_half_MS_GT_minus_half_T_two_stage_sampler_support",
        "support_scope": "bicubic_sampler_support_not_CNN_receptive_field_guarantee",
        "evaluation_references": "original_unshifted_PAN_LMS_MS_GT_no_alignment_mask",
    }


@dataclass(frozen=True)
class MidpointSpec(Stage2Spec):
    mask_policy: str = MASK2_POLICY

    def __post_init__(self):
        super().__post_init__()
        if self.sensor != "WV3" or self.mask_policy != MASK2_POLICY:
            raise ValueError("This midpoint campaign preserves WV3/MASK2 only")

    @property
    def method_version(self):
        return METHOD_VERSION

    def manifest(self):
        # Keep the original loss formula, coefficients, and calibration format.
        # Artifact binding additionally pins the new coordinate/target function.
        return {**super().manifest(), "midpoint_geometry": geometry_manifest()}
