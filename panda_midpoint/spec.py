"""Midpoint geometry and explicit Student-only upsampling revision."""
from copy import deepcopy
from dataclasses import dataclass

from panda_stage2_condq.spec import Stage2Spec, MASK2_POLICY
from .contract import UPSAMPLING

METHOD_VERSION = "PANDA_STAGE2_MIDPOINT_TRANSPORT_INTERP23_V2"
GEOMETRY_VERSION = "FROZEN_A_PAN_PLUS_HALF_MS_GT_MINUS_HALF_INTERP23_V2"


def geometry_manifest():
    return {
        "schema": GEOMETRY_VERSION,
        "aligner_input": "original_PAN_and_provided_stored_LMS",
        "shift_unit": "CURRENT_PAN_PIXEL",
        "pan_shift_multiplier": .5,
        "ms_base_shift_multiplier": -.5,
        "gt_shift_multiplier": -.5,
        "ms_base": "interp23tap_LRMS_x4_after_augmentation_phase_corrected_then_one_bicubic_warp",
        "student_upsampling": deepcopy(UPSAMPLING),
        "teacher_base": "original_BICUBIC_MS_unchanged",
        "teacher_target": "original_Stage1_T_warpPAN_plus_u_then_output_minus_half_u",
        "loss_frame": "MIDPOINT",
        "output_frame": "MIDPOINT_no_inverse_output_warp",
        "loss_support": "intersection_PAN_half_MS_GT_minus_half_T_two_stage_sampler_support",
        "support_scope": "bicubic_shift_sampler_support_not_interp23_circular_boundary_or_CNN_support_guarantee",
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
        manifest = super().manifest()
        # The shared constructor/Teacher contract stays byte-for-byte legacy.
        # Actual Student execution supplies a different base to BOTH x_in and
        # residual addition; never label that effective forward as BICUBIC_MS.
        constructor = deepcopy(manifest["student"])
        manifest["student"] = {**constructor, "base_reference": UPSAMPLING["base_reference"],
                               "base_execution": "external_base_x_in_and_residual_addition"}
        return {**manifest, "student_constructor_spec": constructor,
                "student_upsampling": deepcopy(UPSAMPLING),
                "midpoint_geometry": geometry_manifest()}
