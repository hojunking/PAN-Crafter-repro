"""Immutable ZERO / paired-view / rotation-consistency loss contracts."""
from dataclasses import dataclass
from copy import deepcopy

from panda_stage2_condq.spec import Stage2Spec, MASK2_POLICY
from panda_band30.reconstruction import BandReconstructionSpec, upsampler_manifest
from panda_midpoint.contract import UPSAMPLING

METHOD_VERSION = "PANDA_ZERO30_HARDONLY_STAGE2_V1"
HORIZON = 50000
POLICY_IDS = ("ZERO", "PAIR0", "ROTCONS")
_GEOMETRIES = {
    "s000": ("SCALE", 0., None),
}
_PHASES = {policy: ((1, HORIZON, "s000"),) for policy in POLICY_IDS}


def view_manifest(policy):
    policy_id = policy.policy_id if isinstance(policy, MatureSpec) else policy
    if policy_id not in POLICY_IDS:
        raise ValueError("Registered ZERO30 policy required")
    paired = policy_id != "ZERO"
    return {
        "schema": "PANDA_ZERO30_VIEW_LOSS_V1", "policy_id": policy_id,
        "student_views": 2 if paired else 1,
        "companion_transform": "EXACT_ROT90_CCW" if paired else "NONE",
        "hard_reduction": "MEAN_OF_TWO_VIEW_HARD" if paired else "SINGLE_CANONICAL_HARD",
        "consistency_lambda": .05 if policy_id == "ROTCONS" else 0.,
        "consistency_formula": "mean_MASK2(mean_bands(abs(y-R_inverse(y_R))))",
        "consistency_units": "NORMALIZED_RADIOMETRY_MINUS1_PLUS1_SAME_AS_HARD",
        "gradient_policy": ("BOTH_STUDENT_VIEWS_NO_STOPGRAD_NO_TEACHER_TARGET" if paired else
                            "SINGLE_STUDENT_VIEW_NO_TEACHER_TARGET"),
        "reference_policy": ("CANONICAL_A_T_E_G_Q_ECDF_ONCE_THEN_EXACT_ROTATION" if paired else
                             "CANONICAL_A_T_E_G_Q_ECDF_ONCE"),
        "base_policy": ("CANONICAL_PHASE_AWARE_INTERP23_THEN_EXACT_ROTATION" if paired else
                        "CANONICAL_PHASE_AWARE_INTERP23"),
        "support_policy": ("CANONICAL_MASK_INTERSECT_INVERSE_ROTATED_COMPANION_MASK" if paired else
                           "CANONICAL_MASK2_SUPPORT"),
        "gt_policy": "ORIGINAL_MS_COORDINATES_ROTATION_AUGMENTATION_ONLY",
        "inference": "SINGLE_VIEW_NO_TTA_OR_ENSEMBLE",
    }


def geometry_manifest(geometry_key):
    if geometry_key not in _GEOMETRIES:
        raise ValueError("Registered mature applied geometry required")
    operator, scale, cap = _GEOMETRIES[geometry_key]
    return {
        "schema": "PANDA_ZERO30_APPLIED_GEOMETRY_V1",
        "geometry_key": geometry_key, "operator": operator,
        "scale": scale, "cap_px": cap,
        "cap_formula": "u*min(1,cap_px/L2(u))" if cap is not None else None,
        "upsampler": "interp23", "frame": "pan",
        "aligner_outputs": 2, "pan_channels": 1,
        "alignment_target": "SCHARR_OF_MEAN8_GT",
        "shift_unit": "CURRENT_PAN_PIXEL_NO_X4",
        "ms_shift_factor": 0., "output_shift_factor": 0.,
        "output_coordinates": "MS", "loss_coordinates": "MS",
        "teacher_policy": "FIXED_STAGE1_FORWARD_PAN_PLUS_FULL_U",
        "training_target_policy": "ORIGINAL_GT_UNSHIFTED",
        "q_reference": "STUDENT_ACTUAL_SHIFTED_PAN_VS_ORIGINAL_RGB_GT",
        "loss_support": "STUDENT_PAN_AND_FIXED_FULL_U_TEACHER_BICUBIC_INTERSECTION",
        "evaluation_references": "ORIGINAL_UNSHIFTED_UNMASKED_RR20_FR20",
        "zero_pan_execution": "EXACT_ORIGINAL_PAN_NO_IDENTITY_RESAMPLING",
        "inference_aligner_usage": "DIAGNOSTICS_ONLY_APPLIED_CORRECTION_ZERO",
    }


def policy_manifest(policy):
    policy_id = policy.policy_id if isinstance(policy, MatureSpec) else policy
    if policy_id not in _PHASES:
        raise ValueError("Registered mature policy ID required")
    return {
        "schema": "PANDA_ZERO30_POLICY_V1", "policy_id": policy_id,
        "total_updates": HORIZON, "schedule": "PIECEWISE_CONSTANT",
        "phases": [{"first_update": first, "last_update": last,
                    "geometry_key": key, "geometry": geometry_manifest(key)}
                   for first, last, key in _PHASES[policy_id]],
        "evaluation_phase": "PHASE_OF_LAST_COMPLETED_UPDATE",
        "next_train_phase": "PHASE_OF_COMPLETED_UPDATE_PLUS_ONE",
        "initial_inference_phase": "FIRST_PHASE_AT_ZERO_UPDATES",
        "calibration": "EXACT_FULL_RR_TRAIN_PER_APPLIED_GEOMETRY",
        "optimizer_transition": "PRESERVE_OPTIMIZER_LR_RNG_AND_STREAM",
        "view_loss": view_manifest(policy_id),
    }


def geometry_for_update(policy_id, update):
    if type(update) is not int or not 1 <= update <= HORIZON:
        raise ValueError("Applied training update must be in [1,50000]")
    if policy_id not in _PHASES:
        raise ValueError("Registered mature policy required")
    return next(key for first, last, key in _PHASES[policy_id] if first <= update <= last)


@dataclass(frozen=True)
class MatureSpec(Stage2Spec):
    mask_policy: str = MASK2_POLICY
    policy_id: str = "ZERO"
    upsampler: str = "interp23"
    frame: str = "pan"
    aligner_outputs: int = 2
    alignment_target: str = "SCHARR_OF_MEAN8_GT"
    pan_channels: int = 1

    def __post_init__(self):
        super().__post_init__()
        if (self.sensor != "WV3" or self.variant != "panda_condq"
                or self.mask_policy != MASK2_POLICY
                or (self.alpha, self.beta, self.gamma, self.delta) != (.25, 0., .5, 1e-6)):
            raise ValueError("Zero30 hard-only requires WV3 conditional-q .25/0/.5/1e-6 MASK2")
        if (self.upsampler != "interp23" or self.frame != "pan"
                or type(self.aligner_outputs) is not int or self.aligner_outputs != 2
                or type(self.pan_channels) is not int or self.pan_channels != 1
                or self.alignment_target != "SCHARR_OF_MEAN8_GT"):
            raise ValueError("Zero30 requires interp23 global2 mean8 PAN-only geometry")
        policy_manifest(self.policy_id)

    @property
    def geometry_keys(self):
        return tuple(dict.fromkeys(key for _, _, key in _PHASES[self.policy_id]))

    def geometry_at_completed(self, completed_update):
        if type(completed_update) is not int or not 0 <= completed_update <= HORIZON:
            raise ValueError("Completed update must be in [0,50000]")
        return geometry_for_update(self.policy_id, max(1, completed_update))

    def geometry_at_next(self, completed_update):
        if type(completed_update) is not int or not 0 <= completed_update < HORIZON:
            raise ValueError("No next update outside [0,49999]")
        return geometry_for_update(self.policy_id, completed_update + 1)

    @property
    def reconstruction_spec(self):
        return BandReconstructionSpec(8, 112, (1, 2, 3), "P0", "BICUBIC_MS", 1)

    @property
    def method_version(self):
        return METHOD_VERSION

    @property
    def loss_schema(self):
        return "PANDA_ZERO30_GT_HARD_ONLY_CONDQ_ROTATION_MASK2_V1"

    @property
    def paired_views(self):
        return self.policy_id != "ZERO"

    @property
    def consistency_lambda(self):
        return .05 if self.policy_id == "ROTCONS" else 0.

    def manifest(self):
        result = super().manifest()
        result["student_constructor_spec"] = deepcopy(result["student"])
        result["student"] = {**result["student"],
            "base_reference": UPSAMPLING["base_reference"],
            "base_execution": "EXTERNAL_BASE_X_IN_AND_RESIDUAL_ADDITION"}
        return {**result, "student_upsampling": upsampler_manifest("interp23"),
                "mature30_policy": policy_manifest(self),
                "teacher_forward": "ORIGINAL_STAGE1_FULL_U",
                "teacher_usage": "FROZEN_ERROR_MAP_ONLY_NO_SOFT_DISTILLATION",
                "teacher_gate": "DISABLED",
                "soft_loss": "ABSENT_NO_STUDENT_TEACHER_GRADIENT",
                "q_target": "RGB_MEAN_4_2_1_UNCHANGED",
                "zero30_view_loss": view_manifest(self)}

