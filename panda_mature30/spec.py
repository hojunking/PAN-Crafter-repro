"""Immutable mature-donor policies; no per-update mutation of loss contracts."""
from dataclasses import dataclass
from copy import deepcopy

from panda_stage2_condq.spec import Stage2Spec, MASK2_POLICY
from panda_band30.reconstruction import BandReconstructionSpec, upsampler_manifest
from panda_midpoint.contract import UPSAMPLING

METHOD_VERSION = "PANDA_MATURE30_HARDONLY_STAGE2_V2"
HORIZON = 50000
POLICY_IDS = ("FULL", "HALF", "DOWN", "LOW", "CAP085")
_GEOMETRIES = {
    "s100": ("SCALE", 1., None),
    "s050": ("SCALE", .5, None),
    "s0375": ("SCALE", .375, None),
    "s025": ("SCALE", .25, None),
    "cap085": ("RADIAL_CAP", None, .85),
}
_PHASES = {
    "FULL": ((1, HORIZON, "s100"),),
    "HALF": ((1, HORIZON, "s050"),),
    "DOWN": ((1, 10000, "s050"), (10001, 30000, "s0375"),
             (30001, HORIZON, "s025")),
    "LOW": ((1, HORIZON, "s025"),),
    "CAP085": ((1, HORIZON, "cap085"),),
}


def geometry_manifest(geometry_key):
    if geometry_key not in _GEOMETRIES:
        raise ValueError("Registered mature applied geometry required")
    operator, scale, cap = _GEOMETRIES[geometry_key]
    return {
        "schema": "PANDA_MATURE30_APPLIED_GEOMETRY_V1",
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
    }


def policy_manifest(policy):
    policy_id = policy.policy_id if isinstance(policy, MatureSpec) else policy
    if policy_id not in _PHASES:
        raise ValueError("Registered mature policy ID required")
    return {
        "schema": "PANDA_MATURE30_POLICY_V1", "policy_id": policy_id,
        "total_updates": HORIZON, "schedule": "PIECEWISE_CONSTANT",
        "phases": [{"first_update": first, "last_update": last,
                    "geometry_key": key, "geometry": geometry_manifest(key)}
                   for first, last, key in _PHASES[policy_id]],
        "evaluation_phase": "PHASE_OF_LAST_COMPLETED_UPDATE",
        "next_train_phase": "PHASE_OF_COMPLETED_UPDATE_PLUS_ONE",
        "initial_inference_phase": "FIRST_PHASE_AT_ZERO_UPDATES",
        "calibration": "EXACT_FULL_RR_TRAIN_PER_APPLIED_GEOMETRY",
        "optimizer_transition": "PRESERVE_OPTIMIZER_LR_RNG_AND_STREAM",
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
    policy_id: str = "FULL"
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
            raise ValueError("Mature30 hard-only requires WV3 conditional-q .25/0/.5/1e-6 MASK2")
        if (self.upsampler != "interp23" or self.frame != "pan"
                or type(self.aligner_outputs) is not int or self.aligner_outputs != 2
                or type(self.pan_channels) is not int or self.pan_channels != 1
                or self.alignment_target != "SCHARR_OF_MEAN8_GT"):
            raise ValueError("Mature30 requires interp23 global2 mean8 PAN-only geometry")
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
        return "PANDA_GT_HARD_ONLY_E_CONDQ_MASK2_V2"

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
                "q_target": "RGB_MEAN_4_2_1_UNCHANGED"}
