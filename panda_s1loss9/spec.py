"""Student sampling variants with a fixed positive PAN scale and existing Hard loss."""
from dataclasses import dataclass
from copy import deepcopy

from panda_stage2_condq.spec import Stage2Spec, MASK2_POLICY
from panda_s1loss9.base_reconstruction import BandReconstructionSpec, upsampler_manifest
from panda_midpoint.contract import UPSAMPLING
from .contract import (CASES, SCALE_POLICIES, METHOD_REVISION, sampling_manifest,
                       input_manifest, WARP_MANIFEST, SAMPLING_VARIANTS)
from .reconstruction import StudentReconstructionSpec

METHOD_VERSION = METHOD_REVISION
HORIZON = max(row['total_updates'] for row in CASES.values())
REGISTERED_HORIZONS = tuple(sorted({row['total_updates'] for row in CASES.values()}))
POLICY_IDS = tuple(SCALE_POLICIES)
_GEOMETRIES = {policy.lower(): scale for policy, scale in SCALE_POLICIES.items()}
_PHASES = {policy: ((1, HORIZON, policy.lower()),) for policy in POLICY_IDS}


def geometry_manifest(geometry_key, sensor=None):
    if sensor is not None and sensor not in ("QB", "GF2"):
        raise ValueError("S1Loss9 geometry requires QB or GF2")
    if geometry_key not in _GEOMETRIES:
        raise ValueError("Registered positive S1LOSS9 geometry required")
    return {
        "schema": "PANDA_NEXT30_APPLIED_GEOMETRY_V1",
        "geometry_key": geometry_key, "operator": "SCALE",
        "scale": _GEOMETRIES[geometry_key], "cap_px": None, "cap_formula": None,
        "upsampler": "interp23", "frame": "pan",
        "aligner_outputs": 2, "pan_channels": 1,
        "alignment_target": "SCHARR_OF_RGB_MEAN_GT",
        "shift_unit": "CURRENT_PAN_PIXEL_NO_X4",
        "ms_shift_factor": 0., "output_shift_factor": 0.,
        "output_coordinates": "MS", "loss_coordinates": "MS",
        "teacher_policy": "FIXED_STAGE1_FORWARD_PAN_PLUS_FULL_U",
        "training_target_policy": "ORIGINAL_GT_UNSHIFTED",
        "q_reference": "STUDENT_ACTUAL_SHIFTED_PAN_VS_ORIGINAL_RGB_GT",
        "loss_support": "STUDENT_AND_FIXED_FULL_U_TEACHER_LANCZOS4_INTERSECTION",
        "evaluation_references": "ORIGINAL_UNSHIFTED_UNMASKED_RR20_FR20",
        "pan_execution": "LANCZOS4_WITH_POSITIVE_SCALE_TIMES_FROZEN_ALIGNER_U",
        "inference_aligner_usage": "REQUIRED_FROZEN_ALIGNER_APPLIED_POSITIVE_SCALED_CORRECTION",
        "student_warp": deepcopy(WARP_MANIFEST),
    }


def view_manifest(policy):
    policy_id = policy.policy_id if isinstance(policy, MatureSpec) else policy
    if policy_id not in POLICY_IDS:
        raise ValueError("Registered S1LOSS9 policy required")
    return {
        "schema": "PANDA_S1LOSS9_SINGLE_VIEW_HARD_V1", "policy_id": policy_id,
        "student_views": 1, "companion_transform": "NONE",
        "hard_reduction": "SINGLE_CANONICAL_HARD", "consistency_lambda": 0.,
        "consistency_formula": "ABSENT_USER_EXCLUDED_NEW_METHOD",
        "gradient_policy": "SINGLE_STUDENT_VIEW_NO_TEACHER_TARGET",
        "reference_policy": "CANONICAL_A_T_E_G_Q_ECDF_ONCE",
        "base_policy": "CANONICAL_PHASE_AWARE_INTERP23",
        "support_policy": "CANONICAL_MASK2_SUPPORT",
        "gt_policy": "ORIGINAL_MS_COORDINATES_EXISTING_C4_AUGMENTATION_ONLY",
        "inference": "SINGLE_VIEW_NO_TTA_OR_ENSEMBLE",
    }


def policy_manifest(policy):
    policy_id = policy.policy_id if isinstance(policy, MatureSpec) else policy
    if policy_id not in _PHASES:
        raise ValueError("Registered S1LOSS9 policy ID required")
    return {
        "schema": "PANDA_NEXT30_POLICY_V1", "policy_id": policy_id,
        "total_updates": policy.total_updates if isinstance(policy, MatureSpec) else HORIZON, "schedule": "FIXED_POSITIVE_SCALE",
        "phases": [{"first_update": first, "last_update": last,
                    "geometry_key": key, "geometry": geometry_manifest(key)}
                   for first, last, key in ((1, policy.total_updates if isinstance(policy, MatureSpec) else HORIZON, policy_id.lower()),)],
        "evaluation_phase": "PHASE_OF_LAST_COMPLETED_UPDATE",
        "next_train_phase": "PHASE_OF_COMPLETED_UPDATE_PLUS_ONE",
        "initial_inference_phase": "FIRST_PHASE_AT_ZERO_UPDATES",
        "calibration": "EXACT_FULL_RR_TRAIN_PER_APPLIED_GEOMETRY",
        "optimizer_transition": "PRESERVE_OPTIMIZER_LR_RNG_AND_STREAM",
        "view_loss": view_manifest(policy_id),
    }


def geometry_for_update(policy_id, update):
    if type(update) is not int or not 1 <= update <= HORIZON:
        raise ValueError(f"Applied training update must be in [1,{HORIZON}]")
    if policy_id not in _PHASES:
        raise ValueError("Registered S1LOSS9 policy required")
    return policy_id.lower()


@dataclass(frozen=True)
class MatureSpec(Stage2Spec):
    mask_policy: str = MASK2_POLICY
    policy_id: str = "S100"
    upsampler: str = "interp23"
    frame: str = "pan"
    aligner_outputs: int = 2
    alignment_target: str = "SCHARR_OF_RGB_MEAN_GT"
    pan_channels: int = 1
    total_updates: int = 30000
    sampling_variant: str = "UBIL"
    input_variant: str = "CORRECTED_DIFF"
    student_width: int = 112

    def __post_init__(self):
        super().__post_init__()
        allowed = {(row["alpha"], row["gamma"]) for row in CASES.values()
                   if row["policy_id"] == self.policy_id and row["sensor"] == self.sensor}
        if (self.sensor not in ("QB", "GF2") or self.variant != "panda_condq"
                or self.mask_policy != MASK2_POLICY or self.beta != 0. or self.delta != 1e-6
                or (self.alpha, self.gamma) not in allowed):
            raise ValueError("S1LOSS9 requires registered QB/GF2 Hard alpha/gamma, beta=0, delta=1e-6 MASK2")
        if (self.upsampler != "interp23" or self.frame != "pan"
                or type(self.aligner_outputs) is not int or self.aligner_outputs != 2
                or type(self.pan_channels) is not int or self.pan_channels != 1
                or self.alignment_target != "SCHARR_OF_RGB_MEAN_GT"):
            raise ValueError("S1LOSS9 requires interp23 global2 RGB-reference PAN-only geometry")
        if type(self.total_updates) is not int or self.total_updates not in {
                row["total_updates"] for row in CASES.values() if row["sensor"] == self.sensor}:
            raise ValueError("S1Loss9 fixes the Student horizon at 30K")
        if self.sampling_variant != "UBIL" or self.input_variant != "CORRECTED_DIFF":
            raise ValueError("Registered Student sampling structure required")
        if type(self.student_width) is not int or self.student_width != 112:
            raise ValueError('S1Loss9 fixes W112 Teacher and Student')
        input_manifest(self.input_variant,self.student_width)
        if self.input_variant == 'CORRECTED_DIFF' and self.policy_id != 'S100':
            raise ValueError('Difference input uses the full Aligner correction without scale attenuation')
        policy_manifest(self)

    @property
    def geometry_keys(self):
        return (self.policy_id.lower(),)

    def geometry_at_completed(self, completed_update):
        if type(completed_update) is not int or not 0 <= completed_update <= self.total_updates:
            raise ValueError("Completed update must be in [0,total_updates]")
        return geometry_for_update(self.policy_id, max(1, completed_update))

    def geometry_at_next(self, completed_update):
        if type(completed_update) is not int or not 0 <= completed_update < self.total_updates:
            raise ValueError("No next update outside [0,total_updates-1]")
        return geometry_for_update(self.policy_id, completed_update + 1)

    @property
    def reconstruction_spec(self):
        if self.input_variant == 'CORRECTED_DIFF':
            return StudentReconstructionSpec(4, self.student_width, (1, 2, 3), "P0", "BICUBIC_MS", 2)
        return self.teacher_reconstruction_spec

    @property
    def teacher_reconstruction_spec(self):
        return BandReconstructionSpec(4, 112, (1, 2, 3), "P0", "BICUBIC_MS", 1)

    @property
    def method_version(self):
        return METHOD_VERSION

    @property
    def loss_schema(self):
        return "PANDA_NEXT30_EXISTING_GT_HARD_ONLY_E_CONDQ_MASK2_V1"

    @property
    def paired_views(self):
        return False

    @property
    def consistency_lambda(self):
        return 0.

    def manifest(self):
        result = super().manifest()
        result["student_constructor_spec"] = deepcopy(result["student"])
        result["student"] = {**result["student"],
            "base_reference": UPSAMPLING["base_reference"],
            "base_execution": "EXTERNAL_BASE_X_IN_AND_RESIDUAL_ADDITION"}
        result["student_sampling"] = sampling_manifest(self.sampling_variant, self.input_variant,self.student_width)
        result['student_input'] = input_manifest(self.input_variant,self.student_width)
        result['student_warp'] = deepcopy(WARP_MANIFEST)
        result['teacher_constructor_spec'] = self.teacher_reconstruction_spec.manifest()
        result['loss_mask_policy_token'] = 'LEGACY_REDUCTION_COMPATIBILITY_ID'
        result['actual_support_operator'] = 'TEACHER_STUDENT_LANCZOS4_INTERSECTION'
        return {**result, "student_upsampling": upsampler_manifest("interp23"),
                "mature30_policy": policy_manifest(self),
                "teacher_forward": "ORIGINAL_STAGE1_FULL_U",
                "teacher_usage": "FROZEN_ERROR_MAP_ONLY_NO_SOFT_DISTILLATION",
                "teacher_gate": "DISABLED",
                "soft_loss": "ABSENT_NO_STUDENT_TEACHER_GRADIENT",
                "q_target": "RGB_MEAN_2_1_0_UNCHANGED",
                "scale30_view_loss": view_manifest(self)}
