"""Explicit, immutable contracts for full-image and MASK2 Stage 2 objectives.

No coefficient or donor is selected implicitly. This module does not authorize
or launch an experiment and does not change the live Stage 1 implementation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from panda_method.reconstruction import ReconstructionSpec

METHOD_VERSION = "PANDA_STAGE2_CONDQ_FULL_IMAGE_V1"
LOSS_SCHEMA = "PANDA_GT_ALWAYS_SELECTIVE_SOFT_CONDQ_V1"
CALIBRATION_SCHEMA = "PANDA_S2_CONDQ_TRAIN_GLOBAL_V1"
MASK_POLICY = "full_image_no_warp_valid_mask"
MASK2_POLICY = "DETACHED_BICUBIC4_SUPPORT_PER_SAMPLE_MEAN_V1"
MASK2_METHOD_VERSION = "PANDA_STAGE2_CONDQ_MASK2_V1"
MASK2_LOSS_SCHEMA = "PANDA_GT_ALWAYS_SELECTIVE_SOFT_CONDQ_MASK2_V1"
MASK2_CALIBRATION_SCHEMA = "PANDA_S2_CONDQ_TRAIN_VALID_SUPPORT_V1"
VARIANTS = ("gt_only", "panda_e", "panda_condq")
SENSOR_BANDS = {"WV3": 8, "GF2": 4, "QB": 4}


@dataclass(frozen=True)
class Stage2Spec:
    sensor: str
    variant: str
    seed: int
    alpha: float
    beta: float
    gamma: float
    delta: float
    mask_policy: str = MASK_POLICY

    def __post_init__(self):
        if self.sensor not in SENSOR_BANDS or self.variant not in VARIANTS:
            raise ValueError("Explicit supported sensor and new Stage 2 variant required")
        if self.mask_policy not in (MASK_POLICY, MASK2_POLICY):
            raise ValueError("Explicit full-image or Stage 1 MASK2 loss policy required")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("Explicit nonnegative Student initialization seed required")
        for name in ("alpha", "beta", "gamma", "delta"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(name + " must be a finite explicit coefficient")
            if value < 0 or (name == "delta" and value == 0):
                raise ValueError("Nonnegative alpha/beta/gamma and strictly positive delta required")
            object.__setattr__(self, name, float(value))
        if self.variant == "gt_only" and any((self.alpha, self.beta, self.gamma)):
            raise ValueError("gt_only requires alpha=beta=gamma=0")
        if self.variant == "panda_e" and self.gamma != 0:
            raise ValueError("panda_e cannot enable conditional q")
        if self.variant == "panda_condq" and self.gamma <= 0:
            raise ValueError("panda_condq requires an explicitly positive gamma")

    @property
    def bands(self):
        return SENSOR_BANDS[self.sensor]

    @property
    def roi_margin(self):
        return 8

    @property
    def reconstruction_spec(self):
        return ReconstructionSpec(self.bands, 112, (1, 2, 3), "P0", "BICUBIC_MS")

    @property
    def method_version(self):
        return METHOD_VERSION if self.mask_policy == MASK_POLICY else MASK2_METHOD_VERSION

    @property
    def loss_schema(self):
        return LOSS_SCHEMA if self.mask_policy == MASK_POLICY else MASK2_LOSS_SCHEMA

    @property
    def calibration_schema(self):
        return CALIBRATION_SCHEMA if self.mask_policy == MASK_POLICY else MASK2_CALIBRATION_SCHEMA

    @property
    def hard_soft_reduction(self):
        return ("full_BHW_mean_after_band_mean" if self.mask_policy == MASK_POLICY else
                "mean_channels_then_valid_pixel_mean_per_sample_then_batch_mean")

    def manifest(self):
        fields = asdict(self)
        fields.pop("mask_policy")  # Preserve the original full-image manifest exactly.
        return {**fields, "method_version": self.method_version,
                "loss_schema": self.loss_schema, "calibration_schema": self.calibration_schema,
                "student": self.reconstruction_spec.manifest(), "student_role": "S",
                "student_initialization": "fresh_random_no_teacher_copy",
                "loss_mask_policy": self.mask_policy, "hard_soft_reduction": self.hard_soft_reduction,
                "q_roi_margin": 8, "q_stencil_support_margin": 7,
                "teacher_gate": "stopgrad_positive_student_minus_teacher_over_student_plus_delta",
                "hard_weight": "1+alpha*d_e+gamma*d_e*r_q_bar",
                "separate_spatial_loss": False, "stage2_epsilon": False,
                "inference": "frozen_aligner_plus_student_no_teacher_or_GT_or_calibration"}
