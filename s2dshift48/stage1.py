"""Isolated S2D donor Stage 1; the existing objectives are reused unchanged.

The class hierarchy intentionally shares the existing native warp, detached
epsilon target, Scharr support guard and optional MASK2 reduction. It does not
wrap an old trained donor or turn an existing campaign into an S2D experiment.
"""
from __future__ import annotations

from dataclasses import asdict

import torch

from directshift48.method import DirectShiftStage1, DirectTrainingSources
from donor30.masking import MaskedDonorStage1, mask_manifest
from donor30.plan import LOSS_MASK_POLICY
from panda_method.reconstruction import ReconstructionSpec, Reconstructor, build_reconstructor

from .aligner import METHOD_VERSION, S2DGlobalPANAligner, boundary_manifest


METHOD_REVISION = "S2DGlobalShiftR3-S1-v1"
MASKED_METHOD_REVISION = "S2DGlobalShiftR3-MaskedDonor-v1"
MASK_POLICIES = ("MASK2", "UNMASKED")


def _check_teacher_spec(spec):
    if (not isinstance(spec, ReconstructionSpec) or spec.width != 112
            or spec.depth != (1, 2, 3) or spec.input_layout != "P0"
            or spec.base_reference != "BICUBIC_MS"):
        raise ValueError("S2D comparison requires unchanged W112/D123/P0/BICUBIC_MS Teacher")


class S2DShiftStage1(DirectShiftStage1):
    """Fresh S2D Aligner and unchanged Teacher, full-image reconstruction L1."""

    def __init__(self, aligner, teacher, *, sources, epsilon_seed, roi_margin=8,
                 boundary_policy="require_spatial_support", lambda_abs=1., lambda_eps=.001):
        if type(aligner) is not S2DGlobalPANAligner:
            raise TypeError("Explicit S2DGlobalPANAligner required; old donor conversion is forbidden")
        if not isinstance(teacher, Reconstructor):
            raise TypeError("Unchanged PANCrafter Reconstructor required")
        _check_teacher_spec(teacher.spec)
        super().__init__(aligner, teacher, sources=sources, epsilon_seed=epsilon_seed,
                         roi_margin=roi_margin, boundary_policy=boundary_policy,
                         lambda_abs=lambda_abs, lambda_eps=lambda_eps)

    def alignment_manifest(self):
        result = super().alignment_manifest()
        result.update(method=METHOD_VERSION, aligner=self.aligner.manifest(),
                      architecture="S2D4_before_both_stems_normalized_before_S2D_joint_stride1",
                      boundary=boundary_manifest())
        return result

    def pipeline_manifest(self):
        result = super().pipeline_manifest()
        result.update(method=METHOD_VERSION, revision=METHOD_REVISION,
                      aligner=self.aligner.manifest(), boundary=boundary_manifest(),
                      mask_policy="UNMASKED")
        return result

    def get_extra_state(self):
        return {"method": METHOD_VERSION, "pipeline": self.pipeline_manifest(),
                "sources": asdict(self.sources), "epsilon_seed": self.epsilon_seed,
                "epsilon_rng": self._epsilon_rng.get_state().clone(),
                "forward_calls": self.forward_calls}

    def _check_observation(self, batch):
        super()._check_observation(batch)
        # The old adapter accepts FP64 for numerical tests. The new architecture
        # is explicitly FP32; fail rather than silently narrowing its inputs.
        if any(t is not None and t.dtype != torch.float32
               for t in (batch.pan, batch.ms, batch.native_lms, batch.gt)):
            raise ValueError("S2D Stage1 requires finite FP32 observations")

    def predict_shift(self, pan, native_lms, *, resolution, pan_valid_mask=None,
                      return_aligner_details=False):
        if pan.dtype != torch.float32 or native_lms.dtype != torch.float32:
            raise ValueError("S2D Aligner observations must be FP32 before any cast")
        return super().predict_shift(pan, native_lms, resolution=resolution,
                                     pan_valid_mask=pan_valid_mask,
                                     return_aligner_details=return_aligner_details)


class S2DMaskedStage1(MaskedDonorStage1, S2DShiftStage1):
    """Exact existing MASK2 forward with an S2D-specific checkpoint identity."""

    @classmethod
    def from_fresh_stage(cls, stage):
        if type(stage) is not S2DShiftStage1 or stage.forward_calls != 0:
            raise ValueError("Only a fresh, untrained S2DShiftStage1 may receive MASK2")
        return cls(stage.aligner, stage.teacher, sources=stage.sources,
                   epsilon_seed=stage.epsilon_seed, roi_margin=stage.roi_margin,
                   boundary_policy=stage.boundary_policy,
                   lambda_abs=stage.lambda_abs, lambda_eps=stage.lambda_eps)

    def pipeline_manifest(self):
        # Call the S2D identity directly: the old masking class's manifest uses
        # a DirectShift revision, even though its numerical forward is reusable.
        result = S2DShiftStage1.pipeline_manifest(self)
        result.update(revision=MASKED_METHOD_REVISION, mask_policy="MASK2",
                      rec_reduction=LOSS_MASK_POLICY, loss_mask=mask_manifest())
        result["boundary"].update(
            prediction_dependent_mask=True,
            prediction_dependent_mask_scope="reconstruction_only_spatial_ROI8_unchanged")
        return result


def build_s2d_stage1(sources, *, teacher_spec, teacher_seed, aligner_seed, epsilon_seed,
                     mask_policy, aligner_view_margin=8, roi_margin=8,
                     boundary_policy="require_spatial_support", lambda_abs=1., lambda_eps=.001):
    """Construct an independent donor; source registry and all seeds are explicit.

    This function creates no optimizer, run directory, queue, evaluation or
    sheet entry. The caller must select the same mask policy for the baseline.
    """
    _check_teacher_spec(teacher_spec)
    if mask_policy not in MASK_POLICIES:
        raise ValueError("Explicit mask_policy='MASK2' or 'UNMASKED' required")
    for name, value in (("teacher_seed", teacher_seed), ("aligner_seed", aligner_seed),
                        ("epsilon_seed", epsilon_seed)):
        if type(value) is not int:
            raise ValueError(name + " must be an explicit integer")
    if teacher_spec.bands != sources.bands:
        raise ValueError("Sensor and Teacher band identities disagree")
    training_sources = DirectTrainingSources(sources.sensor, sources.manifest_sha("train"))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(aligner_seed)
        aligner = S2DGlobalPANAligner(sources.bands, view_margin=aligner_view_margin)
    teacher = build_reconstructor(teacher_spec, seed=teacher_seed, role="T")
    cls = S2DMaskedStage1 if mask_policy == "MASK2" else S2DShiftStage1
    return cls(aligner, teacher, sources=training_sources, epsilon_seed=epsilon_seed,
               roi_margin=roi_margin, boundary_policy=boundary_policy,
               lambda_abs=lambda_abs, lambda_eps=lambda_eps)
