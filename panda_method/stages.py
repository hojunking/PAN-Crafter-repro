"""Method-only Stage 1/2 objectives, not experiment runners.

Callers own optimizer/AMP policy, batching, checkpoint selection, evaluation and
logging. There are no dataset paths, queue IDs, automatic downloads or launches.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from pa.offset import sample_offsets
from pa.warp import warp_pan, warp_support_mask
from . import METHOD_VERSION
from .calibration import CalibrationBinding, CalibrationScales, ExactTrainCalibrator, canonical_sha256
from .data import ObservationBatch, require_sha256
from .geometry import SceneAwarePANAligner
from .losses import AbsoluteGradientLoss
from .reconstruction import Reconstructor, aligned_observations


@dataclass(frozen=True)
class TrainingSources:
    rr_manifest_sha256: str
    fr_manifest_sha256: str
    prior_mapping_sha256: str

    def __post_init__(self):
        for name, value in asdict(self).items():
            require_sha256(value, name)

    def check(self, batch: ObservationBatch, resolution: str):
        batch.lineage.require_training()
        expected = self.rr_manifest_sha256 if resolution == "RR" else self.fr_manifest_sha256
        if (batch.lineage.resolution != resolution
                or batch.lineage.source_manifest_sha256 != expected
                or batch.lineage.prior_mapping_sha256 != self.prior_mapping_sha256):
            raise ValueError("Batch does not belong to the pinned training source/prior mapping")


def _disjoint(*modules):
    seen = set()
    for module in modules:
        params = {id(p) for p in module.parameters()}
        if seen & params:
            raise ValueError("Aligner/Teacher/Student must not alias learnable parameters")
        seen |= params


def _pipeline(aligner, teacher, boundary_policy, calibration_protocol):
    return {"method": METHOD_VERSION, "aligner_reference": "NATIVE_LMS",
            "aligner_view_margin": aligner.view_margin,
            "aligner_conditioning": aligner.conditioning_manifest(),
            "teacher": teacher.spec.manifest(), "warp": "bicubic_border_align_corners_false_fp32",
            "boundary_policy": boundary_policy, "calibration_protocol": calibration_protocol,
            "rec_reduction": "all_pixels_and_bands_mean",
            "eps": {"radius": 2., "lambda": 1e-3, "stop_reference": True,
                    "cadence": "every_forward_RR_and_FR", "generator": "dedicated_CPU"}}


def _validate_boundary(policy):
    if policy not in ("report", "require_spatial_support"):
        raise ValueError("Explicit boundary_policy='report' or 'require_spatial_support' required")


def _support(pan, correction, spatial, policy):
    support = warp_support_mask(*pan.shape[-2:], correction.detach())
    # Each spatial derivative uses an additional one-pixel stencil. This check
    # diagnoses its input support; it does not create a moving loss mask.
    margin = max(spatial.spec.roi_margin - 1, 0)
    h, w = pan.shape[-2:]
    valid = support[..., margin:h-margin, margin:w-margin].flatten(1).all(1)
    if policy == "require_spatial_support" and not bool(valid.all()):
        raise ValueError("Fixed spatial ROI/stencil exceeds valid native warp support")
    return {"warp_support_fraction": support.float().mean(), "spatial_support_ok": valid}


def epsilon_loss(c_eps, c_native, epsilon):
    if c_eps.shape != c_native.shape or c_eps.shape != epsilon.shape or c_eps.shape[-1] != 2:
        raise ValueError("One (dy,dx) correction per sample required")
    return (c_eps + epsilon.detach() - c_native.detach()).abs().mean()


class Stage1(nn.Module):
    """Shared A: RR native reconstruction/absolute loss + RR/FR epsilon loss.

    ``forward`` computes one complete objective; the outer trainer makes one
    optimizer update. The RNG state counts forwards (not optimizer successes).
    Restore the whole checkpoint, including ``_extra_state``, when resuming.
    """

    def __init__(self, aligner: SceneAwarePANAligner, teacher: Reconstructor,
                 spatial: AbsoluteGradientLoss, *, sources: TrainingSources,
                 epsilon_seed: int, boundary_policy: str, calibration_protocol: str):
        super().__init__()
        _validate_boundary(boundary_policy)
        _disjoint(aligner, teacher)
        if aligner.ms_bands != teacher.spec.bands:
            raise ValueError("Aligner/Teacher band counts differ")
        if not isinstance(calibration_protocol, str) or not calibration_protocol.strip():
            raise ValueError("Explicit calibration population/augmentation protocol required")
        self.aligner, self.teacher, self.spatial = aligner, teacher, spatial
        self.sources, self.boundary_policy = sources, boundary_policy
        self.calibration_protocol = calibration_protocol
        self.epsilon_seed = int(epsilon_seed)
        self._epsilon_rng = torch.Generator(device="cpu").manual_seed(self.epsilon_seed)
        self.forward_calls = 0

    def pipeline_manifest(self):
        return _pipeline(self.aligner, self.teacher, self.boundary_policy, self.calibration_protocol)

    def get_extra_state(self):
        return {"method": METHOD_VERSION, "pipeline": self.pipeline_manifest(),
                "spatial": asdict(self.spatial.spec), "sources": asdict(self.sources),
                "epsilon_seed": self.epsilon_seed, "forward_calls": self.forward_calls,
                "epsilon_rng": self._epsilon_rng.get_state().clone()}

    def set_extra_state(self, state):
        if (state["method"] != METHOD_VERSION or state["pipeline"] != self.pipeline_manifest()
                or state["spatial"] != asdict(self.spatial.spec)
                or state["sources"] != asdict(self.sources)):
            raise ValueError("Checkpoint method/preprocessing/source contract mismatch")
        self.epsilon_seed = int(state["epsilon_seed"])
        self.forward_calls = int(state["forward_calls"])
        self._epsilon_rng.set_state(state["epsilon_rng"].cpu())

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not strict or "_extra_state" not in state_dict:
            raise ValueError("Stage 1 requires a strict method checkpoint with RNG/provenance")
        saved = state_dict["_extra_state"]
        expected = self.get_extra_state()
        for key in ("method", "pipeline", "spatial", "sources"):
            if saved.get(key) != expected[key]:
                raise ValueError(f"Stage 1 checkpoint {key} mismatch")
        return super().load_state_dict(state_dict, strict=True, assign=assign)

    def native_prediction(self, batch: ObservationBatch, *, return_aligner_details=False):
        batch.validate(self.teacher.spec.bands)
        with torch.autocast(device_type=batch.pan.device.type, enabled=False):
            if return_aligner_details:
                c, details = self.aligner(batch.pan.float(), batch.native_lms.float(), batch.prior_dy_dx,
                                          return_details=True)
            else:
                c = self.aligner(batch.pan.float(), batch.native_lms.float(), batch.prior_dy_dx)
            p, low = aligned_observations(batch.pan, c, lpan=batch.lpan,
                                         need_lpan=self.teacher.spec.uses_lpan)
            out = self.teacher(p, batch.ms, batch.native_lms, aligned_lpan=low)
        return {**out, "correction": c,
                **({"aligner_details": details} if return_aligner_details else {})}

    def _epsilon(self, batch, c0, epsilon):
        with torch.no_grad():
            shifted = warp_pan(batch.pan.float(), epsilon)
        c_eps = self.aligner(shifted, batch.native_lms.float(), batch.prior_dy_dx)
        return epsilon_loss(c_eps, c0, epsilon), c_eps

    def forward(self, rr: ObservationBatch, fr: ObservationBatch):
        if not all(p.requires_grad for m in (self.aligner, self.teacher) for p in m.parameters()):
            raise ValueError("Stage 1 requires trainable A/T; do not reuse after transfer to Stage 2")
        rr.validate(self.teacher.spec.bands, training=True, require_gt=True)
        fr.validate(self.teacher.spec.bands, training=True)
        self.sources.check(rr, "RR")
        self.sources.check(fr, "FR")
        rng_before = self._epsilon_rng.get_state()
        try:
            with torch.autocast(device_type=rr.pan.device.type, enabled=False):
                native = self.native_prediction(rr)
                rec = (native["y"] - rr.gt.detach().float()).abs().mean()
                absolute = self.spatial.stage1(native["pan_aligned"], rr.gt)
                support = _support(rr.pan, native["correction"], self.spatial, self.boundary_policy)
                eps_r = sample_offsets(len(rr.pan), 2., self._epsilon_rng).to(rr.pan.device)
                eps_f = sample_offsets(len(fr.pan), 2., self._epsilon_rng).to(fr.pan.device)
                eps_rr, c_eps_r = self._epsilon(rr, native["correction"], eps_r)
                with torch.no_grad():
                    c0_f = self.aligner(fr.pan.float(), fr.native_lms.float(), fr.prior_dy_dx)
                eps_fr, c_eps_f = self._epsilon(fr, c0_f, eps_f)
                con_rr, con_fr = absolute + 1e-3 * eps_rr, 1e-3 * eps_fr
                loss = rec + con_rr + con_fr
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite Stage 1 objective")
                epsilon_support = {}
                for name, observation, eps in (("rr", rr, eps_r), ("fr", fr, eps_f)):
                    mask = warp_support_mask(*observation.pan.shape[-2:], eps)
                    m = self.aligner.view_margin
                    h, w = observation.pan.shape[-2:]
                    view = mask[..., m:h-m, m:w-m]
                    epsilon_support[f"epsilon_{name}_view_support_fraction"] = view.float().mean()
        except Exception:
            self._epsilon_rng.set_state(rng_before)
            raise
        call = self.forward_calls
        self.forward_calls += 1
        return {"loss": loss, "rec": rec, "abs_rr": absolute,
                "eps_rr": eps_rr, "eps_fr": eps_fr, "con_rr": con_rr, "con_fr": con_fr,
                "weighted_eps_rr": 1e-3 * eps_rr, "weighted_eps_fr": 1e-3 * eps_fr,
                "native": native, "epsilon_rr": eps_r, "epsilon_fr": eps_f,
                "c_eps_rr": c_eps_r, "c0_fr": c0_f, "c_eps_fr": c_eps_f,
                "forward_index": call, **support, **epsilon_support}


def calibration_binding(stage1: Stage1):
    return CalibrationBinding.for_modules(
        stage1.aligner, stage1.teacher, pipeline=stage1.pipeline_manifest(),
        source_manifest_sha256=stage1.sources.rr_manifest_sha256,
        prior_mapping_sha256=stage1.sources.prior_mapping_sha256, spatial=stage1.spatial)


@torch.no_grad()
def calibrate_training(stage1: Stage1, batches, *, expected_sample_ids):
    """Explicit opt-in full training-population pass; never called on import.

    Caller must supply every registered training sample exactly once under the
    declared calibration augmentation. The collector rejects repeated IDs.
    The caller supplies the full training sample inventory from its manifest;
    an incomplete pass is rejected, not published as median_train.
    """
    modes = [(m, m.training) for m in stage1.modules()]
    binding = calibration_binding(stage1)
    collector = ExactTrainCalibrator(binding, stage1.spatial,
                                     expected_sample_ids=expected_sample_ids)
    try:
        stage1.eval()
        for batch in batches:
            batch.validate(stage1.teacher.spec.bands, training=True, require_gt=True)
            stage1.sources.check(batch, "RR")
            out = stage1.native_prediction(batch)
            _support(batch.pan, out["correction"], stage1.spatial, stage1.boundary_policy)
            collector.update(out["y"], batch.gt, out["pan_aligned"], split=batch.lineage.split,
                             sample_ids=batch.lineage.sample_ids)
        if calibration_binding(stage1) != binding:
            raise ValueError("Aligner/Teacher/pipeline changed during calibration")
        return collector.finalize()
    finally:
        for module, mode in modes:
            module.training = mode


class Stage2(nn.Module):
    """Exactly one frozen A and one frozen Teacher; only Student is optimized."""

    def __init__(self, stage1: Stage1, student: Reconstructor, calibration: CalibrationScales):
        super().__init__()
        _disjoint(stage1.aligner, stage1.teacher, student)
        if student.spec.bands != stage1.teacher.spec.bands:
            raise ValueError("Teacher/Student spectral band counts differ")
        calibration.assert_binding(calibration_binding(stage1))
        # Ownership transfer by reference, not deepcopy/Student aligner clone.
        self.aligner = stage1.aligner
        self.teacher = stage1.teacher
        self.student = student
        self.spatial = stage1.spatial
        self.sources, self.boundary_policy = stage1.sources, stage1.boundary_policy
        self.calibration = calibration
        self.calibration_protocol = stage1.calibration_protocol
        self.aligner.requires_grad_(False).eval()
        self.teacher.requires_grad_(False).eval()
        for module in (self.aligner, self.teacher):
            for parameter in module.parameters():
                parameter.grad = None
        if not all(p.requires_grad for p in student.parameters()):
            raise ValueError("Student reconstruction parameters must be trainable")

    def train(self, mode=True):
        super().train(mode)
        self.aligner.eval()
        self.teacher.eval()
        return self

    def trainable_parameters(self):
        return self.student.parameters()

    def assert_optimizer(self, optimizer):
        actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
        expected = {id(p) for p in self.student.parameters()}
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("Stage 2 optimizer must contain only, and all, Student parameters")

    def _binding(self):
        return CalibrationBinding.for_modules(
            self.aligner, self.teacher, pipeline=self.pipeline_manifest(),
            source_manifest_sha256=self.sources.rr_manifest_sha256,
            prior_mapping_sha256=self.sources.prior_mapping_sha256, spatial=self.spatial)

    def pipeline_manifest(self):
        return _pipeline(self.aligner, self.teacher, self.boundary_policy, self.calibration_protocol)

    def get_extra_state(self):
        return {"method": METHOD_VERSION, "calibration": self.calibration.to_dict(),
                "pipeline": self.pipeline_manifest(),
                "student": self.student.spec.manifest(), "spatial": asdict(self.spatial.spec),
                "sources": asdict(self.sources)}

    def set_extra_state(self, state):
        expected = self.get_extra_state()
        # Loading different A/T requires recalibration and a new Stage2 object.
        if state != expected:
            raise ValueError("Stage 2 checkpoint/calibration contract mismatch")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not strict or state_dict.get("_extra_state") != self.get_extra_state():
            raise ValueError("Method checkpoints require strict loading")
        result = super().load_state_dict(state_dict, strict=True, assign=assign)
        self.calibration.assert_binding(self._binding())
        self.aligner.requires_grad_(False).eval()
        self.teacher.requires_grad_(False).eval()
        return result

    def forward(self, rr: ObservationBatch):
        rr.validate(self.student.spec.bands, training=True, require_gt=True)
        self.sources.check(rr, "RR")
        if self.spatial.spec.fingerprint() != self.calibration.binding.spatial_spec_sha256:
            raise ValueError("Calibration spatial definition changed")
        if (canonical_sha256(self.pipeline_manifest()) != self.calibration.binding.pipeline_sha256
                or self.sources.rr_manifest_sha256 != self.calibration.binding.source_manifest_sha256
                or self.sources.prior_mapping_sha256 != self.calibration.binding.prior_mapping_sha256):
            raise ValueError("Calibrated preprocessing/source configuration changed")
        if any(p.requires_grad for m in (self.aligner, self.teacher) for p in m.parameters()):
            raise ValueError("Stage 2 Aligner and Teacher must remain frozen")
        self.aligner.eval()
        self.teacher.eval()
        with torch.no_grad(), torch.autocast(device_type=rr.pan.device.type, enabled=False):
            c = self.aligner(rr.pan.float(), rr.native_lms.float(), rr.prior_dy_dx)
            p, low = aligned_observations(rr.pan, c, lpan=rr.lpan,
                need_lpan=self.teacher.spec.uses_lpan or self.student.spec.uses_lpan)
            teacher = self.teacher(p, rr.ms, rr.native_lms, aligned_lpan=low)
            support = _support(rr.pan, c, self.spatial, self.boundary_policy)
        student = self.student(p, rr.ms, rr.native_lms, aligned_lpan=low)
        terms = self.spatial.stage2(student["y"], teacher["y"], rr.gt, p,
                                    self.calibration.tau_e, self.calibration.tau_q)
        return {**terms, "teacher": teacher, "student": student,
                "correction": c, "pan_aligned": p, **support}

    def inference_model(self):
        return StudentInference(self.aligner, self.student)


class StudentInference(nn.Module):
    """Deployment graph: frozen Aligner + Student, with no Teacher/GT/e/q."""

    def __init__(self, aligner: SceneAwarePANAligner, student: Reconstructor):
        super().__init__()
        self.aligner, self.student = aligner, student
        self.aligner.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.aligner.eval()
        return self

    def forward(self, batch: ObservationBatch):
        batch.validate(self.student.spec.bands)
        if batch.gt is not None:
            raise ValueError("Deployment accepts observations only, no GT")
        with torch.no_grad(), torch.autocast(device_type=batch.pan.device.type, enabled=False):
            c = self.aligner(batch.pan.float(), batch.native_lms.float(), batch.prior_dy_dx)
            p, low = aligned_observations(batch.pan, c, lpan=batch.lpan,
                                         need_lpan=self.student.spec.uses_lpan)
            out = self.student(p, batch.ms, batch.native_lms, aligned_lpan=low)
        return {**out, "correction": c}
