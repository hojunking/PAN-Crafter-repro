"""DirectShift donor-bound Stage 2; no edits to the live Stage 1 implementation.

The Aligner and Teacher are frozen by ownership transfer. Calibration is an
exact RR-training-population median, never a validation/test estimate. All
structural maps reuse Stage 1's RGB-mean-first absolute Scharr operator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re

import numpy as np
import torch
from torch import nn

from directshift48.method import DirectShiftStage1, ScharrAbsolute
from pa.warp import warp_pan, warp_support_mask
from panda_method.calibration import canonical_sha256, state_dict_sha256
from panda_method.losses import reliability_weight, teacher_error
from panda_method.reconstruction import Reconstructor
from .masking import MaskedDonorStage1, LOSS_MASK_POLICY, masked_mean, validate_mask, mask_manifest


METHOD_VERSION = "DONOR30_DIRECT_SCHARR_STAGE2_MASKED_V2"
CALIBRATION_PROTOCOL = "unaugmented_bicubic_valid_train_pixels_v2"
MODES = ("gt", "e", "eq", "uniform")


def _roi(x):
    if min(x.shape[-2:]) <= 16:
        raise ValueError("Nonempty fixed ROI8 required")
    return x[..., 8:-8, 8:-8]


def discrepancy(spatial, first, second):
    if first.shape != second.shape or first.device != second.device:
        raise ValueError("Scharr references must share shape and device")
    return (spatial.descriptor(first) - spatial.descriptor(second)).abs().mean(1, keepdim=True)


def reference_maps(spatial, teacher, gt, aligned_pan):
    with torch.no_grad():
        e = teacher_error(teacher, gt)
        q = discrepancy(spatial, aligned_pan.detach(), spatial.rgb_luminance(gt.detach()))
    if e.shape != q.shape:
        raise ValueError("Matching e/q maps required")
    return e, q


def loss_terms(spatial, student, teacher, gt, aligned_pan, *, mode, calibration,
               valid_mask, lambda_spatial=1.):
    """Fitting uses detached PAN support; spatial retains guarded fixed ROI8."""
    if mode not in MODES or not math.isfinite(lambda_spatial) or lambda_spatial < 0:
        raise ValueError("Explicit Stage 2 mode and finite nonnegative spatial coefficient required")
    if student.shape != gt.shape or student.device != gt.device:
        raise ValueError("Student and GT must share shape and device")
    valid_mask, _ = validate_mask(student, valid_mask)
    # Spatial descriptors at ROI8 require their complete 3x3 stencil in support.
    if not bool(valid_mask[..., 7:-7, 7:-7].all()):
        raise ValueError("SUPPORT_VIOLATION: fixed Scharr ROI/stencil is not valid")
    gt = gt.detach().float()
    hard_px = (student.float() - gt).abs().mean(1, keepdim=True)
    hard = masked_mean(hard_px, valid_mask)
    zero = student.new_zeros(())
    if mode == "gt":
        return {"loss": hard, "fit": hard, "hard": hard, "soft": zero,
                "spatial": zero, "weighted_spatial": zero}
    if calibration is None or teacher is None or teacher.shape != student.shape:
        raise ValueError("KD requires a calibrated, matching frozen Teacher output")
    e, q = reference_maps(spatial, teacher, gt, aligned_pan)
    w_e = reliability_weight(e, calibration.tau_e)
    w_q = reliability_weight(q, calibration.tau_q)
    soft_px = (student.float() - teacher.detach().float()).abs().mean(1, keepdim=True)
    fit = masked_mean((1 - w_e) * soft_px + w_e * hard_px, valid_mask)
    edge = discrepancy(spatial, spatial.rgb_luminance(student.float()), spatial.rgb_luminance(gt))
    spatial_loss = (_roi(w_q * edge).mean() if mode == "eq" else
                    _roi(edge).mean() if mode == "uniform" else zero)
    weighted = lambda_spatial * spatial_loss
    return {"loss": fit + weighted, "fit": fit, "hard": hard, "soft": masked_mean(soft_px, valid_mask),
            "spatial": spatial_loss, "weighted_spatial": weighted,
            "e": e, "q": q, "w_e": w_e, "w_q": w_q, "edge_px": edge}


@dataclass(frozen=True)
class DirectCalibrationBinding:
    aligner_sha256: str
    teacher_sha256: str
    pipeline_sha256: str
    source_manifest_sha256: str
    spatial_sha256: str

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(name + " must be a lowercase SHA256")


@dataclass(frozen=True)
class DirectCalibrationScales:
    tau_e: float
    tau_q: float
    n_e: int
    n_q: int
    n_samples: int
    binding: DirectCalibrationBinding
    sample_ids_sha256: str
    protocol: str = CALIBRATION_PROTOCOL
    median_convention: str = "midpoint_even_global_pixels"

    def __post_init__(self):
        if self.protocol != CALIBRATION_PROTOCOL or self.median_convention != "midpoint_even_global_pixels":
            raise ValueError("Explicit unaugmented full train exact median required")
        if not isinstance(self.binding, DirectCalibrationBinding):
            raise ValueError("Checkpoint-bound calibration required")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
               for v in (self.tau_e, self.tau_q)):
            raise ValueError("Finite nonnegative calibration scales required")
        if any(type(v) is not int or v <= 0 for v in (self.n_e, self.n_q, self.n_samples)) or self.n_q > self.n_e:
            raise ValueError("Positive exact calibration counts required")
        if re.fullmatch(r"[0-9a-f]{64}", self.sample_ids_sha256) is None:
            raise ValueError("Pinned sample population SHA256 required")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        data = dict(payload)
        data["binding"] = DirectCalibrationBinding(**data["binding"])
        return cls(**data)

    def assert_binding(self, expected):
        if self.binding != expected:
            raise ValueError("Stale calibration: frozen A/T, source or pipeline changed")


def calibration_pipeline(stage):
    return {"donor_pipeline": stage.pipeline_manifest(),
            "protocol": CALIBRATION_PROTOCOL,
            "e": "all_band_mean_absolute_teacher_vs_GT_bicubic_valid_pixels",
            "loss_mask": mask_manifest(),
            "q": "two_axis_mean_absolute_Scharr_alignedPAN_vs_GT_RGB_ROI8",
            "radiometry": "same_FP32_training_input_units_no_clipping",
            "median": "midpoint_even_global_pixels"}


def calibration_binding(stage):
    if not isinstance(stage, MaskedDonorStage1):
        raise TypeError("A fresh masked-loss Stage 1 donor is required")
    return DirectCalibrationBinding(
        state_dict_sha256(stage.aligner), state_dict_sha256(stage.teacher),
        canonical_sha256(calibration_pipeline(stage)), stage.sources.rr_manifest_sha256,
        canonical_sha256(stage.spatial.manifest()))


class ExactDirectCalibrator:
    """Exact FP32 populations on CPU, with bounded peak allocation checks.

    Inputs originate in FP32; retaining FP32 does not approximate their median.
    In-place NumPy partition avoids FP64 promotion and kthvalue-sized copies.
    """
    def __init__(self, binding, spatial, *, expected_sample_ids,
                 max_host_bytes=2 * 1024**3):
        ids = list(expected_sample_ids)
        if isinstance(expected_sample_ids, str) or not ids or any(not isinstance(s, str) or not s for s in ids) or len(set(ids)) != len(ids):
            raise ValueError("Unique nonempty full training sample inventory required")
        if type(max_host_bytes) is not int or max_host_bytes <= 0:
            raise ValueError("Explicit positive host calibration memory budget required")
        if canonical_sha256(spatial.manifest()) != binding.spatial_sha256:
            raise ValueError("Calibration Scharr definition mismatch")
        self.binding, self.spatial = binding, spatial
        self.expected, self.seen = frozenset(ids), set()
        self.e, self.q, self.bytes = [], [], 0
        self.max_host_bytes, self.closed = max_host_bytes, False

    def update(self, teacher, gt, aligned_pan, *, split, sample_ids, valid_mask):
        if self.closed:
            raise RuntimeError("Calibration already finalized")
        ids = list(sample_ids)
        if split != "train":
            raise ValueError("Calibration accepts train samples only")
        if isinstance(sample_ids, str) or len(ids) != len(teacher) or len(set(ids)) != len(ids) or self.seen.intersection(ids):
            raise ValueError("Calibration batch sample IDs are missing or duplicated")
        if not set(ids).issubset(self.expected):
            raise ValueError("Unexpected training sample ID")
        if teacher.dtype != torch.float32 or gt.dtype != torch.float32 or aligned_pan.dtype != torch.float32:
            raise ValueError("Calibration requires the actual FP32 training radiometry")
        e, q = reference_maps(self.spatial, teacher, gt, aligned_pan)
        valid_mask, counts = validate_mask(e, valid_mask)
        if not bool(valid_mask[..., 7:-7, 7:-7].all()):
            raise ValueError("SUPPORT_VIOLATION: calibration ROI/stencil is not valid")
        projected = self.bytes + 4 * (int(counts.sum()) + _roi(q).numel())
        # At finalization only one concatenated population coexists with chunks.
        # Two times all retained bytes is a conservative bound, plus small maps.
        if 2 * projected > self.max_host_bytes:
            raise MemoryError("Exact calibration exceeds declared host memory budget; no approximation allowed")
        self.e.append(e.detach()[valid_mask].cpu().contiguous().numpy().copy())
        self.q.append(_roi(q).detach().cpu().contiguous().reshape(-1).numpy().copy())
        self.bytes = projected
        self.seen.update(ids)

    @staticmethod
    def _median(chunks):
        values = np.concatenate(chunks)
        chunks.clear()
        n = len(values)
        lower, upper = (n - 1) // 2, n // 2
        values.partition((lower, upper))
        return float(values[lower]) / 2 + float(values[upper]) / 2, n

    def finalize(self):
        if self.closed:
            raise RuntimeError("Calibration already finalized")
        if self.seen != self.expected:
            raise ValueError("Incomplete full train calibration population")
        tau_e, n_e = self._median(self.e)
        tau_q, n_q = self._median(self.q)
        self.closed = True
        return DirectCalibrationScales(tau_e, tau_q, n_e, n_q, len(self.seen),
            self.binding, canonical_sha256(sorted(self.seen)))


@torch.no_grad()
def calibrate_training(stage1, batches, *, expected_sample_ids,
                       calibration_protocol=CALIBRATION_PROTOCOL, max_host_bytes=2 * 1024**3):
    if calibration_protocol != CALIBRATION_PROTOCOL:
        raise ValueError("Only the registered unaugmented full train calibration is supported")
    binding = calibration_binding(stage1)
    collector = ExactDirectCalibrator(binding, stage1.spatial,
        expected_sample_ids=expected_sample_ids, max_host_bytes=max_host_bytes)
    modes = [(m, m.training) for m in stage1.modules()]
    try:
        stage1.eval()
        for batch in batches:
            batch.validate(stage1.teacher.spec.bands, training=True, require_gt=True)
            stage1.sources.check(batch)
            if any(tuple(a) != (False, False, 0) for a in batch.lineage.augmentations):
                raise ValueError("Calibration requires unaugmented registered training samples")
            out = stage1.native_prediction(batch)
            stage1._native_support(batch.pan, out["u"])
            collector.update(out["y"], batch.gt, out["pan_aligned"], split=batch.lineage.split,
                             sample_ids=batch.lineage.sample_ids,
                             valid_mask=warp_support_mask(*batch.pan.shape[-2:], out["u"].detach()))
        if calibration_binding(stage1) != binding:
            raise ValueError("Donor changed during calibration")
        return collector.finalize()
    finally:
        for module, mode in modes:
            module.training = mode


class DirectStage2(nn.Module):
    """Prior-free shared frozen A/T and a fresh trainable P0 Student."""
    def __init__(self, stage1, student, *, mode="eq", calibration=None, lambda_spatial=1.):
        super().__init__()
        if not isinstance(stage1, MaskedDonorStage1) or not isinstance(student, Reconstructor):
            raise TypeError("Masked-loss donor and existing Reconstructor required")
        if mode not in MODES or type(lambda_spatial) not in (int, float) or not math.isfinite(lambda_spatial) or lambda_spatial < 0:
            raise ValueError("Valid Stage 2 mode and spatial coefficient required")
        if student.spec.bands != stage1.teacher.spec.bands or student.spec.input_layout != "P0" or student.spec.base_reference != "BICUBIC_MS":
            raise ValueError("Student must preserve P0/BICUBIC_MS and donor sensor bands")
        groups = [{id(p) for p in m.parameters()} for m in (stage1.aligner, stage1.teacher, student)]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError("Aligner, Teacher and Student parameters must not alias")
        if not all(p.requires_grad for p in student.parameters()):
            raise ValueError("All Student parameters must be trainable")
        binding = calibration_binding(stage1)
        if mode != "gt" and calibration is None:
            raise ValueError("KD modes require complete train-only calibration")
        if calibration is not None:
            calibration.assert_binding(binding)
        self.aligner, self.teacher, self.student = stage1.aligner, stage1.teacher, student
        self.spatial, self.sources = stage1.spatial, stage1.sources
        self.roi_margin, self.boundary_policy = stage1.roi_margin, stage1.boundary_policy
        self.mode, self.calibration, self.lambda_spatial = mode, calibration, float(lambda_spatial)
        self._donor_pipeline = stage1.pipeline_manifest()
        self._alignment_manifest = stage1.alignment_manifest()
        self._donor_binding = binding
        self._freeze()
        self._contract_sha = canonical_sha256(self.get_extra_state())
        self._frozen_versions = self._versions()

    def _freeze(self):
        for module in (self.aligner, self.teacher):
            module.requires_grad_(False).eval()
            for p in module.parameters():
                p.grad = None

    def train(self, mode=True):
        super().train(mode)
        self.aligner.eval()
        self.teacher.eval()
        return self

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        if hasattr(self, "_donor_binding"):
            # Device relocation is allowed; changing frozen weights/dtypes is
            # not. Tensor object identities can change on CPU/CUDA transfer.
            self.assert_frozen_binding()
            self._frozen_versions = self._versions()
        return result

    def trainable_parameters(self):
        return self.student.parameters()

    def assert_optimizer(self, optimizer):
        actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
        if len(actual) != len(set(actual)) or set(actual) != {id(p) for p in self.student.parameters()}:
            raise ValueError("Only and all Student parameters may be optimized")

    def alignment_manifest(self):
        return self._alignment_manifest

    def pipeline_manifest(self):
        return {"method": METHOD_VERSION, "mode": self.mode, "donor": self._donor_pipeline,
                "student": self.student.spec.manifest(), "lambda_spatial": self.lambda_spatial,
                "frozen": ["aligner", "teacher"], "inference": "Aligner+Student_only",
                "fit_reduction": LOSS_MASK_POLICY, "loss_mask": mask_manifest(),
                "spatial_reduction": "ROI8_pixel_mean_not_weight_sum",
                "spatial": self.spatial.manifest()}

    def get_extra_state(self):
        return {"pipeline": self.pipeline_manifest(), "sources": asdict(self.sources),
                "donor_binding": asdict(self._donor_binding),
                "calibration": None if self.calibration is None else self.calibration.to_dict()}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Stage 2 donor/calibration/mode checkpoint contract mismatch")

    def assert_frozen_binding(self):
        if (state_dict_sha256(self.aligner) != self._donor_binding.aligner_sha256 or
                state_dict_sha256(self.teacher) != self._donor_binding.teacher_sha256):
            raise ValueError("Frozen donor A/T weights changed")

    def _versions(self):
        return tuple((id(v), v._version) for module in (self.aligner, self.teacher)
                     for v in list(module.parameters()) + list(module.buffers()))

    def _assert_contract(self):
        if canonical_sha256(self.get_extra_state()) != self._contract_sha:
            raise ValueError("Stage 2 source, pipeline or calibration contract changed")
        if self._versions() != self._frozen_versions:
            raise ValueError("Frozen donor tensors changed after Stage 2 construction/loading")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not strict or state_dict.get("_extra_state") != self.get_extra_state():
            raise ValueError("Stage 2 requires strict same-donor calibrated loading")
        result = super().load_state_dict(state_dict, strict=True, assign=assign)
        self.assert_frozen_binding()
        self._freeze()
        self._frozen_versions = self._versions()
        return result

    def _check_observation(self, batch):
        DirectShiftStage1._check_observation(self, batch)

    def _native_support(self, pan, correction):
        return DirectShiftStage1._native_support(self, pan, correction)

    def predict_shift(self, pan, native_lms, **kwargs):
        with torch.no_grad():
            return DirectShiftStage1.predict_shift(self, pan, native_lms, **kwargs)

    def native_prediction(self, batch, *, return_aligner_details=False):
        self._assert_contract()
        self._check_observation(batch)
        # Official RR/FR inference never calls Teacher, GT, e or q.
        with torch.no_grad(), torch.autocast(device_type=batch.pan.device.type, enabled=False):
            shift = self.predict_shift(batch.pan, batch.native_lms, resolution=batch.lineage.resolution,
                                       return_aligner_details=return_aligner_details)
            pan = warp_pan(batch.pan.float(), shift["u"])
            out = self.student(pan, batch.ms, batch.native_lms)
        return {**out, **shift}

    def forward(self, rr):
        self._assert_contract()
        rr.validate(self.student.spec.bands, training=True, require_gt=True)
        self.sources.check(rr)
        if any(p.requires_grad for m in (self.aligner, self.teacher) for p in m.parameters()):
            raise ValueError("Stage 2 Aligner and Teacher must remain frozen")
        self.aligner.eval()
        self.teacher.eval()
        with torch.no_grad(), torch.autocast(device_type=rr.pan.device.type, enabled=False):
            u = self.aligner(rr.pan.float(), rr.native_lms.float())
            pan = warp_pan(rr.pan.float(), u)
            support = self._native_support(rr.pan, u)
            valid_mask = warp_support_mask(*rr.pan.shape[-2:], u.detach())
            teacher = None if self.mode == "gt" else self.teacher(pan, rr.ms, rr.native_lms)
        with torch.autocast(device_type=rr.pan.device.type, enabled=False):
            student = self.student(pan, rr.ms, rr.native_lms)
            terms = loss_terms(self.spatial, student["y"], None if teacher is None else teacher["y"],
                               rr.gt, pan, mode=self.mode, calibration=self.calibration,
                               valid_mask=valid_mask, lambda_spatial=self.lambda_spatial)
        if not bool(torch.isfinite(terms["loss"])):
            raise FloatingPointError("Nonfinite Stage 2 objective")
        return {**terms, "teacher": teacher, "student": student,
                "native": {**student, "u": u, "correction": u}, "correction": u,
                "pan_aligned": pan, "loss_mask": valid_mask, **support}

    def inference_model(self):
        return StudentInference(self.aligner, self.student, self.sources.sensor)


class StudentInference(nn.Module):
    """Deployment object contains only A+S, no Teacher or calibration maps."""
    def __init__(self, aligner, student, sensor):
        super().__init__()
        self.aligner, self.student, self.sensor = aligner, student, sensor
        self.aligner.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.aligner.eval()
        return self

    @torch.no_grad()
    def forward(self, batch):
        batch.validate(self.student.spec.bands)
        if batch.gt is not None:
            raise ValueError("Deployment accepts observations only, no GT")
        if batch.lineage.sensor != self.sensor or batch.lineage.reference != "PROVIDED_STORED_LMS":
            raise ValueError("Pinned sensor and provided LMS required")
        with torch.autocast(device_type=batch.pan.device.type, enabled=False):
            u = self.aligner(batch.pan.float(), batch.native_lms.float())
            p = warp_pan(batch.pan.float(), u)
            out = self.student(p, batch.ms, batch.native_lms)
        return {**out, "u": u, "correction": u}
