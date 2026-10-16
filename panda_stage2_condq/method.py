"""Frozen Stage 1 geometry plus fresh, same-capacity Student.

Only this new namespace changes the Stage 2 loss contract. Existing Stage 1,
masked donor30, official RR/FR evaluators, and live source pins are untouched.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import json

import torch
from torch import nn

from directshift48.method import DirectPANAligner, ScharrAbsolute
from pa.warp import warp_pan, warp_support_mask
from panda_method.calibration import canonical_sha256, state_dict_sha256
from panda_method.reconstruction import Reconstructor, build_reconstructor
from .losses import compute_loss
from .spec import Stage2Spec, MASK_POLICY, MASK2_POLICY


def native_support(pan, correction):
    """Common geometry admission guard for both full-image and MASK2 variants."""
    if min(pan.shape[-2:]) <= 16:
        raise ValueError("Nonempty fixed ROI8 required")
    support = warp_support_mask(*pan.shape[-2:], correction.detach())
    good = support[..., 7:-7, 7:-7].flatten(1).all(1)
    if not bool(good.all()):
        evidence = {"correction_dy_dx": correction.detach().cpu().tolist(),
                    "support_fraction_per_sample": support.float().flatten(1).mean(1).cpu().tolist(),
                    "spatial_support_ok": good.cpu().tolist()}
        raise ValueError("SUPPORT_VIOLATION: ROI8 Scharr stencil reads extrapolated PAN; "
                         "no adaptive mask or optimizer update permitted; " + json.dumps(evidence))
    return {"warp_support_fraction": support.float().mean(), "spatial_support_ok": good}


@torch.no_grad()
def frozen_donor_predictions(aligner, teacher, batch, *, sensor):
    """Shared calibration/training frozen forward, with exactly one native warp.

    Callers must additionally validate train-source lineage against the donor.
    This helper also enforces the structural support guard for every variant.
    """
    bands = 8 if sensor == "WV3" else 4
    batch.validate(bands, training=True, require_gt=True)
    if batch.lineage.sensor != sensor:
        raise ValueError("Donor/batch sensor mismatch")
    for tensor in (batch.pan, batch.ms, batch.native_lms, batch.gt):
        if tensor.dtype != torch.float32:
            raise ValueError("Stage 2 uses original FP32 training radiometry")
    for module in (aligner, teacher):
        if module is not None:
            if any(p.requires_grad for p in module.parameters()):
                raise ValueError("Calibration/training requires frozen donor modules")
            module.eval()
    with torch.autocast(device_type=batch.pan.device.type, enabled=False):
        correction = aligner(batch.pan, batch.native_lms)
        support = native_support(batch.pan, correction)
        pan_aligned = warp_pan(batch.pan, correction)
        teacher_out = None if teacher is None else teacher(pan_aligned, batch.ms, batch.native_lms)
    return {"teacher": teacher_out, "teacher_y": None if teacher_out is None else teacher_out["y"],
            "pan_aligned": pan_aligned, "correction": correction, "u": correction, **support}


class ConditionalQStage2(nn.Module):
    """New GT/panda_e/condq stages, not a continuation of legacy E/EQ."""
    def __init__(self, spec, aligner, teacher=None, *, donor_binding, stats=None):
        super().__init__()
        if not isinstance(spec, Stage2Spec) or not isinstance(aligner, DirectPANAligner):
            raise TypeError("Explicit Stage2Spec and original DirectPANAligner required")
        if aligner.ms_bands != spec.bands:
            raise ValueError("Aligner sensor band mismatch")
        if spec.variant == "gt_only":
            if teacher is not None or stats is not None:
                raise ValueError("gt_only must not construct Teacher or calibration")
        elif (not isinstance(teacher, Reconstructor)
              or teacher.spec != spec.reconstruction_spec or stats is None):
            raise ValueError("Same W112-D123/P0/BICUBIC_MS Teacher and new calibration required")
        if not isinstance(donor_binding, dict):
            raise TypeError("Pinned donor metadata dictionary required")
        from .calibration import validate_binding
        donor_binding = validate_binding(donor_binding)
        for key in ("aligner_state_sha256", "teacher_state_sha256", "source_manifest_sha256"):
            value = donor_binding.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Missing pinned donor " + key)
        if donor_binding.get("sensor") != spec.sensor or not donor_binding.get("donor_checkpoint_id"):
            raise ValueError("Donor sensor and checkpoint ID required")
        if stats is not None:
            from .calibration import CalibrationStats
            if not isinstance(stats, CalibrationStats):
                raise TypeError("Only the new full-image conditional-q calibration schema is accepted")
            stats.validate_binding(donor_binding)
            if stats.mask_policy != spec.mask_policy or stats.schema != spec.calibration_schema:
                raise ValueError("Calibration schema/loss mask policy mismatch; recalibrate explicitly")
            stats.assert_integrity(full_hash=True)
            if spec.variant == "panda_condq" and stats.metadata["calibration_kind"] != "exact_condq":
                raise ValueError("panda_condq requires exact full train conditional-q tables")
        self.spec = spec
        self.aligner, self.teacher = aligner, teacher
        self.student = build_reconstructor(spec.reconstruction_spec, seed=spec.seed, role="S")
        self.spatial = ScharrAbsolute(spec.sensor)
        self.stats = stats
        self._donor_binding = copy.deepcopy(donor_binding)
        self.student_initial_state_sha256 = state_dict_sha256(self.student)
        if self.student_initial_state_sha256 == donor_binding["teacher_state_sha256"]:
            raise ValueError("Student must be fresh, not copied from the Teacher")
        parameter_groups = [{id(p) for p in m.parameters()}
                            for m in (self.aligner, self.teacher, self.student) if m is not None]
        if any(parameter_groups[i] & parameter_groups[j]
               for i in range(len(parameter_groups)) for j in range(i)):
            raise ValueError("Aligner, Teacher and Student cannot share parameters")
        self._freeze()
        self.assert_frozen_binding()
        self._frozen_versions = self._versions()
        self._contract_sha256 = canonical_sha256(self.get_extra_state())

    @property
    def donor_binding(self):
        return copy.deepcopy(self._donor_binding)

    def _freeze(self):
        for module in (self.aligner, self.teacher):
            if module is not None:
                module.requires_grad_(False).eval()
                for parameter in module.parameters():
                    parameter.grad = None

    def train(self, mode=True):
        super().train(mode)
        self._freeze()
        return self

    def _versions(self):
        return tuple((id(t), t._version) for m in (self.aligner, self.teacher) if m is not None
                     for t in list(m.parameters()) + list(m.buffers()))

    def assert_frozen_binding(self):
        if state_dict_sha256(self.aligner) != self._donor_binding["aligner_state_sha256"]:
            raise ValueError("Frozen donor Aligner state changed")
        if self.teacher is not None and state_dict_sha256(self.teacher) != self._donor_binding["teacher_state_sha256"]:
            raise ValueError("Frozen donor Teacher state changed")

    def _assert_contract(self):
        if canonical_sha256(self.get_extra_state()) != self._contract_sha256:
            raise ValueError("Stage 2 donor/data/calibration/loss contract changed")
        if self._versions() != self._frozen_versions:
            raise ValueError("Frozen donor parameter or buffer version changed")
        if any(p.requires_grad for m in (self.aligner, self.teacher) if m is not None for p in m.parameters()):
            raise ValueError("Frozen donor must not become trainable")
        if not all(p.requires_grad for p in self.student.parameters()):
            raise ValueError("All Student reconstruction parameters must be trainable")
        if self.stats is not None:
            self.stats.validate_binding(self._donor_binding)
            self.stats.assert_integrity()

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        if hasattr(self, "_donor_binding"):
            self.assert_frozen_binding()
            self._frozen_versions = self._versions()
        return result

    def move_calibration(self, device, *, memory_budget_bytes):
        """Explicit table placement, distinct from model .to() and never a LUT fallback."""
        if self.stats is None:
            raise ValueError("gt_only has no calibration to move")
        self.stats = self.stats.to(device, memory_budget_bytes=memory_budget_bytes)
        return self

    def trainable_parameters(self):
        return self.student.parameters()

    def assert_optimizer(self, optimizer):
        actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
        if len(actual) != len(set(actual)) or set(actual) != {id(p) for p in self.student.parameters()}:
            raise ValueError("Optimizer must contain exactly all Student parameters, no frozen A/T")

    def pipeline_manifest(self):
        return {**self.spec.manifest(), "aligner": self.aligner.manifest(),
                "teacher": self.spec.reconstruction_spec.manifest(),
                "teacher_loaded": self.teacher is not None,
                "warp": "bicubic_border_FP32_align_corners_false_current_pan_px",
                "native_lms": "PROVIDED_STORED_LMS_not_bicubic_MS",
                "mask_policy": self.spec.mask_policy, "q_use": "Hard_weight_only_no_output_spatial_loss",
                "frozen": ["aligner"] + ([] if self.teacher is None else ["teacher"])}

    def get_extra_state(self):
        return {"method_version": self.spec.method_version, "loss_schema": self.spec.loss_schema,
                "pipeline": self.pipeline_manifest(), "donor_binding": copy.deepcopy(self._donor_binding),
                "student_initial_state_sha256": self.student_initial_state_sha256,
                "calibration_schema": None if self.stats is None else self.spec.calibration_schema,
                "calibration_artifact_sha256": None if self.stats is None else self.stats.artifact_sha256,
                "tau_e": None if self.stats is None else self.stats.tau_e}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("New Stage 2 checkpoint donor/data/loss/calibration contract mismatch")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not strict or state_dict.get("_extra_state") != self.get_extra_state():
            raise ValueError("Strict same-contract new Stage 2 checkpoint required; old E/EQ forbidden")
        expected = self.state_dict()
        if set(state_dict) != set(expected):
            raise ValueError("Stage 2 checkpoint parameter keys mismatch before load")
        for name, value in state_dict.items():
            if name != "_extra_state":
                target = expected[name]
                if (not isinstance(value, torch.Tensor) or value.shape != target.shape
                        or value.dtype != target.dtype or not bool(torch.isfinite(value).all())):
                    raise ValueError("Stage 2 checkpoint parameter shape/dtype/finite mismatch before load")
        for role, key in (("aligner", "aligner_state_sha256"), ("teacher", "teacher_state_sha256")):
            if getattr(self, role) is not None:
                incoming = {name[len(role) + 1:]: value for name, value in state_dict.items()
                            if name.startswith(role + ".")}
                if state_dict_sha256(incoming) != self._donor_binding[key]:
                    raise ValueError("Checkpoint frozen donor " + role + " hash mismatch before load")
        result = super().load_state_dict(state_dict, strict=True, assign=assign)
        self.assert_frozen_binding()
        self._freeze()
        self._frozen_versions = self._versions()
        return result

    def frozen_predictions(self, rr):
        from .sources import validate_training_batch
        self._assert_contract()
        validate_training_batch(rr, self._donor_binding)
        return frozen_donor_predictions(self.aligner, self.teacher, rr, sensor=self.spec.sensor)

    def forward(self, rr, *, return_maps=False):
        frozen = self.frozen_predictions(rr)
        loss_mask = (warp_support_mask(*rr.pan.shape[-2:], frozen["correction"].detach())
                     if self.spec.mask_policy == MASK2_POLICY else None)
        with torch.autocast(device_type=rr.pan.device.type, enabled=False):
            student = self.student(frozen["pan_aligned"], rr.ms, rr.native_lms)
            terms = compute_loss(student["y"], frozen["teacher_y"], rr.gt,
                frozen["pan_aligned"], spec=self.spec, scharr=self.spatial,
                stats=self.stats, return_maps=return_maps, loss_mask=loss_mask)
        return {**terms, **frozen, "student": student,
                "native": {**student, "u": frozen["correction"], "correction": frozen["correction"]}}

    @torch.no_grad()
    def native_prediction(self, batch, *, return_aligner_details=False):
        if return_aligner_details:
            raise ValueError("Stage 2 deployment does not expose training-only donor diagnostics")
        return self.inference_model()(batch)

    def inference_model(self):
        return StudentInference(self.aligner, self.student, self.spec.sensor)


class StudentInference(nn.Module):
    """Deployment contains A+S only: no Teacher, GT, q, CDF, or calibration."""
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
        # Discard an evaluator-attached RR GT before any validation or tensor
        # access. Neither numerical execution nor admission depends on GT.
        batch = replace(batch, gt=None)
        batch.validate(self.student.spec.bands)
        if (batch.lineage.sensor != self.sensor
                or batch.lineage.reference != "PROVIDED_STORED_LMS"
                or batch.lineage.grid_name != "PAN_" + batch.lineage.resolution):
            raise ValueError("Native sensor/PAN-grid/stored-LMS inference contract required")
        for tensor in (batch.pan, batch.ms, batch.native_lms):
            if tensor.dtype != torch.float32:
                raise ValueError("FP32 inference observations required")
        # No ROI guard at deployment: official FR/RR inference policy is unchanged.
        with torch.autocast(device_type=batch.pan.device.type, enabled=False):
            correction = self.aligner(batch.pan, batch.native_lms)
            pan = warp_pan(batch.pan, correction)
            out = self.student(pan, batch.ms, batch.native_lms)
        return {**out, "correction": correction, "u": correction}


Stage2 = ConditionalQStage2
