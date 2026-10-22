"""Frozen G8 A/T, actual positive-scale PAN warping and the original single-view Hard loss."""
from copy import deepcopy
from dataclasses import replace

import torch
from torch import nn

from directshift48.method import ScharrAbsolute
from pa.warp import warp_pan, warp_support_mask
from panda_band30.aligner import BandPANAligner
from panda_band30.reconstruction import (
    BandReconstructor, build_reconstructor, base_from_batch,
    reconstruct_with_base, upsampler_manifest,
)
from panda_method.calibration import canonical_sha256, state_dict_sha256
from panda_midpoint.contract import UPSAMPLING
from panda_stage2_condq.method import ConditionalQStage2
from panda_stage2_condq.losses import compute_loss
from panda_stage2_condq.sources import validate_training_batch
from .spec import MatureSpec, HORIZON, geometry_manifest, policy_manifest


def applied_correction(u, geometry_key):
    if (not isinstance(u, torch.Tensor) or u.ndim != 2 or u.shape[1] != 2
            or u.dtype != torch.float32 or not bool(torch.isfinite(u).all())):
        raise ValueError("Finite FP32 B2 current-PAN-pixel correction required")
    geometry = geometry_manifest(geometry_key)
    return u * geometry["scale"]


def inputs_for_geometry(batch, u, geometry_key):
    d = applied_correction(u, geometry_key)
    return {
        "pan_input": warp_pan(batch.pan, d),
        "ms_base_frame": base_from_batch(batch, "interp23"),
        "correction": u, "u": u, "pan_correction": d,
        "ms_correction": torch.zeros_like(u), "output_correction": torch.zeros_like(u),
        "geometry_key": geometry_key,
    }


@torch.no_grad()
def warp_support(height, width, u, geometry_key, *, include_teacher=True):
    applied = applied_correction(u.detach(), geometry_key)
    if min(height, width) <= 16:
        raise ValueError("Nonempty fixed ROI8 required")
    mask = warp_support_mask(height, width, applied)
    if include_teacher:
        mask &= warp_support_mask(height, width, u.detach())
    good = mask[..., 7:-7, 7:-7].flatten(1).all(1)
    if not bool(good.all()):
        raise ValueError("SUPPORT_VIOLATION: mature PAN/Teacher warp violates fixed ROI8")
    return {"loss_mask": mask, "warp_support_fraction": mask.float().mean(),
            "spatial_support_ok": good}


@torch.no_grad()
def frozen_predictions(aligner, teacher, batch, *, geometry_key):
    batch.validate(8, training=True, require_gt=True)
    if batch.lineage.sensor != "WV3":
        raise ValueError("Mature30 only admits WV3")
    if any(t.dtype != torch.float32 for t in (batch.pan, batch.ms, batch.native_lms, batch.gt)):
        raise ValueError("Mature30 preserves FP32 training radiometry")
    for module in (aligner, teacher):
        if module is None or any(p.requires_grad for p in module.parameters()):
            raise ValueError("Same-checkpoint frozen Aligner and Teacher are required")
        module.eval()
    with torch.autocast(device_type=batch.pan.device.type, enabled=False):
        u = aligner(batch.pan, batch.native_lms).detach()
        inputs = inputs_for_geometry(batch, u, geometry_key)
        support = warp_support(*batch.pan.shape[-2:], u, geometry_key)
        # Teacher never sees the Student phase/cap. Base, GT and output stay MS.
        original_teacher = reconstruct_with_base(
            teacher, warp_pan(batch.pan, u), inputs["ms_base_frame"], batch.ms)
    return {**inputs, **support, "teacher": original_teacher,
            "teacher_y": original_teacher["y"], "gt_loss": batch.gt,
            "pan_loss": inputs["pan_input"], "pan_aligned": inputs["pan_input"]}


class MatureStage2(ConditionalQStage2):
    def __init__(self, spec, aligner, teacher, *, donor_binding, stats_bank):
        nn.Module.__init__(self)
        from .calibration import validate_base_binding, validate_stats_bank
        if not isinstance(spec, MatureSpec):
            raise TypeError("Explicit MatureSpec required")
        if (not isinstance(aligner, BandPANAligner) or aligner.outputs != 2
                or not isinstance(teacher, BandReconstructor)
                or teacher.spec != spec.reconstruction_spec):
            raise ValueError("G8 global2 Aligner and same-capacity interp23 Teacher required")
        self.spec, self.aligner, self.teacher = spec, aligner, teacher
        self._donor_binding = validate_base_binding(donor_binding, spec)
        self.stats_bank = validate_stats_bank(stats_bank, self._donor_binding, spec, full_hash=True)
        # The inherited protocol has a single-stat slot. It stays empty; the
        # complete immutable bank is validated and serialized below instead.
        self.stats = None
        self.student = build_reconstructor(spec.reconstruction_spec, seed=spec.seed, role="S")
        self.spatial = ScharrAbsolute("WV3")
        self.student_initial_state_sha256 = state_dict_sha256(self.student)
        if self.student_initial_state_sha256 == self._donor_binding["teacher_state_sha256"]:
            raise ValueError("Student must be freshly initialized, not copied from Teacher")
        parameter_groups = [{id(p) for p in m.parameters()}
                            for m in (self.aligner, self.teacher, self.student)]
        if any(parameter_groups[i] & parameter_groups[j] for i in range(3) for j in range(i)):
            raise ValueError("Aligner, Teacher and Student cannot share parameters")
        self.register_buffer("_completed_update_state", torch.zeros((), dtype=torch.int64))
        self._freeze()
        self.assert_frozen_binding()
        self._frozen_versions = self._versions()
        self._contract_sha256 = canonical_sha256(self.get_extra_state())

    @property
    def completed_update(self):
        return int(self._completed_update_state.item())

    def set_completed_update(self, completed_update):
        if type(completed_update) is not int or not 0 <= completed_update <= HORIZON:
            raise ValueError("Completed update outside immutable mature schedule")
        self._completed_update_state.fill_(completed_update)

    @property
    def calibration_memory_bytes(self):
        return sum(stats.memory_bytes for stats in self.stats_bank.values())

    def move_calibration(self, device, *, memory_budget_bytes):
        if (type(memory_budget_bytes) is not int or memory_budget_bytes <= 0
                or self.calibration_memory_bytes > memory_budget_bytes):
            raise MemoryError("Entire exact phase bank exceeds explicit device budget")
        self.stats_bank = {key: stats.to(device, memory_budget_bytes=memory_budget_bytes)
                           for key, stats in self.stats_bank.items()}
        self._assert_contract()
        return self

    def pipeline_manifest(self):
        result = super().pipeline_manifest()
        result.update(
            teacher_constructor_spec=self.spec.reconstruction_spec.manifest(),
            teacher={**self.spec.reconstruction_spec.manifest(),
                     "base_reference": UPSAMPLING["base_reference"],
                     "base_execution": "FROZEN_STAGE1_EXTERNAL_BASE_AND_FULL_U_PAN"},
            mature30_policy=policy_manifest(self.spec),
        )
        return result

    def get_extra_state(self):
        # Progress is a dedicated tensor buffer, not a mutable scientific
        # contract. All phase/table identities remain constant across training.
        return {
            "method_version": self.spec.method_version, "loss_schema": self.spec.loss_schema,
            "pipeline": self.pipeline_manifest(), "donor_binding": deepcopy(self._donor_binding),
            "student_initial_state_sha256": self.student_initial_state_sha256,
            "calibration_schema": self.spec.calibration_schema,
            "calibration_bank": {
                key: {"artifact_sha256": stats.artifact_sha256, "tau_e": stats.tau_e,
                      "binding": stats.binding}
                for key, stats in sorted(self.stats_bank.items())
            },
            "progress_state": "SERIALIZED_INT64_COMPLETED_UPDATE_BUFFER",
        }

    def _assert_contract(self):
        super()._assert_contract()
        from .calibration import validate_stats_bank
        validate_stats_bank(self.stats_bank, self._donor_binding, self.spec)
        counter = self._completed_update_state
        if counter.shape != () or counter.dtype != torch.int64 or not 0 <= self.completed_update <= HORIZON:
            raise ValueError("Invalid mature completed-update state")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        counter = state_dict.get("_completed_update_state")
        if (not isinstance(counter, torch.Tensor) or counter.dtype != torch.int64
                or counter.shape != () or not 0 <= int(counter) <= HORIZON):
            raise ValueError("Missing/invalid mature completed-update checkpoint state")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def deployment_manifest(self):
        key = self.spec.geometry_at_completed(self.completed_update)
        return {
            "schema": "PANDA_SCALE30_DEPLOYMENT_V1",
            "policy_id": self.spec.policy_id, "completed_update": self.completed_update,
            "geometry_key": key, "geometry": geometry_manifest(key),
            "shift_unit": "CURRENT_PAN_PIXEL_NO_X4", "output_coordinates": "MS",
            "teacher_required": False, "calibration_required": False,
        }

    def alignment_manifest(self):
        return {"aligner": self.aligner.manifest(), "policy": policy_manifest(self.spec),
                "deployment": self.deployment_manifest(),
                "warp": "bicubic_border_FP32_align_corners_false_current_pan_px",
                "references": "original_PAN/native_LMS-full_frame-no-mask",
                "shift_unit": "CURRENT_PAN_PIXEL"}

    def frozen_predictions(self, rr):
        self._assert_contract()
        validate_training_batch(rr, self._donor_binding)
        key = self.spec.geometry_at_next(self.completed_update)
        return frozen_predictions(self.aligner, self.teacher, rr, geometry_key=key)

    def forward(self, rr, *, return_maps=False):
        frozen = self.frozen_predictions(rr)
        key = frozen["geometry_key"]
        with torch.autocast(device_type=rr.pan.device.type, enabled=False):
            student = reconstruct_with_base(
                self.student, frozen["pan_input"], frozen["ms_base_frame"], rr.ms)
            terms = compute_loss(
                student["y"], frozen["teacher_y"], frozen["gt_loss"], frozen["pan_loss"],
                spec=self.spec, scharr=self.spatial, stats=self.stats_bank[key],
                return_maps=return_maps, loss_mask=frozen["loss_mask"])
        return {**terms, **frozen, "student": student,
                "native": {**student, **{name: frozen[name] for name in
                           ("u", "correction", "pan_correction", "ms_correction", "output_correction")}}}

    def inference_model(self):
        return MatureInference(self.aligner, self.student, self.spec,
                               completed_update=self.completed_update)


class MatureInference(nn.Module):
    """Only A+S and the checkpoint-bound policy; no GT/Teacher/calibration."""
    def __init__(self, aligner, student, spec, *, completed_update):
        super().__init__()
        if (not isinstance(spec, MatureSpec)
                or not isinstance(aligner, BandPANAligner) or aligner.outputs != 2
                or not isinstance(student, BandReconstructor)
                or student.spec != spec.reconstruction_spec):
            raise ValueError("Matching mature global2 inference architecture required")
        self.geometry_key = spec.geometry_at_completed(completed_update)
        self.spec, self.completed_update = spec, completed_update
        self.aligner, self.student = aligner, student
        self.aligner.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.aligner.eval()
        return self

    @torch.no_grad()
    def forward(self, batch):
        batch = replace(batch, gt=None)
        batch.validate(8)
        if (batch.lineage.sensor != "WV3"
                or batch.lineage.reference != "PROVIDED_STORED_LMS"
                or batch.lineage.grid_name != "PAN_" + batch.lineage.resolution):
            raise ValueError("Original current-PAN-grid/stored-LMS inference required")
        if any(t.dtype != torch.float32 for t in (batch.pan, batch.ms, batch.native_lms)):
            raise ValueError("Inference observations must be FP32")
        with torch.autocast(device_type=batch.pan.device.type, enabled=False):
            u = self.aligner(batch.pan, batch.native_lms)
            inputs = inputs_for_geometry(batch, u, self.geometry_key)
            out = reconstruct_with_base(self.student, inputs["pan_input"],
                                        inputs["ms_base_frame"], batch.ms)
        return {**out, **{name: inputs[name] for name in
                         ("u", "correction", "pan_correction", "ms_correction", "output_correction")},
                "pan_aligned": inputs["pan_input"], "geometry_key": self.geometry_key}
