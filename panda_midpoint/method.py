"""Frozen original Stage 1 A/T with a midpoint-coordinate fresh Student.

No backbone weights, layer shapes, loss formula, evaluator, or shift-warp change.
Student LRMS uses phase-corrected interp23tap after paired LR augmentation.
Teacher is evaluated on its ORIGINAL inputs, then its prediction is transported.
"""
from dataclasses import replace

import torch
from torch import nn

from pa.warp import warp_pan, warp_support_mask, two_stage_support_mask
from panda_stage2_condq.method import ConditionalQStage2, native_support
from panda_stage2_condq.losses import compute_loss
from panda_stage2_condq.sources import validate_training_batch
from .spec import MidpointSpec, geometry_manifest
from .resampling import upsample_lrms_after_augmentation


def validate_midpoint_binding(binding):
    from panda_stage2_condq.calibration import validate_binding
    binding = validate_binding(binding)
    if binding.get("midpoint_geometry") != geometry_manifest():
        raise ValueError("Midpoint geometry/Teacher transport calibration binding required")
    return binding


@torch.no_grad()
def midpoint_support(height, width, correction, *, include_teacher):
    pan = warp_support_mask(height, width, correction * .5)
    ms_gt = warp_support_mask(height, width, correction * -.5)
    mask = pan & ms_gt
    if include_teacher:
        # T uses W(P,u), and its prediction is subsequently W(T,-u/2).
        # This follows the same conservative sampler-support convention as MASK2.
        mask = mask & two_stage_support_mask(height, width, correction, correction * -.5)
    good = mask[..., 7:-7, 7:-7].flatten(1).all(1)
    if not bool(good.all()):
        raise ValueError("SUPPORT_VIOLATION: midpoint ROI8 Scharr stencil reads extrapolation")
    return {"loss_mask": mask, "warp_support_fraction": mask.float().mean(),
            "spatial_support_ok": good}


def reconstruct_midpoint(reconstructor, pan_midpoint, ms_base_midpoint, ms):
    """Reuse exactly the existing P0 backbone with an explicitly warped base.

    Calling the ordinary Reconstructor.forward would silently regenerate an
    UNWARPED bicubic base. Both x_in and the residual addition must use this
    supplied interp23tap base. The legacy spec is only the backbone constructor;
    the effective Student forward is separately pinned in MidpointSpec.manifest.
    """
    b, _, h, w = pan_midpoint.shape
    spec = reconstructor.spec
    if (spec.input_layout != "P0" or spec.base_reference != "BICUBIC_MS"
            or pan_midpoint.shape != (b, 1, h, w)
            or ms_base_midpoint.shape != (b, spec.bands, h, w)
            or ms.shape != (b, spec.bands, h // 4, w // 4) or h % 4 or w % 4):
        raise ValueError("Legacy P0 backbone constructor and matching external-base midpoint grids required")
    with torch.autocast(device_type=pan_midpoint.device.type, enabled=False):
        p, base = pan_midpoint.float(), ms_base_midpoint.float()
        x_in = torch.cat((p, base), dim=1)
        residual = reconstructor.backbone(p, None, ms.float(), p.new_ones(b), x_in=x_in)
        prediction = base + residual.float()
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError("Nonfinite midpoint reconstruction")
    return {"y": prediction, "ms_base": base, "x_in": x_in,
            "pan_aligned": p, "lpan_aligned": None}


@torch.no_grad()
def frozen_midpoint_predictions(aligner, teacher, batch, *, sensor):
    if sensor != "WV3":
        raise ValueError("This isolated midpoint campaign is WV3 only")
    batch.validate(8, training=True, require_gt=True)
    if batch.lineage.sensor != sensor:
        raise ValueError("Donor/batch sensor mismatch")
    for tensor in (batch.pan, batch.ms, batch.native_lms, batch.gt):
        if tensor.dtype != torch.float32:
            raise ValueError("Midpoint training preserves FP32 radiometry")
    for module in (aligner, teacher):
        if module is not None:
            if any(p.requires_grad for p in module.parameters()):
                raise ValueError("Midpoint A/T must be frozen")
            module.eval()
    with torch.autocast(device_type=batch.pan.device.type, enabled=False):
        # Always estimate the ORIGINAL relative shift, not a residual on warped inputs.
        correction = aligner(batch.pan, batch.native_lms).detach()
        support = midpoint_support(*batch.pan.shape[-2:], correction,
                                   include_teacher=teacher is not None)
        pan_midpoint = warp_pan(batch.pan, correction * .5)
        base = upsample_lrms_after_augmentation(batch.ms, batch.lineage.augmentations)
        # One shared sampling grid for all MS bands and HRMS target bands.
        moved = warp_pan(torch.cat((base, batch.gt), dim=1), correction * -.5)
        base_midpoint, gt_midpoint = moved.split(8, dim=1)
        teacher_out, teacher_y = None, None
        if teacher is not None:
            native_support(batch.pan, correction)  # preserve original donor admission
            teacher_pan = warp_pan(batch.pan, correction)
            teacher_out = teacher(teacher_pan, batch.ms, batch.native_lms)
            teacher_y = warp_pan(teacher_out["y"], correction * -.5)
    return {"teacher": teacher_out, "teacher_y": teacher_y,
            "pan_aligned": pan_midpoint, "ms_base_midpoint": base_midpoint,
            "gt_midpoint": gt_midpoint, "correction": correction, "u": correction,
            "pan_correction": correction * .5, "ms_correction": correction * -.5,
            **support}


class MidpointStage2(ConditionalQStage2):
    def __init__(self, spec, aligner, teacher=None, *, donor_binding, stats=None):
        if not isinstance(spec, MidpointSpec):
            raise TypeError("Explicit MidpointSpec required; legacy Stage 2 cannot resume here")
        donor_binding = validate_midpoint_binding(donor_binding)
        super().__init__(spec, aligner, teacher, donor_binding=donor_binding, stats=stats)

    def pipeline_manifest(self):
        return {**super().pipeline_manifest(), "midpoint_geometry": geometry_manifest()}

    def alignment_manifest(self):
        return {"aligner": self.aligner.manifest(), "geometry": geometry_manifest(),
                "warp": self.pipeline_manifest()["warp"], "shift_unit": "CURRENT_PAN_PIXEL",
                "references": "original_PAN/native_LMS-full_frame-no-mask"}

    def frozen_predictions(self, rr):
        self._assert_contract()
        validate_midpoint_binding(self._donor_binding)
        validate_training_batch(rr, self._donor_binding)
        return frozen_midpoint_predictions(self.aligner, self.teacher, rr, sensor=self.spec.sensor)

    def forward(self, rr, *, return_maps=False):
        frozen = self.frozen_predictions(rr)
        student = reconstruct_midpoint(self.student, frozen["pan_aligned"],
                                       frozen["ms_base_midpoint"], rr.ms)
        terms = compute_loss(student["y"], frozen["teacher_y"], frozen["gt_midpoint"],
            frozen["pan_aligned"], spec=self.spec, scharr=self.spatial, stats=self.stats,
            return_maps=return_maps, loss_mask=frozen["loss_mask"])
        return {**terms, **frozen, "student": student,
                "native": {**student, "u": frozen["correction"], "correction": frozen["correction"]}}

    def inference_model(self):
        return MidpointInference(self.aligner, self.student, self.spec.sensor)


class MidpointInference(nn.Module):
    """A+S only; interp23tap MS base, two unchanged bicubic half-shift warps."""
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
        # No GT access even when the RR evaluator attaches it to the batch.
        batch = replace(batch, gt=None)
        batch.validate(self.student.spec.bands)
        if (batch.lineage.sensor != self.sensor
                or batch.lineage.reference != "PROVIDED_STORED_LMS"
                or batch.lineage.grid_name != "PAN_" + batch.lineage.resolution):
            raise ValueError("Native sensor/PAN-grid/stored-LMS inference contract required")
        for tensor in (batch.pan, batch.ms, batch.native_lms):
            if tensor.dtype != torch.float32:
                raise ValueError("FP32 original inference observations required")
        with torch.autocast(device_type=batch.pan.device.type, enabled=False):
            correction = self.aligner(batch.pan, batch.native_lms)
            pan_midpoint = warp_pan(batch.pan, correction * .5)
            base = upsample_lrms_after_augmentation(batch.ms, batch.lineage.augmentations)
            base_midpoint = warp_pan(base, correction * -.5)
            out = reconstruct_midpoint(self.student, pan_midpoint, base_midpoint, batch.ms)
        # No deployment mask, GT, Teacher, q/CDF, reference shift, or output backwarp.
        return {**out, "correction": correction, "u": correction,
                "pan_correction": correction * .5, "ms_correction": correction * -.5}
