"""Fixed A/T; fresh S with explicit coordinate transport and output inverse."""
from dataclasses import replace
from copy import deepcopy
import copy

import torch
from torch import nn

from pa.warp import warp_pan, warp_support_mask, two_stage_support_mask
from panda_stage2_condq.method import ConditionalQStage2
from panda_stage2_condq.losses import compute_loss
from panda_stage2_condq.sources import validate_training_batch
from panda_method.calibration import canonical_sha256, state_dict_sha256
from directshift48.method import ScharrAbsolute
from .spec import (BandSpec, geometry_manifest, upsampling_manifest,
                   q_reference_manifest, teacher_binding)
from .reconstruction import (base_from_batch, reconstruct_with_base, BandReconstructor,
                             build_reconstructor)
from .aligner import BandPANAligner, aligned_pan_views, support_for_shifts
from .contract import CAMPAIGN_ID


def validate_band_binding(binding, spec):
    from panda_stage2_condq.calibration import validate_binding
    binding = validate_binding(binding)
    teacher = binding.get('band30_teacher')
    if teacher != teacher_binding(spec):
        raise ValueError('Fresh Band30 Teacher architecture/target/upsampler/campaign binding required')
    if binding.get('band30_geometry') != geometry_manifest(spec):
        raise ValueError('Band30 geometry/strength/upsampler calibration binding required')
    if binding.get('band30_q_reference') != q_reference_manifest(spec):
        raise ValueError('Band30 q-reference calibration binding required')
    from panda_method.calibration import canonical_sha256
    if canonical_sha256(binding.get('band30_reconstruction_spec')) != canonical_sha256(spec.reconstruction_spec.manifest()):
        raise ValueError('Truthful Band30 reconstruction binding required')
    return binding


@torch.no_grad()
def warp_support(height, width, u, spec, *, include_teacher):
    """Track successive sampling support, never treat inverse as cancellation."""
    expected = (len(u), 8, 2) if spec.pan_channels == 8 else (len(u), 2)
    if u.shape != expected or min(height, width) <= 16:
        raise ValueError('Declared Aligner shape and nonempty ROI8 required')
    if spec.frame == 'band_pan':
        mask = support_for_shifts(height, width, u)
        good = mask[..., 7:-7, 7:-7].flatten(1).all(1)
        if not bool(good.all()):
            raise ValueError('SUPPORT_VIOLATION: Band30 any-band PAN warp violates ROI8')
        return {'loss_mask': mask, 'warp_support_fraction': mask.float().mean(),
                'spatial_support_ok': good}
    if spec.shift_scale == 0:
        mask = torch.ones(len(u), 1, height, width, device=u.device, dtype=torch.bool)
    else:
        d, rho = u * spec.shift_scale, spec.pan_fraction
        p, m = d * rho, d * (rho - 1.)
        ones = torch.ones(len(u), 1, height, width, device=u.device, dtype=torch.bool)
        if spec.returns_output:
            back = -m
            pan_support = (warp_support_mask(height, width, back) if rho == 0 else
                           two_stage_support_mask(height, width, p, back))
            mask = pan_support & two_stage_support_mask(height, width, m, back)
            mask &= warp_support_mask(height, width, d)  # q reference in loss frame
        else:
            mask = warp_support_mask(height, width, p) if rho else ones
            if rho != 1:
                mask &= warp_support_mask(height, width, m)  # MS base and transported GT
        if spec.frame == 'mid' and include_teacher:
            mask &= two_stage_support_mask(height, width, u, m)
    if include_teacher and not (spec.frame == 'mid' and spec.shift_scale != 0):
        mask &= warp_support_mask(height, width, u)
    good = mask[..., 7:-7, 7:-7].flatten(1).all(1)
    if not bool(good.all()):
        raise ValueError('SUPPORT_VIOLATION: Band30 ROI8 Scharr stencil reads extrapolation')
    return {'loss_mask': mask, 'warp_support_fraction': mask.float().mean(),
            'spatial_support_ok': good}


def _inputs(batch, u, spec):
    base = base_from_batch(batch, spec.upsampler)
    d = u * spec.shift_scale
    p, m = d * spec.pan_fraction, d * (spec.pan_fraction - 1.)
    moved_pan = batch.pan if spec.shift_scale == 0 else aligned_pan_views(batch.pan, p)
    moved_base = base if spec.shift_scale == 0 or spec.pan_fraction == 1 else warp_pan(base, m)
    return {'pan_input': moved_pan, 'ms_base_frame': moved_base,
            'correction': u, 'u': u, 'pan_correction': p, 'ms_correction': m,
            'output_correction': -m if spec.returns_output else torch.zeros_like(m)}


def q_pan_reference(pan_views, spec):
    if pan_views.ndim != 4 or pan_views.shape[1] != spec.pan_channels:
        raise ValueError('q PAN views do not match declared single/bandwise geometry')
    return pan_views[:, [4, 2, 1]].mean(1, keepdim=True) if spec.pan_channels == 8 else pan_views


def reconstruct_output(student, inputs, ms, spec):
    """Inverse is inside the Student graph: Hard/Soft supervise returned output."""
    raw = reconstruct_with_base(student, inputs['pan_input'], inputs['ms_base_frame'], ms)
    y = raw['y']
    if spec.returns_output and spec.shift_scale != 0:
        y = warp_pan(y, inputs['output_correction'])
    return {**raw, 'y': y, 'y_frame': raw['y'],
            'output_coordinates': spec.output_coordinates}


@torch.no_grad()
def frozen_predictions(aligner, teacher, batch, *, spec):
    batch.validate(spec.bands, training=True, require_gt=True)
    if batch.lineage.sensor != spec.sensor:
        raise ValueError('Band30 source sensor mismatch')
    for tensor in (batch.pan, batch.ms, batch.native_lms, batch.gt):
        if tensor.dtype != torch.float32:
            raise ValueError('Band30 preserves FP32 training radiometry')
    for module in (aligner, teacher):
        if module is not None:
            if any(p.requires_grad for p in module.parameters()):
                raise ValueError('Band30 A/T must stay frozen')
            module.eval()
    with torch.autocast(device_type=batch.pan.device.type, enabled=False):
        u = aligner(batch.pan, batch.native_lms).detach()
        support = warp_support(*batch.pan.shape[-2:], u, spec, include_teacher=teacher is not None)
        inputs = _inputs(batch, u, spec)
        original_teacher = None
        if teacher is not None:
            # Always original Stage 1 function, independent of Student rho/strength.
            original_teacher = reconstruct_with_base(teacher, aligned_pan_views(batch.pan, u),
                base_from_batch(batch, spec.upsampler), batch.ms)
        t = None if original_teacher is None else original_teacher['y']
        gt, pan_loss = batch.gt, q_pan_reference(inputs['pan_input'], spec)
        if spec.frame == 'mid' and spec.shift_scale != 0:
            gt = warp_pan(batch.gt, inputs['ms_correction'])
            if t is not None:
                t = warp_pan(t, inputs['ms_correction'])
        elif spec.returns_output and spec.shift_scale != 0:
            pan_loss = warp_pan(batch.pan, u * spec.shift_scale)
    return {**inputs, **support, 'teacher': original_teacher, 'teacher_y': t,
            'gt_loss': gt, 'pan_loss': pan_loss, 'pan_aligned': pan_loss}


class BandStage2(ConditionalQStage2):
    def __init__(self, spec, aligner, teacher=None, *, donor_binding, stats=None):
        nn.Module.__init__(self)
        if not isinstance(spec, BandSpec) or not isinstance(aligner, BandPANAligner):
            raise TypeError("Explicit BandSpec and original BandPANAligner required")
        donor_binding = validate_band_binding(donor_binding, spec)
        if aligner.ms_bands != spec.bands or aligner.outputs != spec.aligner_outputs:
            raise ValueError("Aligner sensor band mismatch")
        if spec.variant == "gt_only":
            if teacher is not None or stats is not None:
                raise ValueError("gt_only must not construct Teacher or calibration")
        elif (not isinstance(teacher, BandReconstructor)
              or teacher.spec != spec.reconstruction_spec or stats is None):
            raise ValueError("Same W112-D123/P0/BICUBIC_MS Teacher and new calibration required")
        if not isinstance(donor_binding, dict):
            raise TypeError("Pinned donor metadata dictionary required")
        from panda_stage2_condq.calibration import validate_binding
        donor_binding = validate_binding(donor_binding)
        for key in ("aligner_state_sha256", "teacher_state_sha256", "source_manifest_sha256"):
            value = donor_binding.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Missing pinned donor " + key)
        if donor_binding.get("sensor") != spec.sensor or not donor_binding.get("donor_checkpoint_id"):
            raise ValueError("Donor sensor and checkpoint ID required")
        if stats is not None:
            from panda_stage2_condq.calibration import CalibrationStats
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

    def pipeline_manifest(self):
        value = super().pipeline_manifest()
        teacher = deepcopy(value['teacher'])
        value['teacher_constructor_spec'] = teacher
        value['teacher'] = {**teacher,
            'base_reference': upsampling_manifest(self.spec.upsampler)['base_reference'],
            'base_execution': 'FRESH_STAGE1_EXTERNAL_BASE_X_IN_AND_RESIDUAL_ADDITION'}
        value['band30_geometry'] = geometry_manifest(self.spec)
        value['band30_q_reference'] = q_reference_manifest(self.spec)
        value['band30_teacher'] = teacher_binding(self.spec)
        return value

    def alignment_manifest(self):
        return {'aligner': self.aligner.manifest(), 'geometry': geometry_manifest(self.spec),
                'warp': self.pipeline_manifest()['warp'],
                'references': 'original_PAN/native_LMS-full_frame-no-mask',
                'shift_unit': 'CURRENT_PAN_PIXEL'}

    def frozen_predictions(self, rr):
        self._assert_contract()
        validate_band_binding(self._donor_binding, self.spec)
        validate_training_batch(rr, self._donor_binding)
        return frozen_predictions(self.aligner, self.teacher, rr, spec=self.spec)

    def forward(self, rr, *, return_maps=False):
        f = self.frozen_predictions(rr)
        with torch.autocast(device_type=rr.pan.device.type, enabled=False):
            student = reconstruct_output(self.student, f, rr.ms, self.spec)
            loss = compute_loss(student['y'], f['teacher_y'], f['gt_loss'], f['pan_loss'],
                spec=self.spec, scharr=self.spatial, stats=self.stats,
                return_maps=return_maps, loss_mask=f['loss_mask'])
        return {**loss, **f, 'student': student,
                'native': {**student, **{k: f[k] for k in
                    ('u', 'correction', 'pan_correction', 'ms_correction', 'output_correction')}}}

    def inference_model(self):
        return BandInference(self.aligner, self.student, self.spec)


class BandInference(nn.Module):
    """Only A+S and their deployed upsampler/warps; no GT/Teacher/calibration."""
    def __init__(self, aligner, student, spec):
        super().__init__()
        if not isinstance(spec, BandSpec):
            raise TypeError('Explicit BandSpec required for inference')
        if (not isinstance(aligner, BandPANAligner) or aligner.outputs != spec.aligner_outputs
                or not isinstance(student, BandReconstructor) or student.spec != spec.reconstruction_spec):
            raise ValueError('Declared Band30 head and reconstruction dimensions required')
        self.aligner, self.student, self.spec = aligner, student, spec
        self.aligner.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.aligner.eval()
        return self

    @torch.no_grad()
    def forward(self, batch):
        batch = replace(batch, gt=None)
        batch.validate(self.spec.bands)
        if (batch.lineage.sensor != self.spec.sensor
                or batch.lineage.reference != 'PROVIDED_STORED_LMS'
                or batch.lineage.grid_name != 'PAN_' + batch.lineage.resolution):
            raise ValueError('Original current-PAN-grid/stored-LMS inference required')
        if any(t.dtype != torch.float32 for t in (batch.pan, batch.ms, batch.native_lms)):
            raise ValueError('Band30 inference requires FP32 observations')
        with torch.autocast(device_type=batch.pan.device.type, enabled=False):
            u = self.aligner(batch.pan, batch.native_lms)
            inputs = _inputs(batch, u, self.spec)
            out = reconstruct_output(self.student, inputs, batch.ms, self.spec)
        return {**out, **{k: inputs[k] for k in
            ('u', 'correction', 'pan_correction', 'ms_correction', 'output_correction')}}
