"""Fixed A/T; fresh S with explicit coordinate transport and output inverse."""
from dataclasses import replace
from copy import deepcopy

import torch
from torch import nn

from pa.warp import warp_pan, warp_support_mask, two_stage_support_mask
from panda_stage2_condq.method import ConditionalQStage2
from panda_stage2_condq.losses import compute_loss
from panda_stage2_condq.sources import validate_training_batch
from .spec import WarpSpec, geometry_manifest, upsampling_manifest
from .reconstruction import base_from_batch, reconstruct_with_base
from .contract import CAMPAIGN_ID


def validate_warp_binding(binding, spec):
    from panda_stage2_condq.calibration import validate_binding
    binding = validate_binding(binding)
    teacher = binding.get('warp30_teacher')
    if (not isinstance(teacher, dict) or teacher.get('upsampler') != spec.upsampler
            or teacher.get('campaign_id') != CAMPAIGN_ID):
        raise ValueError('Fresh Warp30 Teacher upsampler/campaign binding required')
    if binding.get('warp30_geometry') != geometry_manifest(spec):
        raise ValueError('Warp30 geometry/strength/upsampler calibration binding required')
    return binding


@torch.no_grad()
def warp_support(height, width, u, spec, *, include_teacher):
    """Track successive sampling support, never treat inverse as cancellation."""
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
        raise ValueError('SUPPORT_VIOLATION: Warp30 ROI8 Scharr stencil reads extrapolation')
    return {'loss_mask': mask, 'warp_support_fraction': mask.float().mean(),
            'spatial_support_ok': good}


def _inputs(batch, u, spec):
    base = base_from_batch(batch, spec.upsampler)
    d = u * spec.shift_scale
    p, m = d * spec.pan_fraction, d * (spec.pan_fraction - 1.)
    moved_pan = batch.pan if spec.shift_scale == 0 or spec.pan_fraction == 0 else warp_pan(batch.pan, p)
    moved_base = base if spec.shift_scale == 0 or spec.pan_fraction == 1 else warp_pan(base, m)
    return {'pan_input': moved_pan, 'ms_base_frame': moved_base,
            'correction': u, 'u': u, 'pan_correction': p, 'ms_correction': m,
            'output_correction': -m if spec.returns_output else torch.zeros_like(m)}


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
        raise ValueError('Warp30 source sensor mismatch')
    for tensor in (batch.pan, batch.ms, batch.native_lms, batch.gt):
        if tensor.dtype != torch.float32:
            raise ValueError('Warp30 preserves FP32 training radiometry')
    for module in (aligner, teacher):
        if module is not None:
            if any(p.requires_grad for p in module.parameters()):
                raise ValueError('Warp30 A/T must stay frozen')
            module.eval()
    with torch.autocast(device_type=batch.pan.device.type, enabled=False):
        u = aligner(batch.pan, batch.native_lms).detach()
        support = warp_support(*batch.pan.shape[-2:], u, spec, include_teacher=teacher is not None)
        inputs = _inputs(batch, u, spec)
        original_teacher = None
        if teacher is not None:
            # Always original Stage 1 function, independent of Student rho/strength.
            original_teacher = reconstruct_with_base(teacher, warp_pan(batch.pan, u),
                base_from_batch(batch, spec.upsampler), batch.ms)
        t = None if original_teacher is None else original_teacher['y']
        gt, pan_loss = batch.gt, inputs['pan_input']
        if spec.frame == 'mid' and spec.shift_scale != 0:
            gt = warp_pan(batch.gt, inputs['ms_correction'])
            if t is not None:
                t = warp_pan(t, inputs['ms_correction'])
        elif spec.returns_output and spec.shift_scale != 0:
            pan_loss = warp_pan(batch.pan, u * spec.shift_scale)
    return {**inputs, **support, 'teacher': original_teacher, 'teacher_y': t,
            'gt_loss': gt, 'pan_loss': pan_loss, 'pan_aligned': pan_loss}


class WarpStage2(ConditionalQStage2):
    def __init__(self, spec, aligner, teacher=None, *, donor_binding, stats=None):
        if not isinstance(spec, WarpSpec):
            raise TypeError('Explicit WarpSpec required')
        donor_binding = validate_warp_binding(donor_binding, spec)
        super().__init__(spec, aligner, teacher, donor_binding=donor_binding, stats=stats)

    def pipeline_manifest(self):
        value = super().pipeline_manifest()
        teacher = deepcopy(value['teacher'])
        value['teacher_constructor_spec'] = teacher
        value['teacher'] = {**teacher,
            'base_reference': upsampling_manifest(self.spec.upsampler)['base_reference'],
            'base_execution': 'FRESH_STAGE1_EXTERNAL_BASE_X_IN_AND_RESIDUAL_ADDITION'}
        value['warp30_geometry'] = geometry_manifest(self.spec)
        return value

    def alignment_manifest(self):
        return {'aligner': self.aligner.manifest(), 'geometry': geometry_manifest(self.spec),
                'warp': self.pipeline_manifest()['warp'],
                'references': 'original_PAN/native_LMS-full_frame-no-mask',
                'shift_unit': 'CURRENT_PAN_PIXEL'}

    def frozen_predictions(self, rr):
        self._assert_contract()
        validate_warp_binding(self._donor_binding, self.spec)
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
        return WarpInference(self.aligner, self.student, self.spec)


class WarpInference(nn.Module):
    """Only A+S and their deployed upsampler/warps; no GT/Teacher/calibration."""
    def __init__(self, aligner, student, spec):
        super().__init__()
        if not isinstance(spec, WarpSpec):
            raise TypeError('Explicit WarpSpec required for inference')
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
            raise ValueError('Warp30 inference requires FP32 observations')
        with torch.autocast(device_type=batch.pan.device.type, enabled=False):
            u = self.aligner(batch.pan, batch.native_lms)
            inputs = _inputs(batch, u, self.spec)
            out = reconstruct_output(self.student, inputs, batch.ms, self.spec)
        return {**out, **{k: inputs[k] for k in
            ('u', 'correction', 'pan_correction', 'ms_correction', 'output_correction')}}
