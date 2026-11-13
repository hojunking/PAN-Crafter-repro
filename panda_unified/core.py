"""Sensor-independent operators for fresh unified Teacher/Student experiments.

The common loss domain is the largest set whose entire 3x3 Scharr stencil has
observed Lanczos-4 support. It is shared by reconstruction, spatial fitting and
the train calibration population. It does not describe the U-Net receptive
field or change the official RR/FR evaluation crops.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from directshift48.method import ScharrAbsolute, _digest
from donor30.masking import masked_mean, validate_mask
from fh12.model import _construction_rng, _named_fresh
from panda_method.reconstruction import (
    ReconstructionSpec, Reconstructor, build_reconstructor as _build_original,
)
from panda_tune30.warp import warp_pan
from panda_tune30.reconstruction import BilinearUpConv
from panda_warp30.reconstruction import base_from_batch as _base_from_batch
from panda_warp30.reconstruction import upsampler_manifest as _upsampler_manifest


WARP_CONTRACT = 'lanczos4_whole_sample_reflection_FP32_current_pan_px'
MASK_POLICY = 'DETACHED_LANCZOS4_INTEGER_TAP_SUPPORT_ERODE3X3_COMMON_V2'
MASK2_POLICY = MASK_POLICY
SENSOR_BANDS = {'WV3': 8, 'QB': 4, 'GF2': 4}
RGB_INDICES = {'WV3': (4, 2, 1), 'QB': (2, 1, 0), 'GF2': (2, 1, 0)}
NIR_INDICES = {'WV3': 6, 'QB': 3, 'GF2': 3}


def reference_indices(sensor, mode):
    if sensor not in SENSOR_BANDS or mode not in ('RGB', 'RGBN'):
        raise ValueError('Explicit WV3/QB/GF2 and RGB/RGBN reference required')
    return RGB_INDICES[sensor] + ((NIR_INDICES[sensor],) if mode == 'RGBN' else ())


def warp_manifest():
    return {'schema': 'PANDA_UNIFIED_LANCZOS4_V2', 'method': 'lanczos4',
            'radius': 4, 'taps_per_axis': 8, 'normalization': 'SUM_TO_ONE',
            'boundary': 'WHOLE_SAMPLE_REFLECTION', 'dtype': 'FP32',
            'convention': 'OUTPUT_YX_READ_Y_PLUS_DY_X_PLUS_DX',
            'shift_unit': 'CURRENT_PAN_PIXEL', 'shift_clamping': False,
            'paths': ['stage1_native', 'stage1_epsilon_injection',
                      'validation_native', 'validation_epsilon_injection',
                      'stage2_teacher', 'stage2_student', 'calibration', 'RR_FR_inference'],
            'aligner_input_crop': 8, 'fixed_loss_roi': None,
            'mask_policy': MASK_POLICY,
            'support_index': 'ARANGE_INT64_PLUS_FLOOR_SHIFT_INT64',
            'training_mask': 'ALL_8_TAPS_OBSERVED_THEN_3X3_EROSION_OUTSIDE_FALSE',
            'reduction': 'MEAN_CHANNELS_THEN_VALID_PIXELS_PER_SAMPLE_THEN_BATCH',
            'official_metric_crop_changed': False}


def warp_support_mask(height, width, delta):
    """All eight taps must be in bounds, using the warp's exact integer indices.

    Endpoint taps remain included even when their coefficient is zero. Crucially
    this uses ``arange(int64) + floor(delta).int64`` rather than FP32
    ``floor(arange + delta)``; the latter can disagree near integer shifts.
    """
    if (type(height) is not int or type(width) is not int or min(height, width) < 2
            or not isinstance(delta, torch.Tensor) or delta.ndim != 2
            or delta.shape[1] != 2 or len(delta) < 1
            or delta.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(delta).all())):
        raise ValueError('Positive image dimensions and finite floating B2 shift required')
    with torch.no_grad():
        integer = torch.floor(delta.detach()).to(torch.int64)
        y = torch.arange(height, device=delta.device, dtype=torch.int64)[None]
        x = torch.arange(width, device=delta.device, dtype=torch.int64)[None]
        iy, ix = y + integer[:, 0, None], x + integer[:, 1, None]
        oky, okx = (iy - 3 >= 0) & (iy + 4 < height), (ix - 3 >= 0) & (ix + 4 < width)
        return (oky[:, :, None] & okx[:, None, :])[:, None]


def erode_scharr_support(mask):
    """Retain a pixel only if all nine mask values around it are true."""
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or mask.ndim != 4 or mask.shape[1] != 1 or len(mask) == 0):
        raise ValueError('Nonempty Boolean B1HW support mask required')
    with torch.no_grad():
        padded = F.pad(mask.detach(), (1, 1, 1, 1), value=False)
        return (padded.unfold(2, 3, 1).unfold(3, 3, 1).all(-1).all(-1)).contiguous()


def support(pan, delta):
    """Common maximal valid domain for every pixel loss and train calibration."""
    if pan.ndim != 4 or pan.shape[1] != 1 or delta.shape != (len(pan), 2):
        raise ValueError('Matching B1HW PAN and B2 shift required')
    if pan.device != delta.device:
        raise ValueError('PAN and shift must share a device')
    if pan.dtype not in (torch.float32, torch.float64):
        raise ValueError('Floating PAN required')
    # The numerical warp casts its displacement to image.dtype before floor().
    mask = erode_scharr_support(warp_support_mask(*pan.shape[-2:], delta.detach().to(pan.dtype)))
    validate_mask(pan, mask)  # Empty sample support is an error, never a zero loss.
    return mask


def inject(pan, epsilon):
    """Controlled translation; preserve the existing clean Aligner input crop8."""
    mask = warp_support_mask(*pan.shape[-2:], epsilon.detach().to(pan.dtype))
    if min(pan.shape[-2:]) <= 16 or not bool(mask[..., 8:-8, 8:-8].all()):
        raise ValueError('SUPPORT_VIOLATION: epsilon contaminates Aligner clean input crop8')
    return warp_pan(pan, epsilon)


class ReferenceSpatial(ScharrAbsolute):
    """Equal reference-band mean followed by absolute x/y Scharr, normalized /32."""
    def __init__(self, sensor, mode='RGBN'):
        super().__init__(sensor)
        self.mode = mode
        self.reference_indices = reference_indices(sensor, mode)

    def reference(self, image):
        self._check(image, self.bands)
        return image[:, self.reference_indices].mean(1, keepdim=True)

    def rgb_luminance(self, image):
        """Compatibility interface: actual bands are declared by ``mode``."""
        return self.reference(image)

    def magnitude(self, gray):
        return self.descriptor(gray).mean(1, keepdim=True)

    def absolute_loss(self, aligned_pan, gt, *, mask):
        self._check(aligned_pan, 1)
        self._check(gt, self.bands)
        if (len(aligned_pan) != len(gt) or aligned_pan.shape[-2:] != gt.shape[-2:]
                or aligned_pan.dtype != gt.dtype or aligned_pan.device != gt.device):
            raise ValueError('PAN and reference must share batch/grid/dtype/device')
        error = (self.descriptor(aligned_pan) - self.descriptor(self.reference(gt))).abs()
        return masked_mean(error, mask)

    def manifest(self):
        value = super().manifest()
        value.pop('rgb_indices_R_G_B'); value.pop('rgb_mean_before_gradient')
        value.update(reference_mode=self.mode, reference_indices=list(self.reference_indices),
                     reference_bands=list(self.reference_indices),
                     reference_weights=[1 / len(self.reference_indices)] * len(self.reference_indices),
                     mean_before_gradient=True, loss_mask=MASK_POLICY,
                     reduction='mean_axes_then_valid_pixels_per_sample_then_batch')
        return value


@dataclass(frozen=True)
class UnifiedReconstructionSpec:
    sensor: str
    width: int
    depth: tuple[int, int, int] = (1, 2, 3)
    pan_channels: int = 1
    input_layout: str = 'P0'
    base_reference: str = 'INTERP23_MS'

    def __post_init__(self):
        if (self.sensor not in SENSOR_BANDS or type(self.width) is not int or self.width <= 0
                or len(self.depth) != 3 or any(type(d) is not int or d <= 0 for d in self.depth)
                or self.pan_channels not in (1, 2) or self.input_layout != 'P0'
                or self.base_reference != 'INTERP23_MS'):
            raise ValueError('Explicit sensor, width, D123-like depths and interp23 P0 input required')
        object.__setattr__(self, 'depth', tuple(self.depth))

    @property
    def bands(self):
        return SENSOR_BANDS[self.sensor]

    @property
    def input_channels(self):
        return self.bands + self.pan_channels

    @property
    def uses_lpan(self):
        return False

    def manifest(self):
        return {**asdict(self), 'schema': 'PANDA_UNIFIED_UBIL_RECONSTRUCTION_V2',
                'bands': self.bands, 'input_channels': self.input_channels,
                'sampling_variant': 'UBIL', 'teacher_sampling_variant': 'UBIL',
                'norm': 'ln', 'dropout': 0., 'mode_modulation': False, 'attention': False,
                'input_order': (['CORRECTED_PAN'] if self.pan_channels == 1 else
                                ['CORRECTED_PAN', 'CORRECTED_PAN_MINUS_ORIGINAL_PAN']) +
                               [f'MS_{i}' for i in range(self.bands)],
                'down_operator': 'CONV_3X3_STRIDE2_PAD1',
                'up_operator': 'BILINEAR_X2_ALIGN_CORNERS_FALSE_THEN_CONV_3X3_STRIDE1_PAD1',
                'rotation_equivariance_guaranteed': False,
                'base_reference_is_constructor_compatibility_only': False}


class UnifiedReconstructor(Reconstructor):
    """Both roles use UBIL; callers must supply the explicit interp23 base."""
    def __init__(self, backbone, spec, role):
        if role not in ('T', 'S') or spec.pan_channels != (1 if role == 'T' else 2):
            raise ValueError('Teacher PAN1 / Student corrected-plus-difference PAN2 required')
        super().__init__(backbone, spec)
        self.pan_channels, self.role = spec.pan_channels, role
        self.sampling_variant = 'UBIL'
        self.input_variant = 'CORRECTED' if role == 'T' else 'CORRECTED_DIFF'
        self.register_buffer('_unified_identity', self._identity())

    def manifest(self):
        return {**self.spec.manifest(), 'role': self.role,
                'parameter_count': sum(p.numel() for p in self.parameters())}

    def _identity(self):
        return torch.tensor(list(bytes.fromhex(_digest(self.manifest()))), dtype=torch.uint8)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        expected, actual = self._identity(), state_dict.get(prefix + '_unified_identity')
        if (not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype
                or actual.shape != expected.shape or not torch.equal(actual.cpu(), expected)):
            raise RuntimeError('Unified architecture/role/checkpoint identity mismatch')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, *args, **kwargs):
        raise RuntimeError('Use reconstruct_with_base; implicit bicubic base is forbidden')


def build_reconstructor(sensor, width, seed, role='T', depth=(1, 2, 3)):
    """Keep historical named-fresh initialization, with UBIL in both roles."""
    spec = UnifiedReconstructionSpec(sensor, width, tuple(depth), 1 if role == 'T' else 2)
    legacy = ReconstructionSpec(spec.bands, width, tuple(depth), 'P0', 'BICUBIC_MS')
    fresh = _build_original(legacy, seed=seed, role=role)
    with _construction_rng():
        if role == 'S':
            old = fresh.backbone.input
            stem = nn.Conv2d(spec.input_channels, width, 3, padding=1)
            with torch.no_grad():
                stem.weight[:, :1].copy_(old.weight[:, :1])
                stem.weight[:, 1:2].zero_()
                stem.weight[:, 2:].copy_(old.weight[:, 1:])
                stem.bias.copy_(old.bias)
            fresh.backbone.input = stem
        for name in ('up2', 'up1'):
            old = getattr(fresh.backbone, name)
            block = BilinearUpConv(old.channels, old.out_channels)
            _named_fresh(block, seed, role, f'backbone.{name}')
            setattr(fresh.backbone, name, block)
    return UnifiedReconstructor(fresh.backbone, spec, role)


def reconstruction_manifest(model, **kwargs):
    if not isinstance(model, UnifiedReconstructor):
        raise TypeError('Unified UBIL reconstructor required')
    if any(type(getattr(model.backbone, name)) is not BilinearUpConv for name in ('up1', 'up2')):
        raise ValueError('Both upsampling blocks must use UBIL')
    if any(getattr(model.backbone, name).conv.stride != (2, 2) for name in ('down1', 'down2')):
        raise ValueError('Preserve stride2 convolution downsampling')
    for key in ('role', 'sampling_variant', 'input_variant'):
        if key in kwargs and kwargs[key] != getattr(model, key):
            raise ValueError('Declared architecture differs from instantiated model')
    return model.manifest()


def upsampler_manifest(kind='interp23'):
    if kind != 'interp23':
        raise ValueError('Unified experiments use interp23 for the reconstruction MS base')
    result = _upsampler_manifest(kind)
    result['actual_shift_warp'] = WARP_CONTRACT
    return result


def base_from_batch(batch, upsampler='interp23'):
    upsampler_manifest(upsampler)
    return _base_from_batch(batch, 'interp23')


def reconstruct_with_base(model, pan, base, ms):
    if not isinstance(model, UnifiedReconstructor):
        raise TypeError('Unified reconstructor required')
    if pan.ndim != 4 or len(pan) == 0:
        raise ValueError('Nonempty BCHW PAN views required')
    n, c, h, w = pan.shape
    bands = model.spec.bands
    if (c != model.pan_channels or h % 4 or w % 4
            or base.shape != (n, bands, h, w) or ms.shape != (n, bands, h // 4, w // 4)):
        raise ValueError('Declared PAN channels and exact MS/base grids required')
    if any(t.dtype != torch.float32 or t.device != pan.device or not bool(torch.isfinite(t).all())
           for t in (pan, base, ms)):
        raise ValueError('Finite FP32 observations on one device required')
    with torch.autocast(device_type=pan.device.type, enabled=False):
        x_in = torch.cat((pan, base), 1)
        residual = model.backbone(pan[:, :1], None, ms, pan.new_ones(n), x_in=x_in)
        y = base + residual.float()
    if not bool(torch.isfinite(y).all()):
        raise FloatingPointError('Nonfinite unified reconstruction')
    return {'y': y, 'ms_base': base, 'x_in': x_in, 'pan_aligned': pan, 'lpan_aligned': None}


@torch.no_grad()
def reference_maps(spatial, teacher_y, gt, aligned_pan):
    if teacher_y.shape != gt.shape or aligned_pan.shape != (len(gt), 1, *gt.shape[-2:]):
        raise ValueError('Teacher, GT and aligned PAN must share a grid')
    e = (teacher_y.detach() - gt.detach()).abs().mean(1, keepdim=True)
    edges = spatial.descriptor(spatial.reference(gt.detach()))
    q = (spatial.descriptor(aligned_pan.detach()) - edges).abs().mean(1, keepdim=True)
    return e, edges.mean(1, keepdim=True), q


@torch.no_grad()
def adaptive_weight(e, g, q, stats, mask, cfg):
    """Detached adaptive fitting, on exactly the common calibration population."""
    alpha, gamma, beta = float(cfg['alpha']), float(cfg['gamma']), float(cfg.get('beta', 0))
    if beta != 0 or any(not math.isfinite(v) or v < 0 for v in (alpha, gamma)):
        raise ValueError('Finite nonnegative Hard weights and beta=0 required')
    validate_mask(mask.float(), mask)
    weight = torch.ones_like(mask, dtype=torch.float32)
    if not (alpha or gamma):
        return weight
    if e is None or e.shape != mask.shape or e.device != mask.device or not bool(torch.isfinite(e).all()):
        raise ValueError('Finite Teacher error on the common grid required')
    tau = float(stats.tau_e)
    if not math.isfinite(tau) or tau < 0:
        raise ValueError('Finite nonnegative calibrated tau_e required')
    cue = e.detach() / (e.detach() + tau).clamp_min(torch.finfo(torch.float32).tiny)
    rank = torch.zeros_like(weight)
    if gamma:
        if (g is None or q is None or g.shape != mask.shape or q.shape != mask.shape
                or g.device != mask.device or q.device != mask.device):
            raise ValueError('g/q on the same common support grid required')
        values = stats.lookup_exact(g[mask], q[mask])
        if (values.shape != q[mask].shape or values.dtype != torch.float32
                or values.device != mask.device or not bool(torch.isfinite(values).all())
                or not bool(((values >= 0) & (values <= 1)).all())):
            raise ValueError('Finite conditional ranks in [0,1] required')
        rank[mask] = values
    weight[mask] += (alpha * cue + gamma * cue * rank)[mask]
    return weight.detach()


@torch.no_grad()
def frozen_predictions(aligner, teacher, batch, *, geometry_key='s100'):
    if geometry_key != 's100':
        raise ValueError('Unified experiments use full Aligner scale1')
    bands = teacher.spec.bands
    batch.validate(bands, training=True, require_gt=True)
    if batch.lineage.sensor != teacher.spec.sensor:
        raise ValueError('Teacher and data sensor mismatch')
    if any(t.dtype != torch.float32 for t in (batch.pan, batch.ms, batch.native_lms, batch.gt)):
        raise ValueError('FP32 training observations required')
    for module in (aligner, teacher):
        if module is None or any(p.requires_grad for p in module.parameters()):
            raise ValueError('Same-checkpoint frozen Aligner and Teacher required')
        module.eval()
    u = aligner(batch.pan, batch.native_lms).detach()
    pan = warp_pan(batch.pan, u)
    base = base_from_batch(batch)
    mask = support(batch.pan, u)
    t = reconstruct_with_base(teacher, pan, base, batch.ms)
    return {'u': u, 'correction': u, 'pan_input': pan, 'pan_aligned': pan,
            'pan_loss': pan, 'ms_base_frame': base, 'teacher': t, 'teacher_y': t['y'],
            'gt_loss': batch.gt, 'loss_mask': mask,
            'warp_support_fraction': warp_support_mask(*pan.shape[-2:], u).float().mean(),
            'valid_fraction': mask.float().mean(),
            'spatial_support_ok': mask.flatten(1).any(1)}
