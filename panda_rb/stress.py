"""RB02: native-frame, paired inference stress diagnostics (never a selector).

The injected shift and the learned correction are sampling-coordinate offsets
in (dy, dx) HR pixels. A_ZERO overrides the complete synchronized frontend.
No support-dependent crop or offset clamp is permitted here.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F

from fh12.data import RECIPE
from pa.warp import warp_pan, warp_support_mask
from tools.metrics.eval_rr import ergas, sam
from tools.eval_dlpan import psnr_global
from tools.repair_lpan import make_lpan

MODES = ('A_ON', 'A_ZERO_INFERENCE_ONLY')
RAW_SUBSET = {'scene_indices': [0, 10], 'shift_ids': ['D000', 'D017', 'D042'],
              'max_additional_failure_raw_per_curve': 8,
              'policy': 'predeclared, identical across every case/seed/mode; additional first8 failures/outliers in scene-major order'}
STRESS_KEYS = ('ergas', 'psnr', 'sam', 'edge_error_dn')


def array_digest(array):
    value = np.ascontiguousarray(array)
    header = repr((value.dtype.str, tuple(value.shape))).encode()
    return hashlib.sha256(header + value.tobytes()).hexdigest()


def validate_grid(document):
    shifts = document['shifts'] if isinstance(document, dict) else document
    if len(shifts) != 49 or [x['id'] for x in shifts] != [f'D{i:03d}' for i in range(49)]:
        raise ValueError('RB02 requires the original ordered 49-point shift grid')
    if (float(shifts[0]['dy']), float(shifts[0]['dx'])) != (0., 0.):
        raise ValueError('D000 must be the unique zero shift')
    for i, item in enumerate(shifts[1:]):
        radius = (.25, .5, 1., 2., 3., 4.)[i // 8]
        angle = (i % 8) * 45
        expected = (radius * math.sin(math.radians(angle)), radius * math.cos(math.radians(angle)))
        actual = (float(item['dy']), float(item['dx']))
        if (item['radius_hr'] != radius or item['angle_degrees'] != angle
                or not np.allclose(actual, expected, rtol=0, atol=1e-12)):
            raise ValueError(f'Registered shift changed: {item["id"]}')
    return shifts


def regenerate_lp(shifted_pan):
    """Frozen FH12 recipe: float64 DN Gaussian, float32 cache, then normalize.

Do not filter a stale LP sidecar or clamp interpolation overshoot. The PAN
passed here is already shifted exactly once with the FP32 official warp.
"""
    if shifted_pan.ndim != 4 or shifted_pan.shape[1] != 1:
        raise ValueError('Expected N1HW shifted PAN')
    dn = (shifted_pan.detach().cpu().numpy().astype(np.float64) + 1.) * 1023.5
    low_dn = make_lpan(dn).astype(np.float32)
    low = torch.from_numpy(low_dn).mul_(2. / 2047.).sub_(1.)
    return low.to(device=shifted_pan.device, dtype=torch.float32)


def shifted_inputs(pan, native_lp, epsilon):
    if epsilon.shape != (len(pan), 2) or not torch.isfinite(epsilon).all():
        raise ValueError('Invalid per-image injected shift')
    if torch.count_nonzero(epsilon).item() == 0:
        # In particular do not re-filter the original LP or zero-warp PAN.
        return pan, native_lp
    shifted = warp_pan(pan.float(), epsilon.float())
    return shifted, regenerate_lp(shifted)


@torch.no_grad()
def stress_forward(model, pan, ms, native_lp, epsilon, mode):
    if mode not in MODES:
        raise ValueError('Unregistered RB02 inference mode')
    shifted, lp = shifted_inputs(pan, native_lp, epsilon)
    base = F.interpolate(ms.float(), scale_factor=4, mode='bicubic', align_corners=False)
    estimated = model.predict_delta(shifted, base)
    # Computing A even in A_ZERO is diagnostic only; it never reaches U.
    applied = estimated if mode == 'A_ON' else torch.zeros_like(estimated)
    out = model(shifted, ms, lp, delta_override=applied)
    if not torch.allclose(out['delta'], applied.float(), atol=0, rtol=0, equal_nan=True):
        raise ValueError('Model did not honor the complete-frontend delta override')
    return out, estimated, shifted, lp


def relative_response(estimated, epsilon, native_correction):
    """Coordinate absolute SUM, unlike original q's coordinate mean."""
    return (estimated + epsilon - native_correction).abs().sum(dim=1)


def _sample_valid(mask, qy, qx):
    """Conservative 4-tap bicubic support, including zero-weight integer taps."""
    batch, _, height, width = mask.shape
    fy, fx = torch.floor(qy).long(), torch.floor(qx).long()
    valid = torch.ones((batch, 1, qy.shape[-1], qx.shape[-1]), dtype=torch.bool, device=mask.device)
    bi = torch.arange(batch, device=mask.device)[:, None, None]
    for oy in (-1, 0, 1, 2):
        yy = fy + oy
        for ox in (-1, 0, 1, 2):
            xx = fx + ox
            inside = ((yy >= 0) & (yy < height))[:, :, None] & ((xx >= 0) & (xx < width))[:, None, :]
            value = mask[:, 0][bi, yy.clamp(0, height - 1)[:, :, None], xx.clamp(0, width - 1)[:, None, :]]
            valid[:, 0] &= inside & value
    return valid


@torch.no_grad()
def geometric_coverage(height, width, epsilon, applied, margin=32):
    """Report, never mask, support of PAN and LP/HP frontend sampling.

Tracks both PAN warps and the LP 41x41 filter / offset2 decimation / bicubic
upsample / correction warp. It does not claim to model U's receptive field.
Outside-image filter taps count invalid even with replicate padding.
"""
    if height % 4 or width % 4 or min(height, width) <= 2 * margin:
        raise ValueError('Invalid fixed diagnostic ROI')
    batch = len(epsilon); device = epsilon.device
    if epsilon.shape != (batch, 2) or applied.shape != (batch, 2):
        raise ValueError('Coverage needs corresponding (dy,dx) shifts')
    if not torch.isfinite(epsilon).all() or not torch.isfinite(applied).all():
        raise FloatingPointError('Nonfinite geometry; do not clamp it')
    first = warp_support_mask(height, width, epsilon)
    is_zero = (epsilon == 0).all(1)
    first[is_zero] = True  # native input is reused without a first warp
    # Uniform shifts have rectangular supports: separable erosion is exact.
    good_y, good_x = first[:, 0].any(2), first[:, 0].any(1)
    def erode_axis(good):
        kernel = torch.ones((1, 1, 41), device=device)
        return F.conv1d(good.float()[:, None], kernel, padding=20)[:, 0] == 41
    low_y, low_x = erode_axis(good_y)[:, 2::4], erode_axis(good_x)[:, 2::4]
    low_mask = (low_y[:, :, None] & low_x[:, None, :])[:, None]
    uy = ((torch.arange(height, device=device).float() + .5) / 4 - .5).expand(batch, -1)
    ux = ((torch.arange(width, device=device).float() + .5) / 4 - .5).expand(batch, -1)
    up_low = _sample_valid(low_mask, uy, ux)
    qy = torch.arange(height, device=device).float()[None] + applied[:, :1]
    qx = torch.arange(width, device=device).float()[None] + applied[:, 1:]
    pan_mask = _sample_valid(first, qy, qx)
    low_mask = _sample_valid(up_low, qy, qx)
    def coverage(mask):
        return mask[..., margin:-margin, margin:-margin].float().mean((1, 2, 3))
    return dict(pan=coverage(pan_mask), lp=coverage(low_mask), all_pan_paths=coverage(pan_mask & low_mask))


def signed_edge_error(pred, truth):
    """Signed Scharr/32 mismatch on the fixed ROI, averaged over x/y in DN.

Reuse the training derivative kernel, with reflect padding at this diagnostic
ROI and every ROI center scored. No new magnitude-only edge convention.
"""
    from pa.losses import scharr
    pred, truth = np.asarray(pred, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    a = torch.from_numpy(np.ascontiguousarray(pred))[None]
    b = torch.from_numpy(np.ascontiguousarray(truth))[None]
    ax, ay = scharr(a); bx, by = scharr(b)
    return float(.5 * ((ax - bx).abs().mean() + (ay - by).abs().mean()))


def stress_metrics(pred, truth, margin=32):
    pred, truth = np.asarray(pred), np.asarray(truth)
    if pred.shape != (8, 256, 256) or truth.shape != pred.shape or margin != 32:
        raise ValueError('RB02 uses full256 input and the fixed192 ROI only')
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise FloatingPointError('Nonfinite RB02 prediction/reference, including borders')
    a = pred[:, margin:-margin, margin:-margin].astype(np.float64)
    b = truth[:, margin:-margin, margin:-margin].astype(np.float64)
    hwc_a, hwc_b = a.transpose(1, 2, 0), b.transpose(1, 2, 0)
    return dict(ergas=float(ergas(hwc_a, hwc_b)), psnr=float(psnr_global(hwc_a, hwc_b, 2047.)),
                sam=float(sam(hwc_a, hwc_b)), edge_error_dn=signed_edge_error(a, b))


def warp_convention_test(device='cpu'):
    size = 64
    ramp = torch.arange(size, device=device).float().view(1, 1, 1, size).expand(1, 1, size, size)
    epsilon = torch.tensor([[0., 1.]], device=device)
    shifted = warp_pan(ramp, epsilon)
    error = float((shifted[..., 4:-4, 4:-4] - ramp[..., 4:-4, 5:-3]).abs().max())
    impulse = torch.zeros(1, 1, size, size, device=device); impulse[0, 0, 32, 32] = 1
    shifted_impulse = warp_pan(impulse, epsilon)
    peak = int(shifted_impulse[0, 0].argmax())
    if error > 1e-4 or (peak // size, peak % size) != (32, 31):
        raise AssertionError('PAN sampling-coordinate sign differs from declared RB02 convention')
    return dict(passed=True, coordinate_order=['dy', 'dx'], unit='HR_pixel',
                positive_dx='source x increases; visible impulse moves left',
                positive_dx_impulse_peak=[32, 31], ramp_max_error=error,
                cancellation='c_shift + epsilon - c_native', response_reduction='abs(dy)+abs(dx)',
                mode='bicubic', padding='border', align_corners=False, dtype='float32',
                lp_recipe=RECIPE)
