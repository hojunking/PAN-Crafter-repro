"""Experimental differentiable global shifts, explicitly outside production.

Convention: W(P, d)[b,c,y,x] = P[b,c,y + d_y, x + d_x].  The delta
tensor stays in the autograd graph.  Images and outputs are FP32 or FP64.

FFT reflect padding does not make finite images periodic or band limited.  The
real projection discards imaginary Nyquist components, and cropping followed
by re-padding prevents a shift/inverse-shift pair from being a unitary inverse.
Consequently this module makes no lossless-warp or finite-support claim for FFT.
None of these functions changes pa.warp or the frozen Teacher's warp.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


METHODS = ("fft", "lanczos4", "lanczos6", "kaiser12_beta6")


def _validate(image, delta):
    if (not isinstance(image, torch.Tensor) or image.ndim != 4
            or image.dtype not in (torch.float32, torch.float64)
            or min(image.shape) < 1 or min(image.shape[-2:]) < 2):
        raise ValueError("image must be a nonempty FP32/FP64 B,C,H,W tensor, H,W >= 2")
    if (not isinstance(delta, torch.Tensor) or delta.shape != (image.shape[0], 2)
            or delta.dtype not in (torch.float32, torch.float64)
            or delta.device != image.device):
        raise ValueError("delta must be an FP32/FP64 B,2 tensor on the image device")
    if not bool(torch.isfinite(image).all()) or not bool(torch.isfinite(delta).all()):
        raise ValueError("image and delta must be finite")
    return delta.to(dtype=image.dtype)


def _padding(pad):
    if type(pad) is not int or pad < 1:
        raise ValueError("FFT reflection padding must be a positive integer")
    return pad


def fft_shift(image, delta, *, padding=32):
    """Phase-shift a reflect-padded image and return its cropped real part.

    Padding must exceed each applied displacement and be smaller than H and W.
    This is a boundary assumption, not a guarantee of uncontaminated interior.
    PyTorch's reflect convention does not repeat the boundary pixel.
    """
    d = _validate(image, delta)
    p = _padding(padding)
    if p >= min(image.shape[-2:]) or bool((d.abs() >= p).any()):
        raise ValueError("FFT padding must be < min(H,W) and > abs(delta) on both axes")
    padded = F.pad(image, (p, p, p, p), mode="reflect")
    height, width = padded.shape[-2:]
    ky = torch.fft.fftfreq(height, device=image.device, dtype=image.dtype)[None, :, None]
    kx = torch.fft.fftfreq(width, device=image.device, dtype=image.dtype)[None, None, :]
    angle = 2 * math.pi * (ky * d[:, 0, None, None] + kx * d[:, 1, None, None])
    phase = torch.exp(1j * angle)[:, None]
    shifted = torch.fft.ifft2(torch.fft.fft2(padded) * phase).real
    return shifted[..., p:-p, p:-p]


def _kernel(radius, window, beta):
    if type(radius) is not int or radius < 2:
        raise ValueError("windowed-sinc radius must be an integer >= 2")
    if window not in ("lanczos", "kaiser"):
        raise ValueError("window must be lanczos or kaiser")
    if window == "kaiser" and (not isinstance(beta, (int, float))
                               or not math.isfinite(beta) or beta <= 0):
        raise ValueError("Kaiser beta must be finite and positive")
    if window == "lanczos" and beta is not None:
        raise ValueError("Lanczos does not use a Kaiser beta")


def _reflect_indices(index, length):
    period = 2 * (length - 1)
    index = torch.remainder(index, period)
    return torch.where(index < length, index, period - index)


def _shift_axis(image, shift, axis, *, radius, window, beta):
    integer = torch.floor(shift)
    fraction = shift - integer
    taps = torch.arange(-radius + 1, radius + 1, device=image.device, dtype=image.dtype)
    z = fraction[:, None] - taps[None]
    if window == "lanczos":
        taper = torch.sinc(z / radius)
    else:
        # At a window endpoint, I0(beta*sqrt(v)) has a finite analytic limit,
        # but a literal sqrt(0) backward produces 0*inf.  The positive clamp
        # avoids that numerical singularity, including exact integer shifts.
        v = (1 - (z / radius).square()).clamp_min(torch.finfo(image.dtype).eps)
        beta_tensor = image.new_tensor(beta)
        taper = torch.special.i0(beta_tensor * torch.sqrt(v)) / torch.special.i0(beta_tensor)
    weights = torch.sinc(z) * taper * (z.abs() < radius)
    weights = weights / weights.sum(dim=1, keepdim=True)
    length = image.shape[axis]
    coordinates = torch.arange(length, device=image.device)[None] + integer.to(torch.int64)[:, None]
    result = torch.zeros_like(image)
    for i, offset in enumerate(range(-radius + 1, radius + 1)):
        indices = _reflect_indices(coordinates + offset, length)
        if axis == -1:
            gather = indices[:, None, None, :].expand_as(image)
        else:
            gather = indices[:, None, :, None].expand_as(image)
        result = result + image.gather(axis, gather) * weights[:, i, None, None, None]
    return result


def windowed_sinc_shift(image, delta, *, radius=4, window="lanczos", beta=None):
    """Separable 2*radius-tap interpolation with whole-sample reflection.

    Floor/index selection is piecewise constant; fractional coefficients carry
    shift gradients.  Kaiser derivatives at exact support changes need not be
    smooth.  Reflection itself is defined for arbitrary integer displacements.
    """
    d = _validate(image, delta)
    _kernel(radius, window, beta)
    result = _shift_axis(image, d[:, 0], -2, radius=radius, window=window, beta=beta)
    return _shift_axis(result, d[:, 1], -1, radius=radius, window=window, beta=beta)


def windowed_sinc_support_mask(height, width, delta, *, radius):
    """Conservative B,1,H,W mask: all finite taps read observed pixels.

    Zero-weight endpoint taps are still included.  This is not an FFT mask.
    """
    if type(height) is not int or type(width) is not int or min(height, width) < 2:
        raise ValueError("height and width must be integers >= 2")
    _kernel(radius, "lanczos", None)
    if (not isinstance(delta, torch.Tensor) or delta.ndim != 2 or delta.shape[1] != 2
            or delta.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(delta).all())):
        raise ValueError("finite floating B,2 delta required")
    with torch.no_grad():
        y = torch.arange(height, device=delta.device, dtype=delta.dtype)[None]
        x = torch.arange(width, device=delta.device, dtype=delta.dtype)[None]
        fy = torch.floor(y + delta[:, 0, None])
        fx = torch.floor(x + delta[:, 1, None])
        oky = (fy - radius + 1 >= 0) & (fy + radius < height)
        okx = (fx - radius + 1 >= 0) & (fx + radius < width)
    return (oky[:, :, None] & okx[:, None, :])[:, None]


def candidate_manifest(method, *, fft_padding=32):
    """Explicit research identity; not a campaign admission/publishing record."""
    if method not in METHODS:
        raise ValueError(f"unknown experimental method: {method}")
    common = {"schema": "PANDA_WARP_CANDIDATE_V1", "method": method,
              "experimental_only": True, "connected_to_training": False,
              "convention": "output[y,x]=input[y+dy,x+dx]", "delta_shape": "B,2",
              "image_shape": "B,C,H,W", "dtype": "preserve_float32_or_float64",
              "boundary": "whole_sample_reflection", "shift_autograd": True}
    if method == "fft":
        return {**common, "padding": _padding(fft_padding), "support": "global",
                "finite_observed_support_mask": False, "real_projection": True,
                "nyquist_information_loss_possible": True,
                "nyquist_policy": "joint_2d_phase_then_real_projection",
                "c4_equivariance_guaranteed": False,
                "pad_crop_roundtrip_guaranteed_lossless": False}
    radius = {"lanczos4": 4, "lanczos6": 6, "kaiser12_beta6": 6}[method]
    return {**common, "support": "finite", "radius": radius, "taps_per_axis": 2 * radius,
            "window": "kaiser" if method.startswith("kaiser") else "lanczos",
            "beta": 6.0 if method.startswith("kaiser") else None,
            "coefficient_normalization": "sum_to_one_per_axis"}


def warp_global(image, delta, *, method, fft_padding=32):
    """Apply a named candidate.  No production import or global monkeypatch."""
    config = candidate_manifest(method, fft_padding=fft_padding)
    if method == "fft":
        return fft_shift(image, delta, padding=config["padding"])
    return windowed_sinc_shift(image, delta, radius=config["radius"],
                               window=config["window"], beta=config["beta"])
