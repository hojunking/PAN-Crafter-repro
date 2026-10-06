"""Frozen, non-learning PAN registration in current PAN-pixel coordinates.

The output is a *sampling* displacement: W(P,c)[y,x] = P[y+dy,x+dx].
No registration threshold is tuned using reconstruction/test quality. A failed
quadratic fit is zero, never a silently accepted nearest-integer displacement.
"""
from __future__ import annotations

import math
import time
from collections import Counter

import cv2
import numpy as np
import torch

from align.resample import transform_delta


ESTIMATOR = {
    "schema": "TA2_OFFLINE_REGISTRATION_v1", "bound": 8, "margin": 16,
    "opencv_version": cv2.__version__, "numpy_version": np.__version__,
    "reference_support": "fixed 16px interior for every integer candidate",
    "grid": "all integer (dy,dx) in [-8,8]^2", "top_bands": 3,
    "aggregation": "componentwise median of three highest valid peak NCC bands",
    "lms_sigmas": [1.98, 0.7], "gt_sigmas": [0.8, 0.8],
    "gaussian_kernel": "2*ceil(3*sigma)+1", "padding": "reflect101",
    "gradient": "Scharr/32; independently mean-centered dx/dy NCC",
    "quadratic": "least-squares 3x3 2D, negative-definite Hessian, |offset|<=1",
    "target_rms_threshold": 1e-4,
    "fallback": "zero with reason; requires three valid bands",
    "physical_displacement_ground_truth": False,
}


def transform_shifts(shifts, rots, *, hflip=True, vflip=True):
    """Same hflip -> vflip -> CCW rotation as the native HR observations."""
    shifts = torch.as_tensor(shifts)
    if shifts.ndim != 2 or shifts.shape[1] != 2:
        raise ValueError("shifts must be Bx2 (dy,dx)")
    rotations = torch.as_tensor(rots, device=shifts.device, dtype=torch.long).reshape(-1)
    if len(rotations) != len(shifts) or bool(((rotations < 0) | (rotations > 3)).any()):
        raise ValueError("One rotation in {0,1,2,3} is required per shift")
    hf = torch.as_tensor(hflip, device=shifts.device).expand(len(shifts))
    vf = torch.as_tensor(vflip, device=shifts.device).expand(len(shifts))
    return transform_delta(shifts, hf, vf, rotations)


def _features(image, sigma, descriptor):
    image = np.asarray(image, dtype=np.float32)
    size = 2 * math.ceil(3 * sigma) + 1
    z = cv2.GaussianBlur(image, (size, size), sigma, borderType=cv2.BORDER_REFLECT_101)
    if descriptor == "intensity":
        return z
    if descriptor != "gradient":
        raise ValueError("descriptor must be gradient or intensity")
    return np.stack((cv2.Scharr(z, cv2.CV_32F, 1, 0, scale=1/32),
                     cv2.Scharr(z, cv2.CV_32F, 0, 1, scale=1/32)), axis=-1)


def _ncc(x, y):
    """NCC of intensity, or concatenated independently centered derivatives."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    axes = (0, 1)
    x, y = x-x.mean(axis=axes, keepdims=True), y-y.mean(axis=axes, keepdims=True)
    denominator = math.sqrt(float(np.square(x).sum()) * float(np.square(y).sum()))
    return None if denominator <= 1e-20 else float(np.sum(x*y) / denominator)


def quadratic_peak(score, iy, ix):
    """Validate a true local maximum; no clipping or integer fallback."""
    if iy in (0, score.shape[0]-1) or ix in (0, score.shape[1]-1):
        return None, "GRID_BOUNDARY"
    yy, xx = np.mgrid[-1:2, -1:2]
    design = np.stack((np.ones(9), yy.ravel(), xx.ravel(), yy.ravel()**2,
                       yy.ravel()*xx.ravel(), xx.ravel()**2), axis=1)
    coef = np.linalg.lstsq(design, score[iy-1:iy+2, ix-1:ix+2].ravel(), rcond=None)[0]
    hessian = np.array(((2*coef[3], coef[4]), (coef[4], 2*coef[5])))
    if not np.isfinite(hessian).all() or np.linalg.eigvalsh(hessian).max() >= -1e-12:
        return None, "NOT_LOCAL_MAXIMUM"
    offset = -np.linalg.solve(hessian, coef[1:3])
    if not np.isfinite(offset).all() or np.max(np.abs(offset)) > 1:
        return None, "QUADRATIC_OUTSIDE_NEIGHBORHOOD"
    return offset, None


def register_band(pan, reference, *, descriptor="gradient", sigma_p=1.98,
                  sigma_r=0.7, bound=8, margin=16, target_rms_threshold=1e-4):
    """Unit-intensity 2D arrays; output records even invalid evidence."""
    p, r = np.asarray(pan), np.asarray(reference)
    if p.ndim != 2 or p.shape != r.shape or min(p.shape) <= 2*margin:
        raise ValueError("PAN/reference must be same 2D shape with nonempty fixed support")
    if bound < 1 or margin < bound + math.ceil(3*max(sigma_p, sigma_r)) + 1:
        raise ValueError("Registration Gaussian/Scharr/source support is invalid")
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError("Nonfinite registration observation")
    a, b = _features(p, sigma_p, descriptor), _features(r, sigma_r, descriptor)
    template = np.ascontiguousarray(b[margin:-margin, margin:-margin])
    centered = template-template.mean(axis=(0, 1), keepdims=True)
    target_rms = float(np.sqrt(np.mean(np.square(centered, dtype=np.float64))))
    base = dict(shift=[0.0, 0.0], valid=False, peak=None, peak_gap=None,
                boundary=False, target_rms=target_rms, local_maximum=False)
    if target_rms < target_rms_threshold:
        return dict(base, fallback_reason="TARGET_LOW_TEXTURE")
    border = margin-bound
    search = np.ascontiguousarray(a[border:-border, border:-border])
    # CCOEFF centers each derivative channel independently. Both modes then
    # normalize the concatenated descriptor; gradient CCORR is deliberately not
    # used because it would omit the declared spatial mean removal.
    scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    if scores.shape != (2*bound+1, 2*bound+1) or not np.isfinite(scores).all():
        raise ValueError("Invalid registration score surface")
    iy, ix = np.unravel_index(int(scores.argmax()), scores.shape)
    peak = float(scores[iy, ix])
    alternatives = scores.copy()
    alternatives[max(0, iy-1):iy+2, max(0, ix-1):ix+2] = -np.inf
    gap = peak-float(alternatives.max())
    base.update(peak=peak, peak_gap=gap, boundary=iy in (0, 2*bound) or ix in (0, 2*bound))
    offset, reason = quadratic_peak(scores, iy, ix)
    if reason:
        return dict(base, fallback_reason=reason)
    candidate = a[margin+iy-bound:p.shape[0]-margin+iy-bound,
                  margin+ix-bound:p.shape[1]-margin+ix-bound]
    if _ncc(candidate, template) is None:
        return dict(base, fallback_reason="SOURCE_LOW_TEXTURE")
    base.update(shift=(np.array((iy-bound, ix-bound))+offset).tolist(),
                valid=True, local_maximum=True, fallback_reason=None)
    return base


def estimate_shift(pan, reference, *, descriptor="gradient", target="LMS",
                   max_dn=1.0, split=None, oracle=False):
    """Estimate one image independently, with explicit GT train/oracle gating."""
    start = time.perf_counter()
    target = target.upper()
    if target not in {"LMS", "NATIVE_LMS", "BICUBIC_MS", "GT"}:
        raise ValueError("Unregistered registration target")
    if target == "GT" and split != "train" and not (oracle and split in {"val", "rr"}):
        raise ValueError("GT_REGISTRATION_FORBIDDEN: train supervision or explicit val/RR diagnostic oracle only")
    if max_dn <= 0:
        raise ValueError("max_dn must be positive")
    p = np.asarray(pan, dtype=np.float64)
    if p.ndim == 3 and p.shape[0] == 1:
        p = p[0]
    ref = np.asarray(reference, dtype=np.float64)
    if ref.ndim != 3 or ref.shape[1:] != p.shape:
        raise ValueError("reference must be CxHxW matching PAN")
    sp, sr = (0.8, 0.8) if target == "GT" else (1.98, 0.7)
    bands = [register_band(p/max_dn, band/max_dn, descriptor=descriptor,
                           sigma_p=sp, sigma_r=sr) for band in ref]
    valid = [i for i, b in enumerate(bands) if b["valid"]]
    chosen = sorted(valid, key=lambda i: (-bands[i]["peak"], i))[:3]
    okay = len(chosen) == 3
    shifts = np.array([bands[i]["shift"] for i in chosen])
    shift = np.median(shifts, axis=0) if okay else np.zeros(2)
    return dict(shift=shift.tolist(), valid=okay,
                fallback_reason=None if okay else "INSUFFICIENT_VALID_BANDS",
                band_failure_counts=dict(Counter(b["fallback_reason"] for b in bands if not b["valid"])),
                selected_bands=chosen, bands=bands,
                confidence=float(np.mean([bands[i]["peak"] for i in chosen])) if okay else None,
                spread=float(np.median(np.linalg.norm(shifts-shift, axis=1))) if okay else None,
                boundary_hit_count=sum(b["boundary"] for b in bands),
                descriptor=descriptor, target=target,
                record_type=("ORACLE_RR_ONLY" if split == "rr" else "ORACLE_VALIDATION_ONLY")
                    if target == "GT" and oracle and split != "train" else "INPUT_ONLY" if target != "GT" else "TRAIN_GT_PROXY",
                registration_seconds=time.perf_counter()-start)


def fixed_band_weights(pan, lms, *, max_dn=1.0, scales=(0.8, 1.6)):
    """CAL-only zero-shift |NCC|, median over images, separately per scale."""
    p, l = np.asarray(pan), np.asarray(lms)
    if p.ndim != 4 or l.ndim != 4 or p.shape[0] != l.shape[0] or p.shape[1] != 1:
        raise ValueError("CAL observations must be Nx1xHxW and NxCxHxW")
    measurements = np.full((len(scales), len(p), l.shape[1]), np.nan)
    for s, sigma in enumerate(scales):
        for i in range(len(p)):
            pf = _features(p[i, 0]/max_dn, sigma, "gradient")[16:-16, 16:-16]
            for b in range(l.shape[1]):
                rf = _features(l[i, b]/max_dn, sigma, "gradient")[16:-16, 16:-16]
                if float(np.sqrt(np.square(rf).mean())) >= 1e-4:
                    corr = _ncc(pf, rf)
                    if corr is not None:
                        measurements[s, i, b] = abs(corr)
    reliability = np.zeros((len(scales), l.shape[1]))
    for s in range(len(scales)):
        for b in range(l.shape[1]):
            valid = measurements[s, :, b][np.isfinite(measurements[s, :, b])]
            if len(valid):
                reliability[s, b] = np.median(valid)
    if bool((reliability.sum(axis=1) <= 0).any()):
        raise ValueError("CAL_LOW_TEXTURE: no valid frozen band reliability")
    weights = reliability/reliability.sum(axis=1, keepdims=True)
    return dict(weights=weights.tolist(), median_absolute_correlation=reliability.tolist(),
                valid_counts=np.isfinite(measurements).sum(axis=1).tolist(), scales=list(scales),
                target="NATIVE_LMS", split="train", uses_gt=False)
