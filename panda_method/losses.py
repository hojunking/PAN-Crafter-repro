"""MethodPlan-v1.3 spatial losses (unchanged from v1.2) and detached reliability maps.

The discrete gradient, radiometric units, RGB indices and fixed spatial ROI
are explicit inputs, not experimental decisions hidden in a default.  Fitting
uses all pixels.  Spatial terms use the declared ROI, with its pixel count as
the denominator (never the sum of reliability weights).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SpatialLossSpec:
    sensor: str
    rgb_indices: tuple[int, int, int]
    operator: str
    padding: str
    roi_margin: int
    radiometry: str

    def __post_init__(self) -> None:
        if not isinstance(self.sensor, str) or not self.sensor.strip():
            raise ValueError("sensor must be explicitly identified")
        if not isinstance(self.rgb_indices, tuple) or len(self.rgb_indices) != 3:
            raise ValueError("rgb_indices must be an immutable (R,G,B) tuple")
        if any(type(i) is not int or i < 0 for i in self.rgb_indices):
            raise ValueError("RGB indices must be nonnegative integers")
        if len(set(self.rgb_indices)) != 3:
            raise ValueError("RGB indices must select three distinct bands")
        if self.operator not in {"central_difference", "sobel_1_8", "forward_difference"}:
            raise ValueError("unsupported fixed gradient operator")
        if self.padding not in {"reflect", "replicate", "constant_zero"}:
            raise ValueError("gradient padding must be explicitly declared")
        if type(self.roi_margin) is not int or self.roi_margin < 0:
            raise ValueError("roi_margin must be an explicit nonnegative integer")
        if not isinstance(self.radiometry, str) or not self.radiometry.strip():
            raise ValueError("radiometric normalization must be explicitly declared")

    def fingerprint(self) -> str:
        payload = {"version": "panda_method_spatial_v1", **asdict(self)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()


def sipsa_spatial_spec(*, sensor: str, rgb_indices: tuple[int, int, int],
                      roi_margin: int, radiometry: str) -> SpatialLossSpec:
    """Canonical descriptor, independently implemented from SIPSA's operator.

    Official https://github.com/brachiohyup/SIPSA ``sobelfilter2XY`` uses
    [[0,0,0],[0,1,0],[0,-1,0]] and its transpose, with REFLECT padding one.
    Despite its name it is a first difference, NOT the usual Sobel stencil.
    Our forward-difference kernels have the opposite sign; the per-component
    absolute value makes the descriptor exactly equivalent.  Main training
    compares those absolute components using a mean L1.

    The selected dataset's RGB indices, shared radiometric units and fixed
    spatial ROI remain explicit decisions; this does not assert that PANDA's
    reference frame, targets, or whole training objective reproduce SIPSA.
    """
    return SpatialLossSpec(sensor=sensor, rgb_indices=rgb_indices,
                           operator="forward_difference", padding="reflect",
                           roi_margin=roi_margin, radiometry=radiometry)


def _image(x: Tensor, name: str, *, channels: int | None = None) -> Tensor:
    if not isinstance(x, Tensor) or x.ndim != 4 or not x.is_floating_point():
        raise ValueError(f"{name} must be a floating-point [B,C,H,W] tensor")
    if any(n <= 0 for n in x.shape) or (channels is not None and x.shape[1] != channels):
        raise ValueError(f"invalid {name} shape: {tuple(x.shape)}")
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f"{name} contains nonfinite values")
    # Loss arithmetic is FP32 minimum, independent of mixed-precision training.
    return x if x.dtype == torch.float64 else x.float()


def _matching(a: Tensor, b: Tensor, name: str) -> None:
    if a.shape != b.shape or a.device != b.device:
        raise ValueError(f"{name} tensors must have identical shapes and devices")


def teacher_error(teacher: Tensor, gt: Tensor) -> Tensor:
    """Detached all-band Teacher-vs-GT absolute error, [B,1,H,W]."""
    t = _image(teacher.detach(), "teacher")
    y = _image(gt.detach(), "gt")
    _matching(t, y, "Teacher/GT")
    return _image((t - y).abs().mean(dim=1, keepdim=True), "Teacher error", channels=1)


def reliability_weight(error: Tensor, tau: float | Tensor) -> Tensor:
    """Detached x/(x+tau), with a dtype-tiny denominator floor.

    The floor is ``torch.finfo(computation_dtype).tiny`` (smallest positive
    normal number), not a tunable loss parameter.  Half/bfloat16 inputs are
    promoted to FP32; FP64 remains FP64.  The explicitly defined 0/0 case is
    zero.  A zero median otherwise gives weight one to positive errors.
    """
    x = _image(error.detach(), "error", channels=1)
    if bool((x < 0).any()):
        raise ValueError("reliability errors must be nonnegative")
    scale = torch.as_tensor(tau, dtype=x.dtype, device=x.device).detach()
    if scale.numel() != 1 or not bool(torch.isfinite(scale).all()) or bool((scale < 0).any()):
        raise ValueError("tau must be a finite nonnegative scalar")
    denominator = x + scale.reshape(())
    if not bool(torch.isfinite(denominator).all()):
        raise ValueError("reliability denominator overflowed")
    return x / denominator.clamp_min(torch.finfo(x.dtype).tiny)


class AbsoluteGradientLoss(nn.Module):
    """Per-axis absolute gradient discrepancy with explicitly fixed semantics.

    Kernels use cross-correlation as in ``conv2d``.  Component order is (dy,dx).
    The central difference is [-1,0,1]/2; Sobel is normalized by 1/8;
    forward difference is center-to-next-pixel [-1,1] (no division).
    These choices implement the stated family, not an assertion that every
    operator is numerically identical to SIPSA's official implementation.
    """

    def __init__(self, spec: SpatialLossSpec):
        super().__init__()
        if not isinstance(spec, SpatialLossSpec):
            raise TypeError("an explicit SpatialLossSpec is required")
        self.spec = spec
        if spec.operator == "central_difference":
            ky = [[0., -.5, 0.], [0., 0., 0.], [0., .5, 0.]]
        elif spec.operator == "sobel_1_8":
            ky = [[-1./8, -2./8, -1./8], [0., 0., 0.], [1./8, 2./8, 1./8]]
        else:
            ky = [[0., 0., 0.], [0., -1., 0.], [0., 1., 0.]]
        kernel_y = torch.tensor(ky, dtype=torch.float64)
        kernels = torch.stack((kernel_y, kernel_y.T)).unsqueeze(1)
        # Derived from immutable configuration: no trainable or checkpoint-
        # overrideable derivative coefficients.
        self.register_buffer("_kernels", kernels, persistent=False)

    def rgb_luminance(self, ms: Tensor) -> Tensor:
        x = _image(ms, "MS")
        if max(self.spec.rgb_indices) >= x.shape[1]:
            raise ValueError("RGB band index exceeds the supplied MS channel count")
        return x[:, self.spec.rgb_indices].mean(dim=1, keepdim=True)

    def descriptor(self, image: Tensor) -> Tensor:
        x = _image(image, "gradient image", channels=1)
        if self.spec.padding == "reflect" and min(x.shape[-2:]) < 2:
            raise ValueError("reflect gradient padding needs H,W >= 2")
        mode = "constant" if self.spec.padding == "constant_zero" else self.spec.padding
        # An outer autocast context must not silently change the operator's
        # dtype or the calibrated q scale.
        with torch.autocast(device_type=x.device.type, enabled=False):
            padded = F.pad(x, (1, 1, 1, 1), mode=mode)
            return F.conv2d(padded, self._kernels.to(device=x.device, dtype=x.dtype)).abs()

    def discrepancy(self, first: Tensor, second: Tensor) -> Tensor:
        _matching(first, second, "gradient reference")
        return (self.descriptor(first) - self.descriptor(second)).abs().mean(1, keepdim=True)

    def pixel_mask(self, values: Tensor) -> Tensor:
        x = _image(values, "spatial map", channels=1)
        margin = self.spec.roi_margin
        h, w = x.shape[-2:]
        if 2 * margin >= min(h, w):
            raise ValueError("declared fixed spatial ROI is empty")
        mask = torch.zeros((1, 1, h, w), dtype=torch.bool, device=x.device)
        mask[..., margin:h-margin, margin:w-margin] = True
        return mask.expand(x.shape[0], 1, h, w)

    def reduce(self, values: Tensor) -> Tensor:
        x = _image(values, "spatial map", channels=1)
        return x[self.pixel_mask(x)].mean()

    def stage1(self, aligned_pan: Tensor, gt: Tensor) -> Tensor:
        """Native corrected PAN spatial loss; GT is a detached reference."""
        return self.reduce(self.discrepancy(aligned_pan, self.rgb_luminance(gt.detach())))

    def reference_maps(self, teacher: Tensor, gt: Tensor, aligned_pan: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            e = teacher_error(teacher, gt)
            q = self.discrepancy(aligned_pan.detach(), self.rgb_luminance(gt.detach()))
        _matching(e, q, "e/q")
        return e, q

    def stage2(self, student: Tensor, teacher: Tensor, gt: Tensor, aligned_pan: Tensor,
               tau_e: float | Tensor, tau_q: float | Tensor) -> dict[str, Tensor]:
        """Exactly L_fit + L_spatial (lambda_S=1), only Student gradients."""
        s = _image(student, "student")
        t, y = _image(teacher.detach(), "teacher"), _image(gt.detach(), "gt")
        _matching(s, t, "Student/Teacher")
        _matching(s, y, "Student/GT")
        e, q = self.reference_maps(t, y, aligned_pan)
        w_e, w_q = reliability_weight(e, tau_e), reliability_weight(q, tau_q)
        soft = (s - t).abs().mean(1, keepdim=True)
        hard = (s - y).abs().mean(1, keepdim=True)
        fit = ((1 - w_e) * soft + w_e * hard).mean()
        edge = self.discrepancy(self.rgb_luminance(s), self.rgb_luminance(y))
        spatial = self.reduce(w_q * edge)
        return {"loss": fit + spatial, "fit": fit, "spatial": spatial,
                "e": e, "q": q, "w_e": w_e, "w_q": w_q,
                "soft_px": soft, "hard_px": hard, "edge_px": edge}
