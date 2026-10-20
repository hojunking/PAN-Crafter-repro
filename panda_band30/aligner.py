"""Campaign-local all-band global or eight matched PAN translations."""
from __future__ import annotations

import torch
from torch import nn

from directshift48.method import DirectPANAligner, ScharrAbsolute
from pa.warp import warp_pan, warp_support_mask


GLOBAL_TARGET = "SCHARR_OF_MEAN8_GT"
BAND_TARGET = "MEAN8_MATCHED_BAND_SCHARR_LOSSES"


class BandPANAligner(DirectPANAligner):
    CHECKPOINT_REVISION = 3019

    def __init__(self, outputs=2):
        if type(outputs) is not int or outputs not in (2, 16):
            raise ValueError("Explicit global2 or band16 head required")
        self.outputs = outputs
        super().__init__(8, view_margin=8)
        if outputs == 16:
            self.fc2 = nn.Linear(32, 16)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def manifest(self):
        result = super().manifest()
        result.update(method="PANDA_BAND30_ALL8_INPUT_V1", head_dims=[64, 32, self.outputs],
                      output=("total_dy_dx_current_PAN_px" if self.outputs == 2 else
                              "B_8_2_band_order_0_to_7_each_dy_dx_current_PAN_px"))
        return result

    def forward(self, pan, native_lms, *, return_details=False):
        value = super().forward(pan, native_lms, return_details=return_details)
        u, details = value if return_details else (value, None)
        if self.outputs == 16:
            u = u.reshape(len(pan), 8, 2)
        return (u, details) if return_details else u


def bandwise_warp(image, shifts):
    """Independent channel warps; W(X,u)[y,x]=X[y+dy,x+dx]."""
    if (image.ndim != 4 or len(image) == 0 or shifts.shape != (*image.shape[:2], 2)
            or image.dtype != torch.float32 or shifts.dtype != torch.float32
            or image.device != shifts.device or not bool(torch.isfinite(image).all())
            or not bool(torch.isfinite(shifts).all())):
        raise ValueError("Finite FP32 BCHW image and BC2 shifts on one device required")
    b, c, h, w = image.shape
    return warp_pan(image.reshape(b*c, 1, h, w), shifts.reshape(b*c, 2)).reshape(b, c, h, w)


def aligned_pan_views(pan, shifts):
    if pan.ndim != 4 or pan.shape[1] != 1:
        raise ValueError("Original single-channel PAN required")
    if shifts.shape == (len(pan), 2):
        return warp_pan(pan, shifts)
    if shifts.shape != (len(pan), 8, 2):
        raise ValueError("Explicit B2 or B82 PAN correction required")
    return bandwise_warp(pan.expand(-1, 8, -1, -1), shifts)


def support_for_shifts(height, width, shifts, *, intersection=True):
    if shifts.ndim == 2 and shifts.shape[1] == 2:
        return warp_support_mask(height, width, shifts.detach())
    if shifts.ndim != 3 or shifts.shape[1:] != (8, 2):
        raise ValueError("Explicit B2 or B82 support shifts required")
    support = warp_support_mask(height, width, shifts.detach().reshape(-1, 2))
    support = support.reshape(len(shifts), 8, height, width)
    return support.all(dim=1, keepdim=True) if intersection else support


class AllBandScharr(ScharrAbsolute):
    """Global: Scharr(mean8); bandwise: mean8 of matched Scharr losses."""
    def __init__(self, outputs):
        super().__init__("WV3")
        if outputs not in (2, 16):
            raise ValueError("Explicit global2 or band16 objective required")
        self.outputs = outputs

    def absolute_loss(self, aligned_pan, gt, *, roi_margin=8):
        channels = 1 if self.outputs == 2 else 8
        self._check(aligned_pan, channels)
        self._check(gt, 8)
        if (len(aligned_pan) != len(gt) or aligned_pan.shape[-2:] != gt.shape[-2:]
                or aligned_pan.dtype != gt.dtype or aligned_pan.device != gt.device
                or type(roi_margin) is not int or roi_margin != 8
                or min(gt.shape[-2:]) <= 2*roi_margin):
            raise ValueError("Matched PAN/GT and fixed ROI8 required")
        if self.outputs == 2:
            p, target = aligned_pan, gt.mean(1, keepdim=True)
        else:
            h, w = gt.shape[-2:]
            p, target = aligned_pan.reshape(-1, 1, h, w), gt.reshape(-1, 1, h, w)
        delta = self.descriptor(p) - self.descriptor(target)
        return delta[..., roi_margin:-roi_margin, roi_margin:-roi_margin].abs().mean()

    def manifest(self):
        result = super().manifest()
        result.pop("rgb_indices_R_G_B")
        result.pop("rgb_mean_before_gradient")
        result.update(target=GLOBAL_TARGET if self.outputs == 2 else BAND_TARGET,
                      target_bands=list(range(8)), mean8_before_gradient=self.outputs == 2,
                      reduction="mean_samples_bands_if_bandwise_two_axes_fixed_ROI8")
        return result
