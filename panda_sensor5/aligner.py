"""Four-band Aligner with a single global translation and mean4-GT target."""
from __future__ import annotations

import torch

from directshift48.method import DirectPANAligner, ScharrAbsolute
from pa.warp import warp_pan, warp_support_mask


GLOBAL_TARGET = "SCHARR_OF_MEAN4_GT"


class BandPANAligner(DirectPANAligner):
    """Four-band input, one global dy/dx output; no bandwise branch."""
    CHECKPOINT_REVISION = 5026

    def __init__(self, outputs=2):
        if type(outputs) is not int or outputs != 2:
            raise ValueError("Sensor5 requires one global2 head")
        self.outputs = outputs
        super().__init__(4, view_margin=8)

    def manifest(self):
        result = super().manifest()
        result.update(method="PANDA_SENSOR5_ALL4_INPUT_V1", head_dims=[64, 32, 2],
                      output="total_dy_dx_current_PAN_px")
        return result


def aligned_pan_views(pan, shifts):
    if pan.ndim != 4 or pan.shape[1] != 1 or shifts.shape != (len(pan), 2):
        raise ValueError("Original PAN and one global dy/dx per sample required")
    return warp_pan(pan, shifts)


def support_for_shifts(height, width, shifts, *, intersection=True):
    if shifts.ndim != 2 or shifts.shape[1] != 2:
        raise ValueError("Global2 support shifts required")
    return warp_support_mask(height, width, shifts.detach())


class AllBandScharr(ScharrAbsolute):
    """Stage1 target is Scharr(mean4 GT), including NIR."""
    def __init__(self, outputs, *, sensor):
        super().__init__(sensor)
        if self.bands != 4:
            raise ValueError("Four-band sensor required")
        if outputs != 2:
            raise ValueError("Explicit global2 objective required")
        self.outputs = outputs

    def absolute_loss(self, aligned_pan, gt, *, roi_margin=8):
        channels = 1
        self._check(aligned_pan, channels)
        self._check(gt, 4)
        if (len(aligned_pan) != len(gt) or aligned_pan.shape[-2:] != gt.shape[-2:]
                or aligned_pan.dtype != gt.dtype or aligned_pan.device != gt.device
                or type(roi_margin) is not int or roi_margin != 8
                or min(gt.shape[-2:]) <= 2*roi_margin):
            raise ValueError("Matched PAN/GT and fixed ROI8 required")
        p, target = aligned_pan, gt.mean(1, keepdim=True)
        delta = self.descriptor(p) - self.descriptor(target)
        return delta[..., roi_margin:-roi_margin, roi_margin:-roi_margin].abs().mean()

    def manifest(self):
        result = super().manifest()
        result.pop("rgb_indices_R_G_B")
        result.pop("rgb_mean_before_gradient")
        result.update(target=GLOBAL_TARGET,
                      target_bands=list(range(4)), mean4_before_gradient=True,
                      reduction="mean_samples_two_axes_fixed_ROI8")
        return result
