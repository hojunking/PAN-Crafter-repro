"""Explicit effective LRMS base for the isolated warp30 A/T and A/S paths.

The backbone's historical ReconstructionSpec remains BICUBIC_MS for strict
architecture compatibility. This adapter, and its explicit manifest, own the
effective interpolation; callers must not invoke Reconstructor.forward for an
interp23-trained network, since that would silently reconstruct a bicubic base.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from panda_method.edge_data import EdgeObservationBatch
from panda_method.reconstruction import Reconstructor
from panda_midpoint.resampling import UPSAMPLER_VERSION, upsample_lrms_after_augmentation


UPSAMPLERS = ("bicubic", "interp23")


def upsampler_manifest(upsampler):
    if upsampler not in UPSAMPLERS:
        raise ValueError("Explicit bicubic or interp23 LRMS upsampler required")
    return {"kind": upsampler, "scale": 4,
            "implementation": ("torch_interpolate_bicubic_align_corners_false" if upsampler == "bicubic"
                               else UPSAMPLER_VERSION),
            "augmentation_order": "LRMS_Hflip_Vflip_CCW_rotation_then_upsampling",
            "augmentation_phase": ("center_symmetric_phase1.5" if upsampler == "bicubic"
                                   else "phase2_corrected_by_exact_integer_Rv_minus_v_roll"),
            "base_usage": "same_explicit_base_for_U_input_and_residual_skip",
            "actual_shift_warp": "bicubic_border_align_corners_false_FP32",
            "aligner_reference": "PROVIDED_STORED_LMS_UNCHANGED"}


def base_from_batch(batch, upsampler):
    """Upsample the already-augmented batch.ms without replacing stored LMS."""
    upsampler_manifest(upsampler)
    if not isinstance(batch, EdgeObservationBatch):
        raise TypeError("Typed PAN/MS/stored-LMS observation batch required")
    batch.validate(batch.ms.shape[1])
    if batch.ms.dtype != torch.float32:
        raise ValueError("Warp30 model paths require FP32 observations")
    with torch.autocast(device_type=batch.ms.device.type, enabled=False):
        if upsampler == "interp23":
            return upsample_lrms_after_augmentation(batch.ms, batch.lineage.augmentations)
        return F.interpolate(batch.ms, scale_factor=4, mode="bicubic", align_corners=False)


def reconstruct_with_base(reconstructor, pan, base, ms):
    """Run the unchanged U-Net with one caller-provided input/residual base."""
    if (not isinstance(reconstructor, Reconstructor)
            or reconstructor.spec.input_layout != "P0"
            or reconstructor.spec.base_reference != "BICUBIC_MS"):
        raise ValueError("Existing P0 Reconstructor architecture descriptor required")
    bands = reconstructor.spec.bands
    if pan.ndim != 4 or len(pan) == 0:
        raise ValueError("Nonempty N1HW PAN required")
    n, _, h, w = pan.shape
    if (pan.shape != (n, 1, h, w) or h % 4 or w % 4
            or base.shape != (n, bands, h, w) or ms.shape != (n, bands, h // 4, w // 4)):
        raise ValueError("Explicit PAN/MS/reconstruction-base grid correspondence required")
    if any(t.dtype != torch.float32 or t.device != pan.device or not bool(torch.isfinite(t).all())
           for t in (pan, base, ms)):
        raise ValueError("Finite FP32 reconstruction inputs on one device required")
    with torch.autocast(device_type=pan.device.type, enabled=False):
        x_in = torch.cat((pan, base), dim=1)
        residual = reconstructor.backbone(pan, None, ms, pan.new_ones(n), x_in=x_in)
        prediction = base + residual.float()
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError("Nonfinite warp30 reconstruction")
    return {"y": prediction, "ms_base": base, "x_in": x_in,
            "pan_aligned": pan, "lpan_aligned": None}
