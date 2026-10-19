"""Phase-aware x4 LRMS interpolation after online flips/rotations.

This reuses the repository's Torch DLPan interp23tap port, including its
circular boundary convention. It is an upsampler, NOT a continuous-shift warp;
PAN/GT/output warps remain separate operations. DLPan provenance/licensing is
unchanged by this wrapper; this is not a clean-license deployment export.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from align.resample import interp23tap


UPSAMPLER_VERSION = "DLPAN_INTERP23TAP_X4_POST_AUG_PHASE_CORRECTED_V1"


def _phase_roll(augmentation):
    """Integer HR correction R v - v, with phase offset v=(.5,.5).

    At x4, interp23tap places LR sample j at HR 4*j+2, whereas the flip center
    has phase 1.5. Thus naive U(aug(LR)) differs from aug(U(LR)). Transforming
    twice the phase vector avoids floating-point/rounding choices altogether.
    """
    hflip, vflip, rot90 = augmentation
    dy, dx = 1, 1
    if hflip:
        dx = -dx
    if vflip:
        dy = -dy
    for _ in range(rot90):
        dy, dx = -dx, dy
    return (dy - 1) // 2, (dx - 1) // 2


def upsample_lrms_after_augmentation(ms, augmentations):
    """Interpolate already-augmented LRMS, preserving the paired HR grid.

    Args:
        ms: Finite FP32/FP64 NCHW LRMS tensor. Rectangular images are supported;
            both LR sides must be >=6 for the existing 23-tap circular pad.
        augmentations: One (hflip: bool, vflip: bool, CCW rot90: int in 0..3)
            triple per sample, in the SAME Hflip -> Vflip -> rotation order
            already applied to LRMS and the paired PAN/LMS/GT observations.

    Returns:
        N,C,4H,4W tensor on the same dtype/device, with input gradients intact.
        The integer phase roll introduces no additional interpolation. This
        reproduces augment(interp23tap(original LRMS)); it does not claim that
        every dataset's stored LMS was generated from its currently active MS.

    No input is augmented a second time, no shift is inferred, and no image
    values are clipped. Inference uses (False, False, 0) for every sample.
    """
    if not isinstance(ms, torch.Tensor):
        raise TypeError("LRMS must be a torch Tensor")
    if ms.ndim != 4 or ms.shape[0] == 0 or ms.shape[1] == 0:
        raise ValueError("Nonempty NCHW LRMS required")
    if min(ms.shape[-2:]) < 6:
        raise ValueError("LRMS sides >=6 required by x4 interp23tap circular padding")
    if ms.dtype not in (torch.float32, torch.float64):
        raise ValueError("LRMS interpolation requires FP32 or FP64")
    if not bool(torch.isfinite(ms).all()):
        raise ValueError("Finite LRMS required")
    if (not isinstance(augmentations, Sequence)
            or isinstance(augmentations, (str, bytes))
            or len(augmentations) != len(ms)):
        raise ValueError("Explicit per-sample augmentation metadata required")
    rolls = []
    for augmentation in augmentations:
        if (not isinstance(augmentation, (tuple, list)) or len(augmentation) != 3
                or type(augmentation[0]) is not bool
                or type(augmentation[1]) is not bool
                or type(augmentation[2]) is not int
                or augmentation[2] not in range(4)):
            raise ValueError("Augmentation must be (bool Hflip, bool Vflip, integer CCW rotation 0..3)")
        rolls.append(_phase_roll(augmentation))
    with torch.autocast(device_type=ms.device.type, enabled=False):
        up = interp23tap(ms, ratio=4)
        if all(roll == rolls[0] for roll in rolls):
            result = torch.roll(up, shifts=rolls[0], dims=(-2, -1))
        else:
            result = torch.cat([
                torch.roll(up[i:i + 1], shifts=roll, dims=(-2, -1))
                for i, roll in enumerate(rolls)
            ], dim=0)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("Nonfinite interp23tap LRMS output")
    return result
