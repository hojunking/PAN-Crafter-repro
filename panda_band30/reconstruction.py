"""Truthful 9/16-channel P0 stem with explicit post-augmentation MS base."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from directshift48.method import _digest
from panda_method.reconstruction import (ReconstructionSpec, Reconstructor,
                                         build_reconstructor as _build_original)
from panda_warp30.reconstruction import base_from_batch, upsampler_manifest


@dataclass(frozen=True)
class BandReconstructionSpec(ReconstructionSpec):
    pan_channels: int = 1

    def __post_init__(self):
        super().__post_init__()
        if (self.bands != 8 or self.pan_channels not in (1, 8) or type(self.pan_channels) is not int
                or self.input_layout != "P0" or self.base_reference != "BICUBIC_MS"):
            raise ValueError("WV3 P0 explicit single/eight PAN input specification required")

    @property
    def input_channels(self):
        return self.bands + self.pan_channels

    def manifest(self):
        result = super().manifest()
        result.update(schema="PANDA_BAND30_RECONSTRUCTION_V1", pan_channels=self.pan_channels,
                      input_channels=self.input_channels,
                      input_order=[f"PAN_{k}" for k in range(self.pan_channels)] +
                                  [f"MS_{k}" for k in range(8)],
                      stem_initialization="original_PAN_weight_split_equally_over_PAN_channels",
                      base_reference_is_constructor_compatibility_only=True)
        return result


class BandReconstructor(Reconstructor):
    def __init__(self, backbone, spec):
        if not isinstance(spec, BandReconstructionSpec):
            raise TypeError("Campaign-local truthful 9/16-channel specification required")
        super().__init__(backbone, spec)
        self.pan_channels = spec.pan_channels
        self.register_buffer("_band30_identity", self._identity())

    def _identity(self):
        return torch.tensor(list(bytes.fromhex(_digest(self.spec.manifest()))), dtype=torch.uint8)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        expected, actual = self._identity(), state_dict.get(prefix + "_band30_identity")
        if (not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype
                or actual.shape != expected.shape or not torch.equal(actual.cpu(), expected)):
            raise RuntimeError("Band30 reconstruction stem/spec checkpoint mismatch")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, *args, **kwargs):
        raise RuntimeError("Band30 requires reconstruct_with_base: no implicit bicubic fallback")


def build_reconstructor(spec, *, seed, role="T", pan_channels=1):
    if isinstance(spec, BandReconstructionSpec):
        if pan_channels not in (1, spec.pan_channels):
            raise ValueError("Conflicting PAN-channel arguments")
        pan_channels = spec.pan_channels
    actual = BandReconstructionSpec(spec.bands, spec.width, spec.depth,
                                   spec.input_layout, spec.base_reference, pan_channels)
    original_spec = ReconstructionSpec(spec.bands, spec.width, spec.depth,
                                       spec.input_layout, spec.base_reference)
    fresh = _build_original(original_spec, seed=seed, role=role)
    if pan_channels == 8:
        original = fresh.backbone.input
        with torch.random.fork_rng(devices=[]):
            stem = nn.Conv2d(16, spec.width, 3, padding=1)
        with torch.no_grad():
            stem.weight[:, :8].copy_(original.weight[:, :1].expand(-1, 8, -1, -1) / 8)
            stem.weight[:, 8:].copy_(original.weight[:, 1:])
            stem.bias.copy_(original.bias)
        fresh.backbone.input = stem
    return BandReconstructor(fresh.backbone, actual)


def reconstruct_with_base(reconstructor, pan, base, ms):
    if not isinstance(reconstructor, BandReconstructor):
        raise TypeError("Campaign-local BandReconstructor required")
    if pan.ndim != 4 or len(pan) == 0:
        raise ValueError("Nonempty BCHW PAN views required")
    n, c, h, w = pan.shape
    if (c != reconstructor.pan_channels or h % 4 or w % 4
            or base.shape != (n, 8, h, w) or ms.shape != (n, 8, h//4, w//4)):
        raise ValueError("Declared PAN channels and exact MS/base grids required")
    if any(t.dtype != torch.float32 or t.device != pan.device or not bool(torch.isfinite(t).all())
           for t in (pan, base, ms)):
        raise ValueError("Finite FP32 reconstruction observations on one device required")
    with torch.autocast(device_type=pan.device.type, enabled=False):
        x_in = torch.cat((pan, base), dim=1)
        residual = reconstructor.backbone(pan[:, :1], None, ms, pan.new_ones(n), x_in=x_in)
        y = base + residual.float()
    if not bool(torch.isfinite(y).all()):
        raise FloatingPointError("Nonfinite band30 reconstruction")
    return {"y": y, "ms_base": base, "x_in": x_in, "pan_aligned": pan, "lpan_aligned": None}
