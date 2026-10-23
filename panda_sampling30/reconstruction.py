"""Isolated Student sampling ablations for the existing three-scale U-Net.

The frozen Teacher remains byte-for-byte compatible with the existing builder.
These sampling changes do not enforce rotation equivariance: learned convolution
kernels remain unconstrained. PAN warping and reconstruction loss are untouched.
"""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from directshift48.method import _digest
from fh12.model import _construction_rng, _named_fresh
from model.pancrafter import DownConv, UpConv
from panda_band30.reconstruction import (
    BandReconstructionSpec, BandReconstructor,
    build_reconstructor as _build_original,
    base_from_batch, reconstruct_with_base, upsampler_manifest,
)


SAMPLING_VARIANTS = ("BASE", "DAVG", "UBIL", "DAVG_UBIL")
SAMPLING_SCHEMA = "PANDA_SAMPLING30_RECONSTRUCTION_V1"


class AverageDownConv(nn.Module):
    """2x2 average pooling, then a learned 3x3 stride-one convolution."""

    def __init__(self, channels, out_channels):
        super().__init__()
        self.channels, self.out_channels = channels, out_channels
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0,
                                 ceil_mode=False, count_include_pad=False)
        self.conv = nn.Conv2d(channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(self.pool(x))


class BilinearUpConv(nn.Module):
    """Bilinear resize by two, then a learned 3x3 stride-one convolution."""

    def __init__(self, channels, out_channels):
        super().__init__()
        self.channels, self.out_channels = channels, out_channels
        self.conv = nn.Conv2d(channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="bilinear",
                                       align_corners=False))


def operator_manifest(sampling_variant):
    """JSON-only architecture description; no equivariance claim is implied."""
    if sampling_variant not in SAMPLING_VARIANTS:
        raise ValueError("Registered BASE/DAVG/UBIL/DAVG_UBIL sampling variant required")
    down = ("AVGPOOL_2X2_STRIDE2_THEN_CONV_3X3_STRIDE1_PAD1"
            if sampling_variant in ("DAVG", "DAVG_UBIL")
            else "CONV_3X3_STRIDE2_PAD1")
    up = ("BILINEAR_X2_ALIGN_CORNERS_FALSE_THEN_CONV_3X3_STRIDE1_PAD1"
          if sampling_variant in ("UBIL", "DAVG_UBIL")
          else "CONV_TRANSPOSE_2X2_STRIDE2_PAD0")
    return {
        "schema": SAMPLING_SCHEMA, "sampling_variant": sampling_variant,
        "down_operator": down, "down_locations": ["down1", "down2"],
        "up_operator": up, "up_locations": ["up2", "up1"],
        "convolution_padding_mode": "zeros", "teacher_sampling_variant": "BASE",
        "rotation_equivariance_guaranteed": False,
        "initialization": "FH12_NAMED_FRESH_SHARED_LAYERS_UNCHANGED_NEW_UP_KERNEL_SAME_ROLE_SEED_POLICY",
        "pan_warp_changed": False, "loss_changed": False,
    }


def _check_operators(reconstructor, sampling_variant):
    operator_manifest(sampling_variant)
    expected_down = (AverageDownConv if sampling_variant in ("DAVG", "DAVG_UBIL") else DownConv)
    expected_up = (BilinearUpConv if sampling_variant in ("UBIL", "DAVG_UBIL") else UpConv)
    for name in ("down1", "down2"):
        block = getattr(reconstructor.backbone, name)
        if type(block) is not expected_down:
            raise ValueError("Declared sampling variant differs from actual down operator")
        conv = block.conv
        stride = (1, 1) if expected_down is AverageDownConv else (2, 2)
        if (type(conv) is not nn.Conv2d or conv.kernel_size != (3, 3)
                or conv.stride != stride or conv.padding != (1, 1)
                or conv.padding_mode != "zeros" or conv.dilation != (1, 1)
                or conv.groups != 1 or conv.bias is None
                or conv.in_channels != reconstructor.spec.width
                or conv.out_channels != reconstructor.spec.width):
            raise ValueError("Unregistered down-convolution geometry")
        if expected_down is AverageDownConv:
            pool = block.pool
            if (type(pool) is not nn.AvgPool2d or pool.kernel_size != 2
                    or pool.stride != 2 or pool.padding != 0 or pool.ceil_mode
                    or pool.count_include_pad or pool.divisor_override is not None):
                raise ValueError("Unregistered average-pooling geometry")
    for name in ("up2", "up1"):
        block = getattr(reconstructor.backbone, name)
        if type(block) is not expected_up:
            raise ValueError("Declared sampling variant differs from actual up operator")
        conv = block.conv
        bilinear = expected_up is BilinearUpConv
        if (type(conv) is not (nn.Conv2d if bilinear else nn.ConvTranspose2d)
                or conv.kernel_size != ((3, 3) if bilinear else (2, 2))
                or conv.stride != ((1, 1) if bilinear else (2, 2))
                or conv.padding != ((1, 1) if bilinear else (0, 0))
                or conv.padding_mode != "zeros" or conv.dilation != (1, 1)
                or conv.groups != 1 or conv.bias is None
                or conv.in_channels != reconstructor.spec.width
                or conv.out_channels != reconstructor.spec.width
                or (not bilinear and conv.output_padding != (0, 0))):
            raise ValueError("Unregistered up-convolution geometry")


def reconstruction_manifest(reconstructor, *, sampling_variant=None, role="S"):
    """Describe and validate actual operators and count actual model parameters."""
    if not isinstance(reconstructor, BandReconstructor):
        raise TypeError("Existing BandReconstructor-compatible model required")
    actual = getattr(reconstructor, "sampling_variant", "BASE")
    variant = actual if sampling_variant is None else sampling_variant
    if variant != actual or role not in ("T", "S") or (role == "T" and variant != "BASE"):
        raise ValueError("Teacher must be BASE and the sampling declaration must match the model")
    _check_operators(reconstructor, variant)
    return {
        **operator_manifest(variant), "role": role,
        "constructor_spec": deepcopy(reconstructor.spec.manifest()),
        "parameter_count": sum(parameter.numel() for parameter in reconstructor.parameters()),
    }


def assert_sampling_contract(reconstructor, sampling_variant, *, role="S"):
    """Check configuration against actual sampler modules and checkpoint identity."""
    manifest = reconstruction_manifest(reconstructor, sampling_variant=sampling_variant, role=role)
    if sampling_variant != "BASE":
        expected = reconstructor._sampling_identity()
        actual = reconstructor._sampling30_identity
        if (actual.dtype != expected.dtype or actual.shape != expected.shape
                or not torch.equal(actual.detach().cpu(), expected)):
            raise RuntimeError("Sampling30 in-memory architecture identity mismatch")
    return manifest


class SamplingReconstructor(BandReconstructor):
    """Variant-bound checkpoint identity even when tensor shapes match BASE."""

    def __init__(self, backbone, spec, sampling_variant):
        if sampling_variant == "BASE":
            raise ValueError("BASE must use the unchanged original BandReconstructor")
        operator_manifest(sampling_variant)
        self.sampling_variant = sampling_variant
        super().__init__(backbone, spec)
        _check_operators(self, sampling_variant)
        self.register_buffer("_sampling30_identity", self._sampling_identity())

    def _sampling_identity(self):
        manifest = reconstruction_manifest(self, role="S")
        return torch.tensor(list(bytes.fromhex(_digest(manifest))), dtype=torch.uint8)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        expected = self._sampling_identity()
        actual = state_dict.get(prefix + "_sampling30_identity")
        if (not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype
                or actual.shape != expected.shape or not torch.equal(actual.cpu(), expected)):
            raise RuntimeError("Sampling30 architecture/checkpoint mismatch")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)


def build_reconstructor(spec, *, seed, role="S", sampling_variant="BASE"):
    """Preserve original named fresh initialization and change only Student sampling.

    BASE returns the original builder's exact model with no extra state. Existing
    donor loading therefore retains its strict checkpoint contract. Other variants
    add an identity buffer so DAVG cannot silently accept same-shaped BASE weights.
    """
    operator_manifest(sampling_variant)
    if role not in ("S", "T") or (role == "T" and sampling_variant != "BASE"):
        raise ValueError("Only Student sampling may change; Teacher requires BASE")
    if not isinstance(spec, BandReconstructionSpec) or spec.pan_channels != 1:
        raise TypeError("Single-PAN BandReconstructionSpec required")
    original = _build_original(spec, seed=seed, role=role)
    if sampling_variant == "BASE":
        reconstruction_manifest(original, sampling_variant="BASE", role=role)
        return original
    with _construction_rng():
        if sampling_variant in ("DAVG", "DAVG_UBIL"):
            for name in ("down1", "down2"):
                old = getattr(original.backbone, name)
                replacement = AverageDownConv(old.channels, old.out_channels)
                # Same parameter shape/name; only sampling order/stride changes.
                replacement.conv.load_state_dict(old.conv.state_dict(), strict=True)
                setattr(original.backbone, name, replacement)
        if sampling_variant in ("UBIL", "DAVG_UBIL"):
            for name in ("up2", "up1"):
                old = getattr(original.backbone, name)
                replacement = BilinearUpConv(old.channels, old.out_channels)
                _named_fresh(replacement, seed, role, f"backbone.{name}")
                setattr(original.backbone, name, replacement)
    return SamplingReconstructor(original.backbone, original.spec, sampling_variant)
