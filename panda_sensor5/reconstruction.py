"""Isolated Student sampling ablations for the existing three-scale U-Net.

The frozen Teacher remains byte-for-byte compatible with the existing builder.
These sampling changes do not enforce rotation equivariance: learned convolution
kernels remain unconstrained. The additional difference channel starts at zero stem weight; existing loss is unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from panda_method.reconstruction import ReconstructionSpec
from .contract import input_manifest

import torch
from torch import nn
from torch.nn import functional as F

from directshift48.method import _digest
from fh12.model import _construction_rng, _named_fresh
from model.pancrafter import DownConv, UpConv
from panda_sensor5.base_reconstruction import (
    BandReconstructionSpec, BandReconstructor,
    build_reconstructor as _build_original,
    base_from_batch, reconstruct_with_base, upsampler_manifest,
)


SAMPLING_VARIANTS = ("BASE", "DAVG", "UBIL", "DAVG_UBIL")
SAMPLING_SCHEMA = "PANDA_NEXT30_RECONSTRUCTION_V1"


@dataclass(frozen=True)
class StudentReconstructionSpec(BandReconstructionSpec):
    """Isolated truthful 5/6-channel Student spec; donor PAN1 is unchanged."""
    def __post_init__(self):
        ReconstructionSpec.__post_init__(self)
        if (self.bands != 4 or type(self.pan_channels) is not int
                or self.pan_channels not in (1, 2) or self.input_layout != 'P0'
                or self.base_reference != 'BICUBIC_MS'):
            raise ValueError('four-band sensor corrected-PAN / corrected-plus-difference input required')

    def manifest(self):
        variant = 'CORRECTED_DIFF' if self.pan_channels == 2 else 'CORRECTED'
        result = ReconstructionSpec.manifest(self)
        result.update(schema='PANDA_SENSOR5_STUDENT_RECONSTRUCTION_SPEC_V1',
            pan_channels=self.pan_channels, input_channels=self.input_channels,
            input_order=(['CORRECTED_PAN', 'CORRECTED_PAN_MINUS_ORIGINAL_PAN']
                         if self.pan_channels == 2 else ['CORRECTED_PAN']) +
                        [f'MS_{i}' for i in range(self.bands)],
            student_input=input_manifest(variant,self.width),
            base_reference_is_constructor_compatibility_only=True)
        return result


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
        "sampling_builder_changes_pan_warp": False, "loss_changed": False,
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


def reconstruction_manifest(reconstructor, *, sampling_variant=None, input_variant=None, role="S"):
    """Describe and validate actual operators and count actual model parameters."""
    if not isinstance(reconstructor, BandReconstructor):
        raise TypeError("Existing BandReconstructor-compatible model required")
    actual = getattr(reconstructor, "sampling_variant", "BASE")
    variant = actual if sampling_variant is None else sampling_variant
    if variant != actual or role not in ("T", "S") or (role == "T" and variant != "BASE"):
        raise ValueError("Teacher must be BASE and the sampling declaration must match the model")
    actual_input = getattr(reconstructor, 'input_variant', 'CORRECTED')
    declared_input = actual_input if input_variant is None else input_variant
    if (declared_input != actual_input or declared_input not in ('CORRECTED', 'CORRECTED_DIFF')
            or (role == 'T' and declared_input != 'CORRECTED')
            or reconstructor.spec.pan_channels != (2 if declared_input == 'CORRECTED_DIFF' else 1)
            or reconstructor.backbone.input.in_channels != reconstructor.spec.input_channels):
        raise ValueError('Declared Student PAN inputs differ from actual convolution stem')
    _check_operators(reconstructor, variant)
    return {
        **operator_manifest(variant), "role": role, "input_variant": declared_input,
        "student_input": input_manifest(declared_input,reconstructor.spec.width),
        "constructor_spec": deepcopy(reconstructor.spec.manifest()),
        "parameter_count": sum(parameter.numel() for parameter in reconstructor.parameters()),
    }


def assert_sampling_contract(reconstructor, sampling_variant, *, input_variant=None, role="S"):
    """Check configuration against actual sampler modules and checkpoint identity."""
    manifest = reconstruction_manifest(reconstructor, sampling_variant=sampling_variant, input_variant=input_variant, role=role)
    if isinstance(reconstructor, SamplingReconstructor):
        expected = reconstructor._sampling_identity()
        actual = reconstructor._sampling30_identity
        if (actual.dtype != expected.dtype or actual.shape != expected.shape
                or not torch.equal(actual.detach().cpu(), expected)):
            raise RuntimeError("Sampling30 in-memory architecture identity mismatch")
    return manifest


class SamplingReconstructor(BandReconstructor):
    """Variant-bound checkpoint identity even when tensor shapes match BASE."""

    def __init__(self, backbone, spec, sampling_variant, input_variant="CORRECTED"):
        if sampling_variant == "BASE" and input_variant == "CORRECTED":
            raise ValueError("BASE must use the unchanged original BandReconstructor")
        operator_manifest(sampling_variant)
        self.sampling_variant = sampling_variant
        self.input_variant = input_variant
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


def build_reconstructor(spec, *, seed, role="S", sampling_variant="BASE", input_variant="CORRECTED"):
    """Preserve original named fresh initialization and change only Student sampling.

    BASE returns the original builder's exact model with no extra state. Existing
    donor loading therefore retains its strict checkpoint contract. Other variants
    add an identity buffer so DAVG cannot silently accept same-shaped BASE weights.
    """
    operator_manifest(sampling_variant)
    if role not in ("S", "T") or (role == "T" and (sampling_variant != "BASE" or input_variant != "CORRECTED")):
        raise ValueError("Only Student sampling may change; Teacher requires BASE")
    expected_channels = 2 if input_variant == 'CORRECTED_DIFF' else 1
    input_manifest(input_variant)
    if not isinstance(spec, BandReconstructionSpec) or spec.pan_channels != expected_channels:
        raise TypeError('Matching isolated Student input spec required')
    base_spec = BandReconstructionSpec(spec.bands, spec.width, spec.depth,
                                      spec.input_layout, spec.base_reference, 1)
    original = _build_original(base_spec, seed=seed, role=role)
    if sampling_variant == "BASE" and input_variant == 'CORRECTED':
        reconstruction_manifest(original, sampling_variant="BASE", role=role)
        return original
    with _construction_rng():
        if input_variant == 'CORRECTED_DIFF':
            old = original.backbone.input
            stem = nn.Conv2d(6, spec.width, 3, padding=1)
            with torch.no_grad():
                stem.weight[:, :1].copy_(old.weight[:, :1])
                stem.weight[:, 1:2].zero_()
                stem.weight[:, 2:].copy_(old.weight[:, 1:])
                stem.bias.copy_(old.bias)
            original.backbone.input = stem
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
    return SamplingReconstructor(original.backbone, spec, sampling_variant, input_variant)
