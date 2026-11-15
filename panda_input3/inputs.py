"""Explicit PAN channel layouts with matched initialization of shared weights."""
import torch
from torch import nn
from fh12.model import _construction_rng
from panda_method.reconstruction import Reconstructor
from panda_unified.core import (UnifiedReconstructor, UnifiedReconstructionSpec,
                                build_reconstructor)

ORDERS = {
    'RAW': ['ORIGINAL_PAN'],
    'RAW_CORRECTED': ['ORIGINAL_PAN', 'CORRECTED_PAN'],
    'CORRECTED_DIFF': ['CORRECTED_PAN', 'CORRECTED_PAN_MINUS_ORIGINAL_PAN'],
}


def pan_views(raw, corrected, variant):
    if raw.shape != corrected.shape or raw.ndim != 4 or raw.shape[1] != 1:
        raise ValueError('Matching B1HW original and corrected PAN required')
    if variant == 'RAW':
        return raw
    if variant == 'RAW_CORRECTED':
        return torch.cat((raw, corrected), 1)
    if variant == 'CORRECTED_DIFF':
        return torch.cat((corrected, corrected - raw), 1)
    raise ValueError('Unknown PAN input layout')


class InputReconstructor(UnifiedReconstructor):
    def __init__(self, backbone, spec, variant):
        if variant not in ORDERS or spec.pan_channels != len(ORDERS[variant]):
            raise ValueError('PAN layout and first-convolution shape differ')
        Reconstructor.__init__(self, backbone, spec)
        self.pan_channels, self.role = spec.pan_channels, 'S'
        self.sampling_variant, self.input_variant = 'UBIL', variant
        self.register_buffer('_unified_identity', self._identity())

    def manifest(self):
        value = super().manifest()
        value.update(schema='PANDA_INPUT3_STUDENT_V1', input_variant=self.input_variant,
            input_order=ORDERS[self.input_variant] + [f'MS_{i}' for i in range(self.spec.bands)],
            initialization='MATCHED_SHARED_WEIGHTS_EXTRA_PAN_ZERO')
        return value


def build_student(sensor, width, seed, variant):
    if variant not in ORDERS:
        raise ValueError('Unknown PAN input layout')
    original = build_reconstructor(sensor, width, seed, role='S')
    channels = len(ORDERS[variant])
    spec = UnifiedReconstructionSpec(sensor, width, pan_channels=channels)
    # The standard two-PAN stem starts with [W_pan, 0, W_ms]. Both B and C
    # start from the same corrected-PAN function. Their trainable bases differ.
    with _construction_rng():
        old = original.backbone.input
        stem = nn.Conv2d(spec.input_channels, width, 3, padding=1)
        with torch.no_grad():
            stem.weight.zero_()
            slot = 1 if variant == 'RAW_CORRECTED' else 0
            stem.weight[:, slot:slot+1].copy_(old.weight[:, :1])
            stem.weight[:, channels:].copy_(old.weight[:, 2:])
            stem.bias.copy_(old.bias)
        original.backbone.input = stem
    return InputReconstructor(original.backbone, spec, variant)
