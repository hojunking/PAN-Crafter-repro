"""Additive C4/C8 component model; legacy FH12/QG40 modules are unchanged."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from fh12.model import _construction_rng, _named_fresh, state_hash, tensor_hash, INIT_POLICY
from model.pancrafter_paper import PANCrafterPaper
from pa.aligner import PANGlobalAligner
from pa.warp import warp_pan
from qg40.model import QG40Model


class ComponentModel(QG40Model):
    def __init__(self, backbone, aligner, bands, role, aligner_mode, mask_l=1, mask_h=1):
        super().__init__(backbone, aligner, 'P0' if role == 'T' else 'PLH', bands)
        self.role, self.aligner_mode = role, aligner_mode
        self.mask_l, self.mask_h = int(mask_l), int(mask_h)
        if aligner_mode == 'CLONE_FROZEN':
            self.aligner.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.aligner_mode == 'CLONE_FROZEN':
            self.aligner.eval()
        return self

    def predict_delta(self, pan, ms_base):
        if self.aligner_mode == 'IDENTITY':
            return pan.new_zeros((len(pan), 2), dtype=torch.float32)
        return super().predict_delta(pan, ms_base)

    def forward(self, pan, ms, lpan, aligner_enabled=True, delta_override=None):
        if (pan.ndim != 4 or pan.shape[1] != 1 or ms.ndim != 4 or ms.shape[1] != self.bands
                or tuple(pan.shape[-2:]) != tuple(4*n for n in ms.shape[-2:])
                or lpan.shape != (len(pan), 1, *ms.shape[-2:])):
            raise ValueError('Explicit sensor bands and 4x PAN/MS/LP correspondence required')
        with torch.autocast(device_type=pan.device.type, enabled=False):
            base = F.interpolate(ms.float(), scale_factor=4, mode='bicubic', align_corners=False)
            low = F.interpolate(lpan.float(), scale_factor=4, mode='bicubic', align_corners=False)
            bypass = self.aligner_mode == 'IDENTITY' or not aligner_enabled
            if bypass:
                if delta_override is not None and bool(torch.count_nonzero(delta_override)):
                    raise ValueError('Identity ablation cannot receive nonzero offset override')
                delta = pan.new_zeros((len(pan), 2), dtype=torch.float32)
                p = pan.float()  # No identity-grid interpolation: exact bypass.
            else:
                delta = self.predict_delta(pan, base) if delta_override is None else delta_override.float()
                if delta.shape != (len(pan), 2):
                    raise ValueError('Expected one (dy,dx) offset per sample')
                aligned = warp_pan(torch.cat((pan.float(), low), dim=1), delta.float())
                p, low = aligned[:, :1], aligned[:, 1:]
            high = p - low
            x_in = (torch.cat((p, base), dim=1) if self.role == 'T' else
                    torch.cat((p, low*self.mask_l, high*self.mask_h, base), dim=1))
            switches = torch.ones(len(pan), device=pan.device, dtype=torch.float32)
            residual = self.backbone(pan, lpan, ms, switches, x_in=x_in)
            return dict(y=base+residual.float(), x_in=x_in, pan_aligned=p,
                        L=low, H=high, ms_base=base, delta=delta)


def build_model(*, bands, seed, role, component=None, teacher_aligner_state=None,
                width=None, depth=None):
    """Common named Student U initialization is independent of component/recipe.

    ``component`` is the resolved catalog mapping. Scratch A uses the common
    Student seed and never constructs or reads a Teacher. Optional catalog rows
    are rejected by the plan layer; this numerical wrapper supports their masks.
    """
    if bands not in (4, 8) or role not in ('T', 'S'):
        raise ValueError('Registered C4/C8 Teacher or Student required')
    width = width if width is not None else (112 if role == 'T' else 104)
    depth = depth if depth is not None else ((1, 2, 3) if role == 'T' else (1, 2, 2))
    component = component or {}
    mode = 'FRESH_TRAINABLE' if role == 'T' else component['aligner']
    if mode not in ('IDENTITY', 'FRESH_TRAINABLE', 'CLONE_TRAINABLE', 'CLONE_FROZEN'):
        raise ValueError('Unregistered alignment mode')
    clone = mode.startswith('CLONE_')
    if clone != (teacher_aligner_state is not None):
        raise ValueError('Only explicit clone cases may consume Teacher A tensors')
    with _construction_rng():
        backbone = PANCrafterPaper(out_channels=bands, hidden_size=width, depth=tuple(depth),
            n_attn=0, attn_locations=[], norm='ln', in_mode='paper', mode_modulation=False)
        _named_fresh(backbone, seed, role, 'backbone')
        aligner = nn.Identity() if mode == 'IDENTITY' else PANGlobalAligner(ms_bands=bands)
        if mode != 'IDENTITY':
            _named_fresh(aligner, seed, role, 'aligner')
        if role == 'S':
            original = backbone.input
            stem = nn.Conv2d(bands+3, width, 3, padding=1)
            with torch.no_grad():
                stem.weight.zero_()
                stem.weight[:, :1].copy_(original.weight[:, :1])
                stem.weight[:, -bands:].copy_(original.weight[:, 1:])
                stem.bias.copy_(original.bias)
            backbone.input = stem
        if clone:
            aligner.load_state_dict({k: v.detach().clone() for k, v in teacher_aligner_state.items()}, strict=True)
        model = ComponentModel(backbone, aligner, bands, role, mode,
                               component.get('mask_L', 1), component.get('mask_H', 1))
    manifest = dict(policy=INIT_POLICY, role=role, seed=int(seed), bands=bands,
        width=width, depth=list(depth), input_channels=bands+(1 if role == 'T' else 3),
        aligner_mode=mode, teacher_aligner_loaded=clone, teacher_backbone_loaded=False,
        hashes=dict(U=state_hash(backbone.state_dict()), A=state_hash(aligner.state_dict()),
                    P=tensor_hash(backbone.input.weight[:, :1]),
                    MS=tensor_hash(backbone.input.weight[:, -bands:]), full=state_hash(model.state_dict())),
        extra_kernel_zero=role == 'T' or not bool(torch.count_nonzero(backbone.input.weight[:, 1:3])))
    return model, manifest
