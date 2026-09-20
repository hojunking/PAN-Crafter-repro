"""Explicit-band QG40 frontend, preserving the FH12 C8 initialization contract.

The original backbone/aligner modules remain untouched. PANCrafterPaper already
accepts out_channels; its attention-free input and output are bound here.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from fh12.model import (INIT_POLICY, LAYOUTS, FH12Model, _construction_rng,
                        _named_fresh, state_hash, tensor_hash)
from model.pancrafter_paper import PANCrafterPaper
from pa.aligner import PANGlobalAligner
from pa.warp import warp_pan


def input_channels(layout, bands):
    if layout not in LAYOUTS or bands not in (4, 8):
        raise ValueError("QG40 requires a registered layout and explicit C4 or C8")
    return bands + len(LAYOUTS[layout]) - 1


def sync_frontend(pan, ms, lpan, delta, layout, ms_base=None, *, bands=4):
    """Warp normalized PAN and U4(LPAN) together, then subtract signed H."""
    input_channels(layout, bands)
    if pan.ndim != 4 or pan.shape[1] != 1 or ms.ndim != 4 or ms.shape[1] != bands:
        raise ValueError(f"Expected PAN[B,1,H,W] and MS[B,{bands},H/4,W/4]")
    if pan.shape[0] != ms.shape[0] or pan.shape[-2:] != tuple(4 * n for n in ms.shape[-2:]):
        raise ValueError("PAN and LRMS must have exact 4x correspondence")
    if lpan is None or lpan.shape != (pan.shape[0], 1, *ms.shape[-2:]):
        raise ValueError("Native-PAN LP cache must match LRMS spatial dimensions")
    if delta.shape != (pan.shape[0], 2):
        raise ValueError("Offsets must be [B,2] (dy,dx) in HR pixels")
    with torch.autocast(device_type=pan.device.type, enabled=False):
        base = (F.interpolate(ms.float(), scale_factor=4, mode="bicubic", align_corners=False)
                if ms_base is None else ms_base.float())
        if base.shape != (pan.shape[0], bands, *pan.shape[-2:]):
            raise ValueError("MS base must have explicit sensor bands at full resolution")
        low = F.interpolate(lpan.float(), scale_factor=4, mode="bicubic", align_corners=False)
        aligned = warp_pan(torch.cat((pan.float(), low), dim=1), delta.float())
        p, low = aligned[:, :1], aligned[:, 1:]
        high = p - low
        parts = {"P": p, "L": low, "H": high, "MS": base}
        inputs = torch.cat([parts[key] for key in LAYOUTS[layout]], dim=1)
    return dict(x_in=inputs, pan_aligned=p, L=low, H=high,
                ms_base=base, delta=delta.float())


class QG40Model(FH12Model):
    def __init__(self, backbone, aligner, input_layout, bands):
        super().__init__(backbone, aligner, input_layout)
        self.bands = bands

    def forward(self, pan, ms, lpan, aligner_enabled=True, delta_override=None):
        with torch.autocast(device_type=pan.device.type, enabled=False):
            base = F.interpolate(ms.float(), scale_factor=4, mode="bicubic", align_corners=False)
            if delta_override is not None:
                delta = delta_override.float()
            elif aligner_enabled:
                delta = self.predict_delta(pan, base)
            else:
                delta = torch.zeros(pan.shape[0], 2, device=pan.device, dtype=torch.float32)
            features = sync_frontend(pan, ms, lpan, delta, self.input_layout,
                                     ms_base=base, bands=self.bands)
            switches = torch.ones(pan.shape[0], device=pan.device, dtype=torch.float32)
            residual = self.backbone(pan, lpan, ms, switches, x_in=features["x_in"])
            features["y"] = base + residual.float()
        return features


def build_model(layout, width, depth, seed, role="T", teacher_aligner_state=None, bands=4, num_bands=None):
    """Fresh U and T-A, or an independent trainable clone of explicit S-A.

    The existing name/shape/role/seed hash is retained, including C8 bitwise
    regression. Profile and sensor names never perturb matching fresh tensors.
    """
    if num_bands is not None:
        if bands != 4 and bands != num_bands:
            raise ValueError("Conflicting band counts")
        bands = num_bands
    channels = input_channels(layout, bands)
    if role not in ("T", "S"):
        raise ValueError("Role must be T or S")
    if len(depth) != 3 or any(int(n) != n or n < 1 for n in depth) or int(width) != width or width < 1:
        raise ValueError("Width and three depths must be positive integers")
    if (role == "T") != (teacher_aligner_state is None):
        raise ValueError("Teacher must be fresh; Student must explicitly clone its Teacher A")
    with _construction_rng():
        backbone = PANCrafterPaper(out_channels=bands, hidden_size=int(width), depth=tuple(depth),
                                  n_attn=0, attn_locations=[], norm="ln", in_mode="paper",
                                  mode_modulation=False)
        aligner = PANGlobalAligner(ms_bands=bands)
        _named_fresh(backbone, seed, role, "backbone")
        _named_fresh(aligner, seed, role, "aligner")
        if layout != "P0":
            original = backbone.input
            stem = nn.Conv2d(channels, int(width), 3, padding=1)
            with torch.no_grad():
                stem.weight.zero_()
                stem.weight[:, :1].copy_(original.weight[:, :1])
                stem.weight[:, -bands:].copy_(original.weight[:, 1:])
                stem.bias.copy_(original.bias)
            backbone.input = stem
        if teacher_aligner_state is not None:
            aligner.load_state_dict({k: v.detach().clone() for k, v in teacher_aligner_state.items()}, strict=True)
        model = QG40Model(backbone, aligner, layout, bands)
    manifest = dict(policy=INIT_POLICY, seed=int(seed), role=role, layout=layout, bands=bands,
                    width=int(width), depth=list(depth), input_channels=channels,
                    input_order=list(LAYOUTS[layout]), from_scratch=role == "T",
                    pretrained_backbone_loads=0, pretrained_aligner_loads=int(role == "S"),
                    student_aligner_independent_clone=role == "S",
                    extra_kernel_zero=not bool(torch.count_nonzero(backbone.input.weight[:, 1:-bands])),
                    hashes=dict(P=tensor_hash(backbone.input.weight[:, :1]),
                                MS=tensor_hash(backbone.input.weight[:, -bands:]),
                                input_bias=tensor_hash(backbone.input.bias),
                                body=state_hash({k: v for k, v in backbone.state_dict().items() if not k.startswith("input.")}),
                                A=state_hash(aligner.state_dict()), full=state_hash(model.state_dict())))
    return model, manifest
