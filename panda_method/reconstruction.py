"""Reuse the existing U-Net without carrying a legacy/Student-owned aligner.

All choices that were left to the future experiment recipe are explicit. This
adapter neither changes a supplied backbone's weights nor its state-dict keys.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from fh12.model import LAYOUTS, _construction_rng, _named_fresh
from model.pancrafter_paper import ChannelLayerNorm, PANCrafterPaper, ResBlock
from pa.warp import warp_pan


@dataclass(frozen=True)
class ReconstructionSpec:
    bands: int
    width: int
    depth: tuple[int, int, int]
    input_layout: str
    base_reference: str

    def __post_init__(self):
        if self.bands not in (4, 8) or self.input_layout not in LAYOUTS:
            raise ValueError("Explicit C4/C8 and P0/PL/PH/PLH layout required")
        if self.base_reference not in ("BICUBIC_MS", "NATIVE_LMS"):
            raise ValueError("Pin the existing reconstruction base explicitly")
        if not isinstance(self.width, int) or self.width <= 0:
            raise ValueError("Positive integer width required")
        if len(self.depth) != 3 or any(not isinstance(d, int) or d <= 0 for d in self.depth):
            raise ValueError("Three positive existing U-Net depths required")
        object.__setattr__(self, "depth", tuple(self.depth))

    @property
    def input_channels(self):
        return self.bands + len(LAYOUTS[self.input_layout]) - 1

    @property
    def uses_lpan(self):
        return self.input_layout != "P0"

    def manifest(self):
        return {**asdict(self), "norm": "ln", "dropout": 0.,
                "mode_modulation": False, "attention": False,
                "input_order": list(LAYOUTS[self.input_layout])}


class Reconstructor(nn.Module):
    """No geometry prediction or PAN warp: caller supplies the shared corrected PAN."""

    def __init__(self, backbone: PANCrafterPaper, spec: ReconstructionSpec):
        super().__init__()
        if not isinstance(backbone, PANCrafterPaper):
            raise TypeError("Reuse the existing PANCrafterPaper backbone")
        if (backbone.input.in_channels != spec.input_channels
                or backbone.input.out_channels != spec.width
                or backbone.output[-1].out_channels != spec.bands
                or tuple(backbone.depth) != spec.depth
                or len(backbone.decoder1) != spec.depth[0]
                or len(backbone.decoder2) != spec.depth[1]):
            raise ValueError("Supplied U-Net does not match the pinned architecture")
        if any(getattr(backbone, name) is not None
               for name in ("cond2_e", "cond_bot", "cond2_d", "se_dec_h2")):
            raise ValueError("No additional attention/SE modules in this method")
        if len(backbone.swin_h2) or len(backbone.swin_btl):
            raise ValueError("No Swin modules in this method")
        for module in backbone.modules():
            if isinstance(module, nn.GroupNorm):
                raise ValueError("Reconstruction U-Net must retain channel LN")
            if isinstance(module, nn.Dropout) and module.p != 0:
                raise ValueError("Common ResBlock requires dropout=0")
            if isinstance(module, ResBlock):
                if module.mod is not None or module.se is not None:
                    raise ValueError("Common ResBlock requires mode_mod=False,se=False")
                if not all(isinstance(m, ChannelLayerNorm)
                           for m in (module.in_layers[0], module.out_layers[0])):
                    raise ValueError("Common ResBlock normalization differs")
        if not isinstance(backbone.output[0], ChannelLayerNorm):
            raise ValueError("Output normalization must be channel LN")
        self.backbone, self.spec = backbone, spec

    def forward(self, aligned_pan, ms, native_lms, *, aligned_lpan=None):
        b, _, h, w = aligned_pan.shape
        if (aligned_pan.shape != (b, 1, h, w) or h % 4 or w % 4
                or ms.shape != (b, self.spec.bands, h // 4, w // 4)
                or native_lms.shape != (b, self.spec.bands, h, w)):
            raise ValueError("Explicit PAN/MS/native LMS correspondence required")
        with torch.autocast(device_type=aligned_pan.device.type, enabled=False):
            base = (F.interpolate(ms.float(), scale_factor=4, mode="bicubic", align_corners=False)
                    if self.spec.base_reference == "BICUBIC_MS" else native_lms.float())
            p = aligned_pan.float()
            parts = {"P": p, "MS": base}
            if self.spec.uses_lpan:
                if aligned_lpan is None or aligned_lpan.shape != p.shape:
                    raise ValueError("PL/PH/PLH requires synchronously warped native LPAN")
                parts.update(L=aligned_lpan.float(), H=p - aligned_lpan.float())
            x_in = torch.cat([parts[key] for key in LAYOUTS[self.spec.input_layout]], dim=1)
            switches = p.new_ones(b)
            # Attention-free x_in bypass is an existing backbone interface.
            residual = self.backbone(p, None, ms.float(), switches, x_in=x_in)
            prediction = base + residual.float()
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("Nonfinite reconstruction")
        return {"y": prediction, "ms_base": base, "x_in": x_in,
                "pan_aligned": aligned_pan, "lpan_aligned": aligned_lpan}


def build_reconstructor(spec: ReconstructionSpec, *, seed: int, role: str):
    """Existing FH12/QG40 named fresh U initialization, without constructing A.

    Role, seed, capacity and input layout have no experiment defaults. Existing
    checkpoints can instead be loaded strictly into a supplied U backbone.
    """
    if role not in ("T", "S"):
        raise ValueError("Explicit T/S initialization role required")
    with _construction_rng():
        backbone = PANCrafterPaper(out_channels=spec.bands, hidden_size=spec.width,
                                  depth=spec.depth, n_attn=0, attn_locations=[],
                                  norm="ln", in_mode="paper", mode_modulation=False)
        _named_fresh(backbone, seed, role, "backbone")
        if spec.input_layout != "P0":
            original = backbone.input
            stem = nn.Conv2d(spec.input_channels, spec.width, 3, padding=1)
            with torch.no_grad():
                stem.weight.zero_()
                stem.weight[:, :1].copy_(original.weight[:, :1])
                stem.weight[:, -spec.bands:].copy_(original.weight[:, 1:])
                stem.bias.copy_(original.bias)
            backbone.input = stem
    return Reconstructor(backbone, spec)


def aligned_observations(pan, correction, *, lpan, need_lpan):
    """Exactly one native warp; LPAN uses the same sampling grid if requested."""
    with torch.autocast(device_type=pan.device.type, enabled=False):
        if need_lpan:
            if lpan is None or lpan.shape != (len(pan), 1, pan.shape[-2] // 4, pan.shape[-1] // 4):
                raise ValueError("Native LR-grid LPAN cache required by existing layout")
            low = F.interpolate(lpan.float(), scale_factor=4, mode="bicubic", align_corners=False)
            aligned = warp_pan(torch.cat((pan.float(), low), dim=1), correction.float())
            return aligned[:, :1], aligned[:, 1:]
        return warp_pan(pan.float(), correction.float()), None
