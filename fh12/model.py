"""FH12-only synchronized PAN/LPAN frontend and coupled fresh initialization.

This module does not alter PAModel or the running legacy training entrypoints.
The model output is in the original MS frame, with the MS base added once.
"""
from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.pancrafter_paper import PANCrafterPaper
from pa.aligner import PANGlobalAligner
from pa.warp import warp_pan


LAYOUTS = {"P0": ("P", "MS"), "PL": ("P", "L", "MS"),
           "PH": ("P", "H", "MS"), "PLH": ("P", "L", "H", "MS")}
INPUT_CHANNELS = {name: len(parts) + 7 for name, parts in LAYOUTS.items()}
INIT_POLICY = "FH12_named_tensor_fresh9_zero_extra_v1"


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode()
                          + value.numpy().tobytes()).hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor_hash(tensor).encode())
    return digest.hexdigest()


@contextmanager
def _construction_rng():
    # Constructors use only CPU Torch RNG. Do not initialize CUDA or modify its
    # RNG, NumPy, or Python; fork_rng restores the caller's CPU state on errors too.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        yield


def _generator(seed, role, name, shape):
    key = json.dumps([INIT_POLICY, int(seed), role, name, list(shape)]).encode()
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63 - 1)
    return torch.Generator(device="cpu").manual_seed(value)


def _named_fresh(module, seed, role, prefix):
    """Default Conv/Linear initialization, keyed by name/shape, not draw order.

    Zero residual projections and the aligner final head retain their intentional
    zero initialization. Norm defaults are constants, already name-independent.
    This makes matching names/shapes equal even when depth/width changes.
    """
    with torch.no_grad():
        for name, layer in module.named_modules():
            if not isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                continue
            qualified = f"{prefix}.{name}"
            if torch.count_nonzero(layer.weight).item() == 0:
                continue
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5),
                                     generator=_generator(seed, role, qualified + ".weight", layer.weight.shape))
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in else 0
                nn.init.uniform_(layer.bias, -bound, bound,
                                 generator=_generator(seed, role, qualified + ".bias", layer.bias.shape))


def sync_frontend(pan, ms, lpan, delta, layout, ms_base=None):
    """One FP32 shared-grid warp of [PAN,U4(LPAN)], then signed subtraction.

    delta=(dy,dx), positive values increase source sample coordinates. No
    clipping/detaching is performed and LPAN must explicitly be supplied.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"unknown FH12 input layout: {layout!r}")
    if pan.ndim != 4 or pan.shape[1] != 1 or ms.ndim != 4 or ms.shape[1] != 8:
        raise ValueError("FH12 requires PAN[B,1,H,W] and WV3 MS[B,8,H/4,W/4]")
    if lpan is None or lpan.shape != (pan.shape[0], 1, *ms.shape[-2:]):
        raise ValueError("FH12 LPAN is mandatory and must have LRMS spatial dimensions")
    if pan.shape[0] != ms.shape[0] or tuple(pan.shape[-2:]) != tuple(4 * n for n in ms.shape[-2:]):
        raise ValueError("PAN and LRMS must have exact 4x spatial correspondence")
    if delta.shape != (pan.shape[0], 2):
        raise ValueError("delta must be [B,2] in (dy,dx) HR pixels")
    with torch.autocast(device_type=pan.device.type, enabled=False):
        p, m, lp, d = pan.float(), ms.float(), lpan.float(), delta.float()
        base = (F.interpolate(m, scale_factor=4, mode="bicubic", align_corners=False)
                if ms_base is None else ms_base.float())
        if base.shape != (pan.shape[0], 8, *pan.shape[-2:]):
            raise ValueError("MS base must be the full-resolution WV3 tensor")
        low = F.interpolate(lp, scale_factor=4, mode="bicubic", align_corners=False)
        aligned = warp_pan(torch.cat((p, low), dim=1), d)
        p_al, l_al = aligned[:, :1], aligned[:, 1:]
        high = p_al - l_al
        parts = {"P": p_al, "L": l_al, "H": high, "MS": base}
        x_in = torch.cat([parts[key] for key in LAYOUTS[layout]], dim=1)
    return dict(x_in=x_in, pan_aligned=p_al, L=l_al, H=high, ms_base=base,
                delta=d)


class FH12Model(nn.Module):
    """Dedicated FH12 wrapper; only its aligner views are cropped by margin 4."""
    def __init__(self, backbone, aligner, input_layout):
        super().__init__()
        self.backbone = backbone
        self.aligner = aligner
        self.input_layout = input_layout
        self.aligner_margin = 4
        self.sampler = True

    def _view(self, x):
        m = self.aligner_margin
        if min(x.shape[-2:]) <= 2 * m + 4:
            raise ValueError("FH12 aligner input is too small for its fixed margin")
        return x[..., m:-m, m:-m]

    def predict_delta(self, pan, ms_base):
        with torch.autocast(device_type=pan.device.type, enabled=False):
            return self.aligner(self._view(pan.float()), self._view(ms_base.float()))

    def forward(self, pan, ms, lpan, aligner_enabled=True, delta_override=None):
        with torch.autocast(device_type=pan.device.type, enabled=False):
            base = F.interpolate(ms.float(), scale_factor=4, mode="bicubic", align_corners=False)
            if delta_override is not None:
                delta = delta_override.float()
            elif aligner_enabled:
                delta = self.predict_delta(pan, base)
            else:
                delta = torch.zeros(pan.shape[0], 2, device=pan.device, dtype=torch.float32)
            features = sync_frontend(pan, ms, lpan, delta, self.input_layout, ms_base=base)
        switches = torch.ones(pan.shape[0], device=pan.device, dtype=torch.float32)
        residual = self.backbone(pan, lpan, ms, switches, x_in=features["x_in"])
        features["y"] = features["ms_base"] + residual.float()
        return features


def build_model(layout: str, width: int, depth: list, seed: int, role: str = "T",
                teacher_aligner_state=None):
    """Build fresh U, and fresh T-A or an independent explicitly supplied S-A.

    A loaded Student checkpoint should pass its own saved aligner sub-state here
    before loading the complete checkpoint. No path/donor/default lookup exists.
    """
    if layout not in LAYOUTS or role not in ("T", "S"):
        raise ValueError("FH12 needs a registered layout and role T or S")
    if len(depth) != 3 or any(int(n) != n or n < 1 for n in depth) or int(width) != width or width < 1:
        raise ValueError("FH12 width and three depths must be positive integers")
    if role == "T" and teacher_aligner_state is not None:
        raise ValueError("FH12 Teacher must be fresh; an aligner donor is forbidden")
    if role == "S" and teacher_aligner_state is None:
        raise ValueError("FH12 Student needs its explicit local Teacher aligner state")
    with _construction_rng():
        backbone = PANCrafterPaper(hidden_size=int(width), depth=tuple(depth), n_attn=0,
                                  attn_locations=[], norm="ln", in_mode="paper",
                                  mode_modulation=False)
        aligner = PANGlobalAligner(ms_bands=8)
        _named_fresh(backbone, seed, role, "backbone")
        _named_fresh(aligner, seed, role, "aligner")
        old_input = backbone.input
        if layout != "P0":
            stem = nn.Conv2d(INPUT_CHANNELS[layout], int(width), 3, padding=1)
            with torch.no_grad():
                stem.weight.zero_()
                stem.weight[:, :1].copy_(old_input.weight[:, :1])
                stem.weight[:, -8:].copy_(old_input.weight[:, 1:])
                stem.bias.copy_(old_input.bias)
            backbone.input = stem
        if teacher_aligner_state is not None:
            aligner.load_state_dict({key: value.detach().clone() for key, value in teacher_aligner_state.items()}, strict=True)
        model = FH12Model(backbone, aligner, layout)
    state = backbone.state_dict()
    manifest = dict(policy=INIT_POLICY, seed=int(seed), role=role, layout=layout,
                    width=int(width), depth=list(depth), input_channels=INPUT_CHANNELS[layout],
                    input_order=list(LAYOUTS[layout]), from_scratch=(role == "T"),
                    pretrained_backbone_loads=0, pretrained_aligner_loads=int(role == "S"),
                    student_aligner_independent_clone=(role == "S"),
                    extra_kernel_zero=bool(torch.count_nonzero(backbone.input.weight[:, 1:-8]).item() == 0),
                    hashes=dict(P=tensor_hash(backbone.input.weight[:, :1]),
                                MS=tensor_hash(backbone.input.weight[:, -8:]),
                                input_bias=tensor_hash(backbone.input.bias),
                                body=state_hash({k: v for k, v in state.items() if not k.startswith("input.")}),
                                A=state_hash(aligner.state_dict()), full=state_hash(model.state_dict())))
    return model, manifest


def frontend_self_test(device="cpu"):
    """Synthetic shape/sign/shared-gradient test without changing caller RNG."""
    dev = torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(87209)
    records = []
    for size in (64, 256, 512):
        pan = torch.randn(2, 1, size, size, generator=gen).to(dev)
        lp = torch.randn(2, 1, size // 4, size // 4, generator=gen).to(dev)
        ms = torch.randn(2, 8, size // 4, size // 4, generator=gen).to(dev)
        delta = torch.tensor([[1.25, -0.75], [-1.5, 0.375]], device=dev, requires_grad=True)
        out = sync_frontend(pan, ms, lp, delta, "PLH")
        native_low = F.interpolate(lp, scale_factor=4, mode="bicubic", align_corners=False)
        direct = warp_pan(pan - native_low, delta)
        probe = torch.randn(2, 1, size, size, generator=gen).to(dev)
        ga = torch.autograd.grad((out["H"] * probe).mean(), delta, retain_graph=True)[0]
        gb = torch.autograd.grad((direct * probe).mean(), delta)[0]
        error = float((out["H"] - direct).abs().max())
        grad_error = float((ga - gb).abs().max())
        if not torch.allclose(out["H"], direct, atol=3e-6, rtol=2e-5):
            raise AssertionError(f"shared-grid decomposition failed at {size}: {error}")
        if not torch.allclose(ga, gb, atol=2e-6, rtol=3e-4):
            raise AssertionError(f"shared c gradient failed at {size}: {grad_error}")
        for layout in LAYOUTS:
            shaped = sync_frontend(pan, ms, lp, delta.detach(), layout)
            assert shaped["x_in"].shape == (2, INPUT_CHANNELS[layout], size, size)
        ramp = torch.arange(size, device=dev, dtype=torch.float32).view(1, 1, 1, size).expand(1, 1, size, size)
        shifted = warp_pan(ramp, torch.tensor([[0., 1.]], device=dev))
        assert torch.allclose(shifted[..., 2:-2, 2:-2], ramp[..., 2:-2, 3:-1], atol=1e-4)
        records.append(dict(size=size, max_forward_error=error, max_delta_grad_error=grad_error,
                            positive_dx="increases_source_x", layouts=list(LAYOUTS)))
    return dict(passed=True, device=str(dev), records=records)
