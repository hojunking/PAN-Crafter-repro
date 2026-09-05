"""M-frame PAN feature alignment (§10–§12).

first conv 분리(§10.1): F0 = Conv([P3, M8]) = Conv(P3; W[:, :3]) + Conv(M8; W[:, 3:]) + b  (선형이라 정확, T18/T19).
global correlator(§11): 25 후보 offset 에 대한 edge-weighted descriptor 상관 -> softmax(τ) -> soft-argmax Δ̂,
  confidence c = 1 - H(p)/log25, gate g = clip(c/0.30, 0, 1), F̃_P = F_P + g[W(F_P, Δ̂) - F_P].
  synthetic 학습: F_P^syn = W(F_P, ε_g), target Δ* = -ε_g  (W(x,δ)[x] = x[x+δ] 규약에서 되돌림).
local(§12, 추론 진단만): F_P + c_l(x)[W(F_P, Δ_l(x)) - F_P] — 밀집 field.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from align.resample import warp_hr, hr_grid


def split_first_conv(conv, x11, n_pan=3):
    """(F_P, F_M): F_P 는 bias 없이 PAN 3ch 기여, F_M 은 MS 8ch 기여 + bias. 합이 conv(x11) 과 같다."""
    W, b = conv.weight, conv.bias
    pad = conv.padding
    f_p = F.conv2d(x11[:, :n_pan], W[:, :n_pan], None, conv.stride, pad, conv.dilation)
    f_m = F.conv2d(x11[:, n_pan:], W[:, n_pan:], b, conv.stride, pad, conv.dilation)
    return f_p, f_m


def edge_weight(pan_hf, q=0.99):
    """w = 0.25 + 0.75·clip(|PAN_HF| / q99(|PAN_HF|), 0, 1)  (표본별 정규화)."""
    a = pan_hf.abs()
    B = a.shape[0]
    scale = torch.quantile(a.reshape(B, -1).float(), q, dim=1).clamp_min(1e-6).view(B, 1, 1, 1).to(a.dtype)
    return 0.25 + 0.75 * (a / scale).clamp(0, 1)


class GlobalCorrelator(nn.Module):
    def __init__(self, channels, desc=16, radius_hr=1.0, n_per_axis=5, tau=0.07, gate_c0=0.30):
        super().__init__()
        self.proj_pan = nn.Conv2d(channels, desc, 1)
        self.proj_ms = nn.Conv2d(channels, desc, 1)
        ax = torch.linspace(-radius_hr, radius_hr, n_per_axis)
        cand = torch.stack(torch.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2)   # [K,2] = (dy,dx)
        self.register_buffer("cand", cand, persistent=False)
        self.register_buffer("is_boundary", (cand.abs().max(dim=1).values >= radius_hr - 1e-6), persistent=False)
        self.register_buffer("is_center", (cand.abs().sum(dim=1) < 1e-6), persistent=False)
        self.tau, self.gate_c0, self.radius = float(tau), float(gate_c0), float(radius_hr)
        self.n_calls = 0

    def scores(self, f_p, f_m, w_edge):
        """s[b,k] = mean_x w(x) <D_M(x), D_P(x+δ_k)>."""
        d_m = F.normalize(self.proj_ms(f_m), dim=1)
        d_p = F.normalize(self.proj_pan(f_p), dim=1)
        B, D, H, W = d_p.shape
        K = self.cand.shape[0]
        d_p_rep = d_p.unsqueeze(1).expand(B, K, D, H, W).reshape(B * K, D, H, W)
        delta = self.cand.to(d_p.dtype).unsqueeze(0).expand(B, K, 2).reshape(B * K, 2)
        shifted = warp_hr(d_p_rep, delta).reshape(B, K, D, H, W)
        corr = (shifted * d_m.unsqueeze(1)).sum(dim=2)                 # [B,K,H,W]
        w = w_edge / w_edge.mean(dim=(2, 3), keepdim=True)            # 가중 평균 (합 1 정규화)
        return (corr * w).mean(dim=(2, 3))                              # [B,K]

    def forward(self, f_p, f_m, w_edge):
        self.n_calls += 1
        s = self.scores(f_p, f_m, w_edge)
        p = torch.softmax(s / self.tau, dim=1)
        delta = p @ self.cand.to(p.dtype)                               # soft-argmax [B,2]
        ent = -(p * (p + 1e-12).log()).sum(dim=1)
        conf = 1.0 - ent / math.log(p.shape[1])
        gate = (conf / self.gate_c0).clamp(0, 1)
        return dict(delta=delta, p=p, conf=conf, gate=gate, scores=s,
                    p_boundary=p[:, self.is_boundary].sum(dim=1), p_center=p[:, self.is_center].sum(dim=1))

    @staticmethod
    def apply(f_p, delta, gate, scale=1.0):
        """F̃_P = F_P + scale·g·[W(F_P, Δ̂) − F_P]. gate 0 이면 F_P 그대로(T24)."""
        if scale == 0.0:
            return f_p
        moved = warp_hr(f_p, delta.to(f_p.dtype))
        return f_p + (scale * gate).to(f_p.dtype).view(-1, 1, 1, 1) * (moved - f_p)


def apply_local_field(f_p, flow_hr, gate_map):
    """§12.4: F_P + c_l(x)[W(F_P, Δ_l(x)) − F_P]. flow_hr [B,2,H,W]=(dy,dx), gate_map [B,1,H,W]∈[0,1]."""
    B, C, H, W = f_p.shape
    yy, xx = hr_grid(B, H, W, f_p.dtype, f_p.device)
    sy, sx = yy + flow_hr[:, 0], xx + flow_hr[:, 1]
    gy, gx = 2.0 * sy / max(H - 1, 1) - 1.0, 2.0 * sx / max(W - 1, 1) - 1.0
    moved = F.grid_sample(f_p, torch.stack([gx, gy], -1), mode="bicubic", padding_mode="border", align_corners=True)
    return f_p + gate_map.to(f_p.dtype) * (moved - f_p)
