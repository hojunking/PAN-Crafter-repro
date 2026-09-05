"""Shift module (계획 §4) · PAN warp (§1.2) · 입력 표현 (§4.2).

입력 표현: E_P = Norm(|∇ P_LR|), E_M = Norm(|∇ mean_c M|)  — P_LR 은 **feeder 가 주는 lpan**(MTF↓PAN, 데이터셋 레시피와
동일 phase, LR 증강과 정합) 을 쓴다. 증강된 HR PAN 을 다시 MTF↓ 하면 decimation phase([2::4]) 가 flip 에 비대칭이라
flip 표본에서 1 LR px 어긋난다 (2026-09-05 검증 지적의 재발 방지).

PAN-derived HR 채널의 warp: LP(W(P)) = W(LP(P)) (translation 과 low-pass 는 가환) 이므로 [P, LP, P−LP] 세 채널을
같은 4δ 로 warp 한다 — MTF 재계산·phase 문제 없이 계획 §1.2 의 I^a 와 같다 (T-test 로 오차 확인).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from align.resample import hr_grid
from align.shiftnet import scharr_mag_t, robust_norm_t


def edge_rep(x):
    """[B,1,h,w] 또는 [B,C,h,w] -> Scharr 크기 + 표본별 median/MAD 정규화 [B,1,h,w] (밴드평균 후)."""
    if x.shape[1] > 1:
        x = x.mean(dim=1, keepdim=True)
    return robust_norm_t(scharr_mag_t(x))


def warp(x, delta, scale=1.0, mode="bilinear", padding_mode="border"):
    """W(x, scale·δ)(y,x) = x(y+s·δy, x+s·δx). δ [B,2]=(dy,dx). 전부 0 이면 x 자체."""
    if delta is None or (not delta.requires_grad and float(delta.detach().abs().max()) == 0.0):
        return x
    B, _, H, W = x.shape
    yy, xx = hr_grid(B, H, W, x.dtype, x.device)
    d = (delta.to(x.dtype) * scale)
    sy, sx = yy + d[:, 0].view(B, 1, 1), xx + d[:, 1].view(B, 1, 1)
    grid = torch.stack([2.0 * sx / max(W - 1, 1) - 1.0, 2.0 * sy / max(H - 1, 1) - 1.0], -1)
    return F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode, align_corners=True)


def warp_pan_channels(pan, lpan_u, pan_hf, delta_lr, mode="bilinear"):
    """PAN 계열 3ch 을 4δ(HR px) 로 함께 warp. δ_LR=0 이면 원 텐서 그대로."""
    if delta_lr is None or (not delta_lr.requires_grad and float(delta_lr.detach().abs().max()) == 0.0):
        return pan, lpan_u, pan_hf
    x = torch.cat([pan, lpan_u, pan_hf], dim=1)
    y = warp(x, delta_lr, scale=4.0, mode=mode)
    return y[:, 0:1], y[:, 1:2], y[:, 2:3]


class ShiftModule(nn.Module):
    """modality 별 작은 encoder + [-r, r]² 정수 offset cost volume → softmax(T) → soft-argmax δ, entropy confidence.

    cost(Δ) = 위치별 L2-정규화 descriptor 의 cosine 을 유효 overlap 에서 평균. Δ 는 E_P 를 옮기는 양:
    W(E_P, Δ) 가 E_M 과 맞을 때 최대 → δ_{MS←PAN}.
    """

    def __init__(self, channels=(16, 32, 32), radius=3, temperature=0.07):
        super().__init__()
        def enc():
            layers, c_in = [], 1
            for c in channels:
                layers += [nn.Conv2d(c_in, c, 3, padding=1), nn.SiLU()]
                c_in = c
            layers.pop()                                   # 마지막 activation 제거
            return nn.Sequential(*layers)
        self.enc_p, self.enc_m = enc(), enc()
        self.radius, self.T = int(radius), float(temperature)
        ax = torch.arange(-self.radius, self.radius + 1, dtype=torch.float32)
        cand = torch.stack(torch.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2)   # [K,2]=(dy,dx)
        self.register_buffer("cand", cand, persistent=False)
        self.register_buffer("is_center", (cand.abs().sum(1) < 1e-6), persistent=False)
        self.register_buffer("is_boundary", (cand.abs().max(1).values >= self.radius - 1e-6), persistent=False)

    def cost_volume(self, e_p, e_m):
        """[B,K]: cosine(W(D_P, Δ), D_M) 의 overlap 평균. 정수 Δ 라 slicing 으로 정확히."""
        d_p = F.normalize(self.enc_p(e_p), dim=1)
        d_m = F.normalize(self.enc_m(e_m), dim=1)
        B, C, H, W = d_p.shape
        r = self.radius
        out = []
        for k in range(self.cand.shape[0]):
            dy, dx = int(self.cand[k, 0]), int(self.cand[k, 1])
            # W(D_P, Δ)(y,x) = D_P(y+dy, x+dx): 유효 영역 y∈[max(0,-dy), H-max(0,dy))
            y0, y1 = max(0, -dy), H - max(0, dy)
            x0, x1 = max(0, -dx), W - max(0, dx)
            a = d_p[:, :, y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            b = d_m[:, :, y0:y1, x0:x1]
            out.append((a * b).sum(1).mean(dim=(1, 2)))
        return torch.stack(out, 1)

    def forward(self, e_p, e_m):
        s = self.cost_volume(e_p, e_m)
        p = torch.softmax(s / self.T, dim=1)
        delta = p @ self.cand.to(p.dtype)
        ent = -(p * (p + 1e-12).log()).sum(1)
        conf = 1.0 - ent / math.log(p.shape[1])
        return dict(delta=delta, conf=conf, p=p, scores=s,
                    p_center=p[:, self.is_center].sum(1), p_boundary=p[:, self.is_boundary].sum(1))


def gated_delta(delta, conf, threshold=None):
    """ĉ·δ. threshold 가 있으면(추론) c<threshold 는 0 (no-harm gate, §4.5/§8.3)."""
    d = conf.unsqueeze(1) * delta
    if threshold is not None:
        d = d * (conf >= threshold).to(d.dtype).unsqueeze(1)
    return d
