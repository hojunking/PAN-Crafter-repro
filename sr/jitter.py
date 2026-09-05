"""Random C2 (§5) · blur control (§8).

translate_hr 는 align.resample.warp_hr 를 그대로 쓴다 (부호·bicubic·border 동일, T03/T04 검증됨).
ε 는 표본당 하나, 8 밴드 동일, HR px 단위. ε=0 이면 warp 를 호출하지 않고 **입력 텐서 그 객체**를 돌려준다(T01).
"""
import math

import torch
import torch.nn.functional as F

from align.resample import warp_hr
from align.shiftnet import scharr_mag_t


def sample_jitter(B, max_abs_hr_px, device, dtype, probability=1.0, generator=None):
    """[B,2] = (dy, dx) ~ U(-r, r)². probability<1 이면 나머지는 정확히 0."""
    e = (torch.rand(B, 2, device=device, dtype=dtype, generator=generator) * 2 - 1) * float(max_abs_hr_px)
    if probability < 1.0:
        keep = (torch.rand(B, 1, device=device, dtype=dtype, generator=generator) < probability).to(dtype)
        e = e * keep
    return e


def translate_hr(x, eps_hr, padding_mode="border"):
    """out[y,x] = x[y+dy, x+dx]. 모든 채널 같은 grid. ε 가 전부 0 이면 x 자체를 돌려준다."""
    if eps_hr is None or (not eps_hr.requires_grad and float(eps_hr.detach().abs().max()) == 0.0):
        return x
    return warp_hr(x, eps_hr.to(x.dtype), padding_mode)


def gaussian_kernel_1d(sigma, dtype, device):
    r = max(1, int(math.ceil(3.0 * sigma)))
    ax = torch.arange(-r, r + 1, dtype=dtype, device=device)
    k = torch.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def gaussian_blur_depthwise(x, sigma, padding_mode="replicate"):
    """채널별 독립(depthwise) 분리형 가우시안. 위치 중심은 바뀌지 않는다(대칭 커널, T11). sigma<=0 이면 x."""
    if sigma is None or float(sigma) <= 0:
        return x
    B, C, H, W = x.shape
    k = gaussian_kernel_1d(float(sigma), x.dtype, x.device)
    r = k.numel() // 2
    kw = k.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    kh = k.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    y = F.conv2d(F.pad(x, (r, r, 0, 0), mode=padding_mode), kw, groups=C)
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode=padding_mode), kh, groups=C)
    return y


def grad_energy(x):
    """E[|∇x|] — 밴드별 Scharr 크기의 전체 평균 (표본·밴드·화소 평균). [B,C,H,W] -> float tensor."""
    B, C, H, W = x.shape
    return scharr_mag_t(x.reshape(B * C, 1, H, W)).mean()


def grad_energy_ratio(cond, base):
    return (grad_energy(cond) / (grad_energy(base) + 1e-12)).item()


def _match_stat(cond, base, match):
    """match='grad_energy': E|∇cond|/E|∇base| (계획 §8.1).  match='mse': E|cond−base|² (perturbation 크기)."""
    if match == "grad_energy":
        return (grad_energy(cond) / (grad_energy(base) + 1e-12)).item()
    return (cond - base).pow(2).mean().item()


@torch.no_grad()
def calibrate_blur_sigma(ms_base, max_abs_hr_px, candidates, n_draw=8, seed=2025, tol=0.01, match="grad_energy"):
    """§8.1: J1 jitter 의 통계(r_jit)와 가장 가까운 Gaussian σ*. 후보 격자에서 상대오차 > tol 이면 이웃 사이를 이분해
    tol 안으로 좁힌다 (T13). match='grad_energy' 가 계획 원안, 'mse' 는 perturbation 크기 매칭 (검토서 참고 —
    bicubic warp 는 gradient 를 줄이지 않아(r_jit>1) 원안은 σ→0 항등으로 퇴화한다)."""
    g = torch.Generator(device=ms_base.device).manual_seed(seed)
    r_jit = 0.0
    for _ in range(n_draw):
        eps = sample_jitter(ms_base.shape[0], max_abs_hr_px, ms_base.device, ms_base.dtype, generator=g)
        r_jit += _match_stat(translate_hr(ms_base, eps), ms_base, match)
    r_jit /= n_draw
    rel = lambda r: abs(r - r_jit) / (abs(r_jit) + 1e-12)
    table = {float(s): _match_stat(gaussian_blur_depthwise(ms_base, s), ms_base, match) for s in candidates}
    best = min(table, key=lambda s: rel(table[s]))
    refined = False
    if rel(table[best]) > tol:
        ss = sorted(table); i = ss.index(best)
        lo, hi = (ss[i - 1] if i > 0 else 0.0), (ss[i + 1] if i + 1 < len(ss) else ss[i] * 2.0)
        # 후보 격자 밖이면 목표를 bracket 할 때까지 상한을 늘린다 (mse 는 σ↑ 에 단조증가)
        for _ in range(12):
            r_hi = _match_stat(gaussian_blur_depthwise(ms_base, hi), ms_base, match); table[hi] = r_hi
            bracketed = (r_hi <= r_jit) if match == "grad_energy" else (r_hi >= r_jit)
            if bracketed:
                break
            lo, hi = hi, hi * 2.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            r = _match_stat(gaussian_blur_depthwise(ms_base, mid), ms_base, match); table[mid] = r
            if rel(r) <= tol:
                best = mid; refined = True; break
            # grad_energy 는 σ↑ 에 단조감소, mse 는 단조증가
            if (r > r_jit) == (match == "grad_energy"):
                lo = mid
            else:
                hi = mid
        best = min(table, key=lambda s: rel(table[s]))
    return dict(match=match, sigma_star=float(best), r_jit=float(r_jit), r_blur_star=float(table[best]),
                rel_err=float(rel(table[best])), within_tol=bool(rel(table[best]) <= tol),
                refined=refined, table={f"{k:.4f}": float(v) for k, v in sorted(table.items())},
                max_abs_hr_px=float(max_abs_hr_px), n_draw=n_draw, seed=seed)
