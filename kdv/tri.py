"""TRI-A/B/C — GT 방향·구조 통계·정합 민감성 기반 soft 감독 조절 (research_log/PAN_S2_W112_D124_TGeo_ABC_Addendum_2026-09-10.md).

공통 삼각관계 (같은 좌표의 표현 z):  u = z_G − sg(z_S) (GT 가 요구하는 수정), v = sg(z_T) − sg(z_S) (Teacher 가 요구하는 수정).
  A (§4): R3 의 hard w_H·soft w_K 는 그대로 두고 soft 안에서 band 별 gate —
          sign: m = 1[u·v > 0] · sign_cap: m·min(1, |u|/(|v|+ε)) · cosine: pixel scalar [cos(u,v)]_[0,1] broadcast · band_adv: w_K 를 band 별 (1−d_c)a_c 로 ·
          mass: κ = Σ w_K m / Σ w_K 스칼라 (soft 총계수량만 맞춘 대조) · shuffle: mask 의 공간 순서를 고정 RNG 로 섞은 부정 대조
  B (§5): 같은 gate 를 통계 표현(GV/GC/SC 성분 j) 의 soft 에 (comp_adv = 성분별 τ_V,j·a_j)
  C (§6): Teacher 출력의 correction 민감도 J_T = ∂z_T/∂(δy,δx) (중심 유한차분, Teacher U-Net 4회 forward) —
          sens: r = s_sens/(s_sens + |J|²) · diag: r = s_C²/(s_C² + diag(J Σ Jᵀ)) · full/qiso/qscalar: quadratic soft (Woodbury 2×2)
모든 gate/J/Σ 는 detach·FP32(이상). soft 만 바뀌고 hard 는 모든 위치에서 유지된다. 참고 함수(direction_mask·masked_l1·fd_jacobian·diag_risk·lowrank_quad)는 addendum §18.1 의 tensor-only reference 를 그대로 옮긴 것."""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from kdv.alignment_kd import sigma_from_precision  # noqa: F401  (C 의 Σ 출처; covariance 대수는 alignment_kd 에)
from kdv.losses_stat import statistic_map
from pa.warp import warp_pan

A_MODES = ('off', 'cosine', 'sign', 'sign_cap', 'band_adv', 'mass', 'shuffle')
B_MODES = ('off', 'sign', 'sign_cap', 'comp_adv', 'mass', 'shuffle')
C_MODES = ('off', 'sens', 'diag', 'full', 'qiso', 'qscalar')
C_PHI = ('identity', 'stat')
C_SIGMA_CONTROLS = ('none', 'rotate', 'shuffle')
TOKEN = {'off': 'OFF', 'cosine': 'COS', 'sign': 'SIGN', 'sign_cap': 'CAP', 'band_adv': 'BANDADV', 'comp_adv': 'COMPADV', 'mass': 'MASS', 'shuffle': 'SHUF',
         'sens': 'SENS', 'diag': 'DIAG', 'full': 'FULL', 'qiso': 'QISO', 'qscalar': 'QSCALAR'}


def _dtype(*xs: Tensor) -> torch.dtype:
    return torch.float64 if any(x.dtype == torch.float64 for x in xs) else torch.float32


def _images(s: Tensor, t: Tensor, g: Tensor) -> None:
    if s.ndim != 4 or s.shape != t.shape or s.shape != g.shape or min(s.shape) <= 0:
        raise ValueError("Expected equal, nonempty [B,D,H,W] tensors.")
    if not (s.device == t.device == g.device):
        raise ValueError("Device mismatch.")
    if not all(x.is_floating_point() for x in (s, t, g)):
        raise TypeError("Floating tensors required.")
    if not all(bool(torch.isfinite(x).all()) for x in (s, t, g)):
        raise FloatingPointError("Non-finite prediction/target.")


# ------------------------------------------------------------------ A/B: 방향 gate (§4.2–4.4, §5.4) — 참고 함수 그대로
@torch.no_grad()
def direction_mask(s: Tensor, t: Tensor, g: Tensor, *, mode: str = "sign", eps: float = 1e-6) -> Tensor:
    """soft 만 gate. D 는 band 또는 통계 성분. 반환 [B,D,H,W] ∈ [0,1], detached."""
    _images(s, t, g)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite.")
    dt = _dtype(s, t, g)
    u, v = g.to(dt) - s.to(dt), t.to(dt) - s.to(dt)
    if mode == "off":
        return torch.ones_like(u)
    same_sign = (((u > 0) & (v > 0)) | ((u < 0) & (v < 0))).to(dt)
    if mode == "sign":
        return same_sign
    if mode == "sign_cap":
        return same_sign * (u.abs() / (v.abs() + eps)).clamp(max=1)
    if mode == "cosine":
        norm_product = u.norm(dim=1, keepdim=True) * v.norm(dim=1, keepdim=True)
        value = (u * v).sum(dim=1, keepdim=True) / norm_product.clamp_min(eps ** 2)
        value = torch.where(norm_product > 0, value, torch.zeros_like(value))
        return value.clamp(0, 1).expand_as(u)
    raise ValueError(f"Unknown mode {mode!r}.")


@torch.no_grad()
def mass_kappa(wk: Tensor, mask: Tensor) -> Tensor:
    """§8.3 MASS 대조: κ = Σ b_j m_j / Σ b_j (b = w_K 를 성분축으로 broadcast), batch 단위 스칼라. Σb = 0 이면 0."""
    b = torch.broadcast_to(wk.detach(), mask.shape).to(mask.dtype)
    den = b.sum()
    return torch.where(den > 0, (b * mask).sum() / den.clamp_min(1e-30), torch.zeros_like(den))


@torch.no_grad()
def shuffle_mask(mask: Tensor, generator: torch.Generator) -> Tensor:
    """§8.3 SHUFFLE 대조: sample 마다 mask 의 공간 위치를 섞는다 (같은 순열을 모든 성분에; band permute 는 별개)."""
    B, D, H, W = mask.shape
    out = torch.empty_like(mask)
    for b in range(B):
        perm = torch.randperm(H * W, generator=generator).to(mask.device)
        out[b] = mask[b].reshape(D, H * W)[:, perm].reshape(D, H, W)
    return out


@torch.no_grad()
def component_weights(s: Tensor, t: Tensor, g: Tensor, tau: Tensor, *, alpha: float, kd_weight: float, eps: float) -> Tensor:
    """§4.6 A-BANDADV / §5.6 B-COMPADV: 성분별 d_j = |t−g|/(|t−g|+τ_j), a_j = [|s−g|−|t−g|]_+/(|s−g|+ε) → w_K,j = β(1−d_j)a_j [B,D,H,W]. hard 는 parent 그대로."""
    _images(s, t, g)
    dt = _dtype(s, t, g)
    et = (t.to(dt) - g.to(dt)).abs(); es = (s.to(dt) - g.to(dt)).abs()
    tau = tau.to(dt).view(1, -1, 1, 1)
    d = et / (et + tau); a = ((es - et).clamp_min(0) / (es + eps)).clamp(max=1.0)
    return kd_weight * (1.0 - d) * a


def masked_l1(s: Tensor, t: Tensor, g: Tensor, wh: Tensor, wk: Tensor, *, mask: Tensor | None = None, risk: Tensor | None = None) -> dict:
    """parent 의 detached w_H·w_K(β 포함) 로 hard 는 그대로, soft 는 gate 를 곱해 다시 계산 (§9.2)."""
    _images(s, t, g)
    shape = (s.shape[0], 1, *s.shape[-2:])
    if wh.shape != shape or wk.shape != shape:
        raise ValueError("wh/wk must have shape [B,1,H,W].")
    if wh.device != s.device or wk.device != s.device:
        raise ValueError("Weight device mismatch.")
    dt = _dtype(s, t, g, wh, wk)
    s, t, g = s.to(dt), t.detach().to(dt), g.detach().to(dt)
    wh, wk = wh.detach().to(dt), wk.detach().to(dt)
    if not bool(torch.isfinite(wh).all() & torch.isfinite(wk).all()):
        raise FloatingPointError("Non-finite weights.")
    if bool((wh < 0).any() | (wk < 0).any()):
        raise ValueError("Nonnegative weights required.")
    gate = torch.ones_like(s)
    for extra in (mask, risk):
        if extra is not None:
            if extra.device != s.device:
                raise ValueError("Gate device mismatch.")
            extra = torch.broadcast_to(extra.detach().to(dt), s.shape)
            if not bool(torch.isfinite(extra).all()) or bool(((extra < 0) | (extra > 1)).any()):
                raise ValueError("Gates must be finite and in [0,1].")
            gate = gate * extra
    hard = (wh * (s - g).abs().mean(1, keepdim=True)).mean()
    soft = (wk * (gate * (s - t).abs()).mean(1, keepdim=True)).mean()
    return {"loss": hard + soft, "hard": hard, "soft": soft, "gate_mean": gate.mean().detach()}


def masked_l1_componentwise(s: Tensor, t: Tensor, g: Tensor, wh: Tensor, wk_comp: Tensor, *, risk: Tensor | None = None) -> dict:
    """BANDADV/COMPADV: hard 는 parent scalar w_H, soft 는 성분별 계수 w_K,j (mask 는 이미 계수에 내포)."""
    _images(s, t, g)
    dt = _dtype(s, t, g, wh, wk_comp)
    s, t, g = s.to(dt), t.detach().to(dt), g.detach().to(dt)
    wk = wk_comp.detach().to(dt)
    if risk is not None:
        wk = wk * torch.broadcast_to(risk.detach().to(dt), wk.shape)
    hard = (wh.detach().to(dt) * (s - g).abs().mean(1, keepdim=True)).mean()
    soft = (wk * (s - t).abs()).mean()
    return {"loss": hard + soft, "hard": hard, "soft": soft, "gate_mean": wk.mean().detach()}


@torch.no_grad()
def routing_stats(s: Tensor, t: Tensor, g: Tensor, mask: Tensor, wk: Tensor) -> dict:
    """§13.2 기전 로그: u/v 0 비율, same-sign 비율, 충돌 성분 비율, 'Teacher 는 좋은데 일부 band 는 나쁜' pixel 비율, soft mass 비율."""
    dt = _dtype(s, t, g); u, v = g.to(dt) - s.to(dt), t.to(dt) - s.to(dt)
    same = ((u > 0) & (v > 0)) | ((u < 0) & (v < 0)); conflict = ((u > 0) & (v < 0)) | ((u < 0) & (v > 0))
    wk_pos = (wk > 0)
    px_good_bad = (wk_pos & (conflict.any(1, keepdim=True))).double().sum() / wk_pos.double().sum().clamp_min(1)
    return dict(u_zero=float((u == 0).double().mean()), v_zero=float((v == 0).double().mean()), same_sign=float(same.double().mean()), conflicting=float(conflict.double().mean()),
                mask_mean=float(mask.mean()), wk_positive_fraction=float(wk_pos.double().mean()), pixel_good_teacher_but_bad_bands=float(px_good_bad),
                soft_mass_ratio=float(mass_kappa(wk, mask)))


# ------------------------------------------------------------------ C: Teacher 출력의 correction 민감도 (§6.2–6.3)
@torch.no_grad()
def teacher_eval(teacher, pan_view: Tensor, ms: Tensor, lpan, delta: Tensor, *, phi: str = 'identity', stat_kind: str | None = None, window: int = 5) -> Tensor:
    """z_T(δ) = Φ(M + F_T([W(P_view, δ), M])) — 원 P_view 에서 각 probe 의 PAN 을 만든다 (누적 warp 금지). Teacher 는 eval·frozen."""
    with torch.autocast(device_type=pan_view.device.type, enabled=False):
        p = warp_pan(pan_view.float(), delta.float())
    ms_base = F.interpolate(ms, scale_factor=4, mode='bicubic')
    sw = torch.ones(p.shape[0], device=p.device, dtype=pan_view.dtype)
    pi = p.to(pan_view.dtype)
    y = ms_base + teacher.backbone(pi, lpan if lpan is not None else pi, ms, sw)
    if phi == 'identity':
        return y.float()
    if phi == 'stat':
        return statistic_map(y.float(), stat_kind, window)
    raise ValueError(phi)


@torch.no_grad()
def fd_jacobian(evaluate: Callable[[Tensor], Tensor], mu: Tensor, *, h: float = .05) -> Tensor:
    """중심 유한차분 [B,D,2,H,W] (axis 0: dy, 1: dx). evaluate(δ) 는 원 P_view 를 F_T 로 다시 렌더링해야 한다."""
    if mu.ndim != 2 or mu.shape[1] != 2 or not bool(torch.isfinite(mu).all()):
        raise ValueError("mu must be finite [B,2], order dy/dx.")
    if not math.isfinite(h) or h <= 0:
        raise ValueError("Positive finite h required.")
    cols = []
    for axis in (0, 1):
        step = torch.zeros_like(mu); step[:, axis] = h
        plus, minus = evaluate(mu.detach() + step), evaluate(mu.detach() - step)
        if plus.ndim != 4 or plus.shape != minus.shape or plus.shape[0] != mu.shape[0]:
            raise ValueError("evaluate must return fixed [B,D,H,W] shape.")
        if not bool(torch.isfinite(plus).all() & torch.isfinite(minus).all()):
            raise FloatingPointError("Non-finite probe outputs.")
        dt = _dtype(plus, minus)
        cols.append((plus.to(dt) - minus.to(dt)) / (2 * h))
    return torch.stack(cols, dim=2)


@torch.no_grad()
def teacher_fd_jacobian(teacher, pan_view, ms, lpan, mu_t, *, h=0.05, phi='identity', stat_kind=None, window=5):
    """J_T [B,D,2,H',W'] (Teacher U-Net 4회 순차 forward, activation 미보존)."""
    return fd_jacobian(lambda d: teacher_eval(teacher, pan_view, ms, lpan, d, phi=phi, stat_kind=stat_kind, window=window), mu_t.detach(), h=h)


@torch.no_grad()
def sens_q(jac: Tensor) -> Tensor:
    """§6.6 q_sens = J_y² + J_x² [B,D,H,W]."""
    return (jac ** 2).sum(dim=2)


@torch.no_grad()
def sens_risk(q: Tensor, s_sens: float) -> Tensor:
    """r_sens = s/(s+q). s_sens ≤ 0 이면 source 무반응 → 감쇠 1 (fallback)."""
    if not (math.isfinite(s_sens) and s_sens > 0):
        return torch.ones_like(q)
    return s_sens / (s_sens + q)


@torch.no_grad()
def diag_risk(jac: Tensor, sigma: Tensor, valid: Tensor, *, s_c: float) -> dict:
    """§6.7 r_C = s_C²/(s_C² + diag(J Σ Jᵀ)); invalid sample 은 r=1 (parent fallback), q_report 는 NaN (진단 표시)."""
    if jac.ndim != 5 or jac.shape[2] != 2:
        raise ValueError("jac must be [B,D,2,H,W].")
    b = jac.shape[0]
    if sigma.shape != (b, 2, 2) or valid.shape != (b,) or valid.dtype != torch.bool:
        raise ValueError("sigma [B,2,2] and bool valid [B] required.")
    if not (jac.device == sigma.device == valid.device):
        raise ValueError("Device mismatch.")
    if not math.isfinite(s_c) or s_c <= 0 or not bool(torch.isfinite(jac).all()):
        raise ValueError("Finite jac and positive finite s_c required.")
    dt = _dtype(jac, sigma)
    j = jac.to(dt)
    safe_sigma = torch.where(valid[:, None, None], sigma.to(dt), torch.zeros_like(sigma, dtype=dt))
    if not bool(torch.isfinite(safe_sigma).all()):
        raise FloatingPointError("Valid covariance contains non-finite entries.")
    if not torch.allclose(safe_sigma, safe_sigma.transpose(-1, -2), atol=1e-8, rtol=1e-5):
        raise ValueError("Symmetric covariance required.")
    safe_sigma = .5 * (safe_sigma + safe_sigma.transpose(-1, -2))
    if bool((torch.linalg.eigvalsh(safe_sigma) < -1e-8).any()):
        raise ValueError("Covariance is not positive semidefinite.")
    q = torch.einsum("bdihw,bij,bdjhw->bdhw", j, safe_sigma, j).clamp_min(0)
    r = s_c ** 2 / (s_c ** 2 + q)
    valid_map = valid[:, None, None, None]
    r = torch.where(valid_map, r, torch.ones_like(r))
    q_report = torch.where(valid_map, q, torch.full_like(q, float("nan")))
    return {"risk": r, "q_report": q_report, "valid": valid.detach()}


def lowrank_quad(d: Tensor, jac: Tensor, sigma: Tensor, *, s_c: float) -> Tensor:
    """§6.8 ℓ_Q = dᵀ W d /(D s_C), W = (I + U Uᵀ)⁻¹, U = J L / s_C (Woodbury 2×2). [B,1,H,W]; J/Σ/U/W detach, d live."""
    if d.ndim != 4 or jac.shape != (d.shape[0], d.shape[1], 2, *d.shape[-2:]):
        raise ValueError("d [B,D,H,W] and matching jac [B,D,2,H,W] required.")
    if sigma.shape != (d.shape[0], 2, 2) or not math.isfinite(s_c) or s_c <= 0:
        raise ValueError("Bad covariance shape/scale.")
    dt = _dtype(d, jac, sigma)
    with torch.no_grad():
        ss = sigma.detach().to(dt)
        if not torch.allclose(ss, ss.transpose(-1, -2), atol=1e-8, rtol=1e-5):
            raise ValueError("Symmetric covariance required.")
        eig, vec = torch.linalg.eigh(ss)
        if not bool(torch.isfinite(eig).all()) or bool((eig < -1e-8).any()):
            raise ValueError("Finite PSD covariance required.")
        root = vec * eig.clamp_min(0).sqrt().unsqueeze(-2)
        jp = jac.detach().to(dt).permute(0, 3, 4, 1, 2)
        u = (jp @ root[:, None, None, :, :]) / s_c
        small = torch.eye(2, device=d.device, dtype=dt) + u.transpose(-1, -2) @ u
    dp = d.to(dt).permute(0, 2, 3, 1).unsqueeze(-1)
    z = u.transpose(-1, -2) @ dp
    energy0 = (dp * dp).sum(dim=(-2, -1))
    correction = (z * torch.linalg.solve(small, z)).sum(dim=(-2, -1))
    energy = energy0 - correction
    if bool((energy.detach() < -1e-5 * energy0.detach().clamp_min(1e-12)).any()):
        raise FloatingPointError("Woodbury subtraction unstable; do not silently clip.")
    return (energy.clamp_min(0) / (d.shape[1] * s_c)).unsqueeze(1)


def iso_quad(d: Tensor, *, s_c: float) -> Tensor:
    """C-QISO: 같은 quadratic soft 에 W = I → ‖d‖²/(D s_C). [B,1,H,W]."""
    dt = _dtype(d)
    return ((d.to(dt) ** 2).sum(dim=1, keepdim=True)) / (d.shape[1] * s_c)


def scalar_quad(d: Tensor, jac: Tensor, sigma: Tensor, *, s_c: float) -> Tensor:
    """C-QSCALAR: W 의 tr(W)/D 만 쓰는 스칼라 감쇠 × ‖d‖²/(D s_C). tr(W) = D − tr((I₂+UᵀU)⁻¹ UᵀU)."""
    dt = _dtype(d, jac, sigma)
    with torch.no_grad():
        ss = sigma.detach().to(dt); eig, vec = torch.linalg.eigh(ss)
        root = vec * eig.clamp_min(0).sqrt().unsqueeze(-2)
        jp = jac.detach().to(dt).permute(0, 3, 4, 1, 2); u = (jp @ root[:, None, None, :, :]) / s_c
        utu = u.transpose(-1, -2) @ u; small = torch.eye(2, device=d.device, dtype=dt) + utu
        tr_w = d.shape[1] - torch.linalg.solve(small, utu).diagonal(dim1=-2, dim2=-1).sum(-1)   # [B,H,W]
        w_scalar = (tr_w / d.shape[1]).unsqueeze(1)
    return w_scalar * iso_quad(d, s_c=s_c)


@torch.no_grad()
def sigma_control(Sig: Tensor, mode: str, generator: torch.Generator | None = None) -> Tensor:
    """§8.3 C 방향 대조: rotate = 고유값 유지·축 90° 회전, shuffle = sample 간 Σ 섞기 (J 는 해당 sample 의 실제 J 유지)."""
    if mode == 'none':
        return Sig
    if mode == 'rotate':
        R = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=Sig.dtype, device=Sig.device)
        return R @ Sig @ R.T
    if mode == 'shuffle':
        perm = torch.randperm(Sig.shape[0], generator=generator).to(Sig.device)
        return Sig[perm]
    raise ValueError(mode)


@torch.no_grad()
def linearization_check(evaluate: Callable[[Tensor], Tensor], mu: Tensor, jac: Tensor, delta: Tensor) -> dict:
    """§6.9-2 국소 선형성: ‖z(μ+δ) − z(μ) − Jδ‖ / ‖z(μ+δ) − z(μ)‖."""
    z0 = evaluate(mu.detach()); z1 = evaluate(mu.detach() + delta)
    pred = torch.einsum('bdihw,bi->bdhw', jac.to(z0.dtype), delta.to(z0.dtype))
    dz = z1 - z0
    return dict(residual_rel=float((dz - pred).norm() / (dz.norm() + 1e-30)), dz_rms=float(dz.pow(2).mean().sqrt()), pred_rms=float(pred.pow(2).mean().sqrt()))
