"""이동량(alignment) KD — 계획 §11: Teacher correction μ_T 의 방향별 신뢰도 Π_T 로 가중한 mean-KD, 그리고 covariance 자체의 KD(G5).

  d_μ = μ_S − sg(μ_T),  L_G = ½·mean_b d_μᵀ M_b d_μ  (invalid sample 은 weight 0, batch 전체 평균)
  G1: M = k0·I (k0 = median_train tr(Π_T)/2) · G2: M = (tr Π_T/2)·I · G3: M = Π_T · G4: M = diag(Π_T) · G-STRUCT: M = J_Y/k_struct (GT 구조 텐서)
  G5: KL( N(μ_S, Σ_S) ‖ sg N(μ_T, Σ_T) ),  Σ = L Lᵀ + eps I (CovHead, 64-d GAP feature → 3)

Π_T 의 출처 (모두 frozen Teacher 에서 계산, 학습되는 head 가 아니다):
  eq_closure   (§11.4): K 개 알려진 probe ε_k 로 closure r_k = μ_ε + ε − μ_0, Q = mean_k r_k r_kᵀ (= C + b bᵀ),  Π = (Q + σ_min² I)⁻¹ (고유값 cap 1/σ_min²).
                        반응 없는 aligner 는 r_k = ε_k → Π ≈ 작다(신뢰 낮음). 완벽 반응은 r=0 → cap. 절대 정합 posterior 가 아니다.
  geo_curvature(§11.2): A3 residual r(δ) 의 Jacobian J = ∂r/∂δ|μ_T (중심 유한차분 h), H = JᵀJ, Σ ≈ s²(H+γI)⁻¹ + σ_min² I.
                        rank 규칙(§11.3): 고유값 < max(τ_abs, τ_rel·λ_max) 인 방향은 precision 0 (damping 으로 생긴 확신 금지). s² 는 [s2_lo, s2_hi] 로 clamp.
  struct       (§11.7): J_Y = mean_{p,c} ∇Y_c ∇Y_cᵀ — GT 만 필요. Σ 의 정답이 아니라 중요도 weighting 대조.
좌표: μ = (dy, dx) HR px. Scharr 는 (gx, gy) 를 내므로 (dy,dx) 순서로 재배열한다. 모든 2×2 대수는 float64."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pa.losses import geometry_residual, scharr
from pa.offset import predict_c
from pa.warp import warp_pan, support_margin_ok

GEOM_MODES = ('G0', 'G1', 'G2', 'G3', 'G4', 'G5', 'G-STRUCT')
COV_SOURCES = ('none', 'eq_closure', 'geo_curvature', 'struct')


# ------------------------------------------------------------------ 2×2 대수
def eigh2(M):
    """대칭 [B,2,2] → (evals 오름차순 [B,2], evecs 열 [B,2,2]), float64."""
    return torch.linalg.eigh(M.double())


def from_eig(evals, V):
    return V @ torch.diag_embed(evals) @ V.transpose(-1, -2)


def precision_from_moment(Q, sigma_min):
    """eq_closure: Π = (Q + σ² I)⁻¹ — 고유값이 자동으로 1/σ² 이하. 반환 (Π [B,2,2] f64, info)."""
    ev, V = eigh2(Q); s2 = float(sigma_min) ** 2
    pi = 1.0 / (ev.clamp_min(0.0) + s2)
    return from_eig(pi, V), dict(evals_Q=ev, precision_evals=pi, cap_hit=(pi >= 0.99 / s2), rank=torch.full_like(ev[:, 0], 2))


def precision_from_curvature(H, s2, gamma, sigma_min, tau_abs, tau_rel, s2_lo, s2_hi):
    """geo_curvature: Σ_dir = s²/(λ+γ) + σ_min²;  λ < max(τ_abs, τ_rel·λ_max) 인 방향은 precision 0 (§11.3-1). 반환 (Π, info)."""
    ev, V = eigh2(H); lmax = ev[:, 1:2]
    thr = torch.maximum(torch.full_like(lmax, float(tau_abs)), float(tau_rel) * lmax)
    valid = ev >= thr
    s2c = s2.double().clamp(float(s2_lo), float(s2_hi))[:, None]
    var = s2c / (ev.clamp_min(0.0) + float(gamma)) + float(sigma_min) ** 2
    pi = torch.where(valid, 1.0 / var, torch.zeros_like(var))
    return from_eig(pi, V), dict(evals_H=ev, precision_evals=pi, valid_dirs=valid, rank=valid.sum(1), s2_clamped=s2c[:, 0], cap_hit=(pi >= 0.99 / float(sigma_min) ** 2))


# ------------------------------------------------------------------ 출처별 Σ/Π
@torch.no_grad()
def sigma_from_precision(Pi: Tensor, info: dict | None = None) -> tuple[Tensor, Tensor]:
    """Π_T → Σ_Δ,T (HR px²). 어느 방향이든 precision 0(정보 없음) 이면 invalid — null direction 을 covariance 0 으로 바꾸지 않는다 (addendum §6.5, 계획 §11.3-3)."""
    ev, V = torch.linalg.eigh(Pi.double())
    valid = (ev > 0).all(dim=1)
    inv = torch.where(ev > 0, 1.0 / ev.clamp_min(1e-300), torch.zeros_like(ev))
    Sig = V @ torch.diag_embed(inv) @ V.transpose(-1, -2)
    return Sig, valid


def structure_tensor_mean(gt):
    """§11.7 J_Y = mean_{p,c} ∇Y_c ∇Y_cᵀ, (dy,dx) 순서, [B,2,2] f64. Scharr 미분 경계 1px 제외."""
    gx, gy = scharr(gt.float()); s = (slice(None), slice(None), slice(1, -1), slice(1, -1))
    gx, gy = gx[s].double(), gy[s].double()
    jyy = (gy * gy).mean(dim=(1, 2, 3)); jxx = (gx * gx).mean(dim=(1, 2, 3)); jyx = (gy * gx).mean(dim=(1, 2, 3))
    return torch.stack((torch.stack((jyy, jyx), 1), torch.stack((jyx, jxx), 1)), 1)


def probe_set(K=16, radius_hr=1.0):
    """고정 probe (dy,dx) [K,2]: K/2 방향 × 반경 {R/2, R}. 학습 jitter 가 아니라 진단 격자 (§11.4)."""
    n = max(1, K // 2); out = []
    for r in (radius_hr / 2.0, radius_hr):
        for i in range(n):
            a = 2.0 * math.pi * i / n; out.append((r * math.sin(a), r * math.cos(a)))
    return torch.tensor(out[:K], dtype=torch.float32)


@torch.no_grad()
def eq_closure_moment(aligner, pan, ms_base, probes, margin, guard_margin=4):
    """§11.4: μ_0 = A^V(P,M), μ_k = A^V(W(P,ε_k),M), r_k = μ_k + ε_k − μ_0. 반환 dict(Q [B,2,2] f64, C, b, mu0, R [B,K,2], valid [B] bool)."""
    B, _, H, W = pan.shape
    with torch.autocast(device_type=pan.device.type, enabled=False):
        p = pan.float(); mu0 = predict_c(aligner, p, ms_base, margin)
        R = []
        for k in range(probes.shape[0]):
            eps = probes[k].to(p.device).expand(B, 2)
            R.append(predict_c(aligner, warp_pan(p, eps), ms_base, margin) + eps - mu0)
        R = torch.stack(R, 1).double()                                   # [B,K,2]
    b = R.mean(1); C = ((R - b[:, None]).transpose(1, 2) @ (R - b[:, None])) / R.shape[1]
    Q = (R.transpose(1, 2) @ R) / R.shape[1]
    ok = torch.stack([support_margin_ok(H, W, probes[k].to(p.device).expand(B, 2), max(int(margin), guard_margin)) for k in range(probes.shape[0])], 1).all(1)
    return dict(Q=Q, C=C, b=b, mu0=mu0, R=R, valid=ok)


def geo_curvature(pan, gt, mu_T, sigma, margin, h=0.05):
    """§11.2: 중심 유한차분으로 J_i = (r(μ+h e_i) − r(μ−h e_i)) / 2h, H = JᵀJ. Teacher μ_T 는 detach 된 leaf.
    반환 dict(H [B,2,2] f64, s2 [B], r0 [B,N], g_fd [B,2] = Jᵀ r0 (autograd 대조용), r0_norm2 [B])."""
    with torch.no_grad(), torch.autocast(device_type=pan.device.type, enabled=False):
        p = pan.float(); mu = mu_T.detach().float()
        r0, info0 = geometry_residual(warp_pan(p, mu), p, gt, sigma, margin)
        J = []
        for i in range(2):
            e = torch.zeros_like(mu); e[:, i] = h
            rp, _ = geometry_residual(warp_pan(p, mu + e), p, gt, sigma, margin)
            rm, _ = geometry_residual(warp_pan(p, mu - e), p, gt, sigma, margin)
            J.append(((rp - rm) / (2.0 * h)).double())
        H = torch.stack([torch.stack([(J[i] * J[j]).sum(1) for j in range(2)], 1) for i in range(2)], 1)
        g_fd = torch.stack([(J[i] * r0.double()).sum(1) for i in range(2)], 1)
    return dict(H=H, s2=info0['s2'].double(), r0=r0, g_fd=g_fd, r0_norm2=(r0.double() ** 2).sum(1))


def geo_autograd_grad(pan, gt, mu_T, sigma, margin):
    """gate C01: ∇_μ ½||r(μ)||² 를 autograd 로 (bicubic grid_sample 1차 미분) — g_fd = J_fdᵀ r0 와 대조."""
    with torch.enable_grad(), torch.autocast(device_type=pan.device.type, enabled=False):
        mu = mu_T.detach().float().clone().requires_grad_(True); p = pan.float()
        r, _ = geometry_residual(warp_pan(p, mu), p, gt, sigma, margin)
        L = 0.5 * (r ** 2).sum()
        return torch.autograd.grad(L, mu)[0].double()


# ------------------------------------------------------------------ loss
def kd_matrix(Pi, mode, k0=1.0):
    """mode 별 가중 행렬 M [B,2,2] f64 (Π 는 detach 된 값)."""
    B = Pi.shape[0]; I = torch.eye(2, dtype=torch.float64, device=Pi.device).expand(B, 2, 2)
    if mode == 'G1':
        return float(k0) * I
    if mode == 'G2':
        return (Pi.diagonal(dim1=1, dim2=2).sum(1) / 2.0)[:, None, None] * I
    if mode in ('G3', 'G-STRUCT'):
        return Pi
    if mode == 'G4':
        return torch.diag_embed(Pi.diagonal(dim1=1, dim2=2))
    raise ValueError(mode)


def mean_kd_loss(mu_s, mu_t, M, valid=None):
    """½·mean_b [valid_b · d_μᵀ M_b d_μ], d_μ = μ_S − sg(μ_T). invalid 는 weight 0 이고 분모는 batch 전체 (§11.3-5)."""
    d = (mu_s.double() - mu_t.detach().double())
    q = torch.einsum('bi,bij,bj->b', d, M.detach(), d)
    if valid is not None:
        q = q * valid.double()
    return 0.5 * q.mean(), q


class CovHead(nn.Module):
    """§11.6 Σ = L Lᵀ + eps I, L = [[softplus(a),0],[c, softplus(b)]] — 64-d aligner GAP feature → 3. 초기 Σ = I."""
    def __init__(self, in_dim=64, eps=1e-4):
        super().__init__()
        self.fc = nn.Linear(in_dim, 3); self.eps = float(eps)
        nn.init.zeros_(self.fc.weight)
        with torch.no_grad():
            self.fc.bias.copy_(torch.tensor([math.log(math.e - 1.0), math.log(math.e - 1.0), 0.0]))   # softplus(x)=1

    def forward(self, feat):
        o = self.fc(feat.float()).double()
        a, b, c = F.softplus(o[:, 0]), F.softplus(o[:, 1]), o[:, 2]
        z = torch.zeros_like(a)
        L = torch.stack((torch.stack((a, z), 1), torch.stack((c, b), 1)), 1)
        return L @ L.transpose(1, 2) + self.eps * torch.eye(2, dtype=torch.float64, device=feat.device)


def gaussian_kl(mu_s, Sig_s, mu_t, Sig_t):
    """KL( N(μ_S,Σ_S) ‖ N(μ_T,Σ_T) ) = ½[ dᵀΣ_T⁻¹d + tr(Σ_T⁻¹Σ_S) − 2 + log det Σ_T − log det Σ_S ], f64, [B]. Teacher 항은 detach."""
    d = (mu_s.double() - mu_t.detach().double())[:, :, None]
    St = Sig_t.detach().double(); Ss = Sig_s.double()
    Lt = torch.linalg.cholesky(St)
    maha = torch.cholesky_solve(d, Lt); maha = (d * maha).sum(dim=(1, 2))
    tr = torch.cholesky_solve(Ss, Lt).diagonal(dim1=1, dim2=2).sum(1)
    logdet_t = 2.0 * torch.log(Lt.diagonal(dim1=1, dim2=2)).sum(1)
    logdet_s = torch.logdet(Ss)
    return 0.5 * (maha + tr - 2.0 + logdet_t - logdet_s)


def precision_stats(Pi, info=None):
    ev = torch.linalg.eigvalsh(Pi.double())
    out = dict(pi_eval_min_median=float(ev[:, 0].median()), pi_eval_max_median=float(ev[:, 1].median()), pi_trace_median=float(ev.sum(1).median() ), anisotropy_median=float(((ev[:, 1] + 1e-30) / (ev[:, 0] + 1e-30)).median()))
    if info is not None:
        if 'rank' in info:
            out['rank_deficient_fraction'] = float((info['rank'] < 2).double().mean())
        if 'cap_hit' in info:
            out['cap_hit_fraction'] = float(info['cap_hit'].double().mean())
    return out
