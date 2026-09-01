# --------------------------------------------------------
# KD·mutual loss 모듈 — 명세 §7·§8·§11·§20.
# 모든 loss 는 (scalar, diagnostics dict) 를 돌려준다 (명세 §5 의 축약형).
# 상대 peer / teacher target 은 호출부에서 반드시 detach 해 넘긴다.
# --------------------------------------------------------

import torch
import torch.nn.functional as F

from kd.ops import build_shift_candidates, mean_normalize, robust_normalize_01


def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps).mean()


def to_nonneg(x):
    """[-1,1] 정규화 공간 -> 비음수 반사도 공간 근사 ((x+1)/2, 하한 0).

    SAM 은 affine offset 에 불변하지 않다 — feeder 가 [-1,1] 로 정규화한 값을
    그대로 각도 계산에 넣으면 실제 spectral angle 이 아니다. **모든 SAM 계열
    계산은 이 변환을 거친 값으로 한다** (L1 항은 affine 공간 무관이라 그대로).
    """
    return ((x + 1.0) * 0.5).clamp_min(1e-6)


def sam_per_candidate(pred_lr, candidates, eps=1e-8):
    """spectral angle: pred (B,C,h,w) vs candidates (B,C,K,h,w) -> (B,K,h,w).

    입력은 [-1,1] 공간이어도 된다 — 내부에서 to_nonneg 로 복원한다.
    """
    p = to_nonneg(pred_lr).unsqueeze(2)
    c = to_nonneg(candidates)
    dot = (p * c).sum(dim=1)
    n1 = p.norm(dim=1)
    n2 = c.norm(dim=1)
    cos = (dot / (n1 * n2 + eps)).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos)


def sis_loss(pred_lr, lrms, radius=1, mode="shared_vector", eta_sam=0.1):
    """SiS (§7): degradation 된 예측을 shifted LRMS 후보와 비교, 최소 채택.

    pred_lr 은 이미 MTFDownsampler 를 통과한 (B,C,h,w). lrms 는 detach 대상.
    radius=0 이면 관례적 L1 과 동일해야 한다 (unit test 로 보장).
    반환 diagnostics: center_shift_ratio, mean_offset_magnitude, boundary_ratio.
    """
    r = radius
    lrms = lrms.detach()
    if r == 0:
        return F.l1_loss(pred_lr, lrms), {"center_shift_ratio": 1.0,
                                          "mean_offset_magnitude": 0.0,
                                          "boundary_ratio": 0.0}
    cands = build_shift_candidates(lrms, r)                    # (B,C,K,h,w)
    K = cands.shape[2]
    if mode == "bandwise":
        cost = (pred_lr.unsqueeze(2) - cands).abs()            # (B,C,K,h,w)
        min_cost, idx = cost.min(dim=2)                        # (B,C,h,w)
    elif mode == "shared_vector":
        l1 = (pred_lr.unsqueeze(2) - cands).abs().mean(dim=1)  # (B,K,h,w)
        cost = l1 + eta_sam * sam_per_candidate(pred_lr, cands)
        min_cost, idx = cost.min(dim=1)                        # (B,h,w)
    else:
        raise ValueError(f"sis mode: {mode}")
    inner = min_cost[..., r:-r, r:-r] if min_cost.dim() == 3 else min_cost[..., r:-r, r:-r]
    loss = inner.mean()
    with torch.no_grad():
        side = 2 * r + 1
        center = (K - 1) // 2
        dy = (idx // side).float() - r
        dx = (idx % side).float() - r
        mag = torch.sqrt(dy ** 2 + dx ** 2)
        diag = {"center_shift_ratio": (idx == center).float().mean().item(),
                "mean_offset_magnitude": mag.mean().item(),
                "boundary_ratio": ((dy.abs() == r) | (dx.abs() == r)).float().mean().item()}
    return loss, diag


def edge_loss(pred_hrms, pan, grad_op):
    """SIPSA 취지의 절대 gradient 크기 비교 (§6.2). luminance = band 평균."""
    lum = pred_hrms.mean(dim=1, keepdim=True)
    g_pred = grad_op(lum)
    with torch.no_grad():
        g_pan = grad_op(pan)
    return F.l1_loss(g_pred, g_pan), {"edge_pred_mean": g_pred.mean().item()}


def uncertainty_nll(pred, gt, theta):
    """U-Know 식 teacher uncertainty loss (§8.2). theta: (B,1,H,W) > 0."""
    e = (pred - gt).abs().mean(dim=1, keepdim=True)
    loss = (e / (2 * theta) + 0.5 * torch.log(theta)).mean()
    return loss, {"theta_mean": theta.mean().item(), "err_mean": e.mean().item()}


def uknow_weights(theta, mode="robust_normalized", tau=1.0, alpha_u=1.0):
    """hard/soft weight map (§8.4). theta 는 detach 상태를 기대한다. 평균 1 정규화."""
    theta = theta.detach()
    if mode == "paper_raw":
        w_hard = tau + theta
        w_soft = (tau - theta).clamp_min(0.0)
    elif mode == "robust_normalized":
        u = robust_normalize_01(theta)
        w_hard = 1.0 + alpha_u * u
        w_soft = 1.0 - u
    else:
        raise ValueError(mode)
    return mean_normalize(w_hard), mean_normalize(w_soft)


def weighted_l1(pred, target, weight=None):
    d = (pred - target).abs()
    if weight is not None:
        d = d * weight
    return d.mean()


def spectral_kd_loss(pred_lr_s, pred_lr_t, weight=None, eta_sam=0.1):
    """LR scale spectral KD (§20 K1-B): L1 + SAM. teacher 는 detach 해 넘길 것.

    SAM 항은 to_nonneg 로 반사도 공간을 복원해 계산한다 (L1 은 affine 무관).
    """
    l1 = weighted_l1(pred_lr_s, pred_lr_t, weight)
    s, t = to_nonneg(pred_lr_s), to_nonneg(pred_lr_t)
    dot = (s * t).sum(dim=1)
    cos = (dot / (s.norm(dim=1) * t.norm(dim=1) + 1e-8)).clamp(-1 + 1e-7, 1 - 1e-7)
    sam = torch.acos(cos).mean()
    return l1 + eta_sam * sam, {"kd_l1": l1.item(), "kd_sam": sam.item()}


# ---------------------------------------------------------------- mutual (§11)

def mutual_residual(res_self, res_peer_detached):
    """vanilla residual mutual (M1). peer target 은 detach 상태."""
    return charbonnier(res_self - res_peer_detached)


def mutual_spectral(pred_lr_self, pred_lr_peer_detached, eta_sam=0.1):
    """spectral component mutual (M3, SiS peer -> spatial peer)."""
    loss, _ = spectral_kd_loss(pred_lr_self, pred_lr_peer_detached, eta_sam=eta_sam)
    return loss


def mutual_edge(pred_self, pred_peer_detached, grad_op):
    """spatial component mutual (M3, edge peer -> spectral peer)."""
    g_s = grad_op(pred_self.mean(dim=1, keepdim=True))
    with torch.no_grad():
        g_p = grad_op(pred_peer_detached.mean(dim=1, keepdim=True))
    return F.l1_loss(g_s, g_p)


# ------------------------------------------------- s2 계획 (2026-09-01) 전용

def local_error_map(pred, gt, k=3):
    """e_loc = A_k( (1/C)Σ_c (Y_c − μ_c)² )  — s2 계획 §2.

    픽셀 절대오차가 아니라 **국소 평균 제곱오차**다. head 가 단일 픽셀 노이즈가
    아니라 영역 난이도를 학습하게 한다.
    """
    se = (pred - gt).pow(2).mean(dim=1, keepdim=True)
    return F.avg_pool2d(se, k, stride=1, padding=k // 2, count_include_pad=False)


def logvar_nll(e_loc, s):
    """L = 0.5·exp(−s)·e_loc + 0.5·s,  s = log σ² (s2 계획 §2).

    e_loc 은 detach 된 상태를 기대한다 — head-only calibration 에서 mean 은 고정이다.
    """
    loss = (0.5 * torch.exp(-s) * e_loc + 0.5 * s).mean()
    return loss, {"s_mean": s.mean().item(), "e_loc_mean": e_loc.mean().item()}


def uknow_weights_fixed(s, q05, q95, soft_floor=0.05, alpha_u=1.0):
    """고정 train-set 분위수로 정규화한 hard/soft weight (s2 계획 §4).

    u = clip((s − q05)/(q95 − q05), 0, 1)
    w_hard = MeanNorm(1 + α·u)          — teacher 가 못 미더운 곳은 GT 를 더 본다
    w_soft = MeanNorm(max(1 − u, floor)) — teacher 가 자신 있는 곳은 teacher 를 따른다

    배치마다 분위수를 다시 재는 방식(uknow_weights)과 달리 **전 실행에서 같은
    스케일**을 쓴다 — λ 스윕에서 배치 구성이 가중치 분포를 흔들지 않게.
    """
    s = s.detach()
    u = ((s - q05) / (q95 - q05 + 1e-6)).clamp(0, 1)
    w_hard = mean_normalize(1.0 + alpha_u * u)
    w_soft = mean_normalize((1.0 - u).clamp_min(soft_floor))
    return w_hard, w_soft, u


def gtvar_loss(res_s, res_gt, kappa, beta=0.1):
    """L_GTVar = SmoothL1( Ṽ(R_S), sg(Ṽ(R_GT)) ) — s2 계획 §6.

    R = (출력 − ↑MS) 잔차. 다중 스케일 국소 분산을 κ 로 squash 해 비교한다.
    GT 쪽은 detach — student 의 detail energy 를 GT 쪽으로 끌어올린다.
    """
    from kd.ops import multiscale_variance, squash_variance
    v_s = squash_variance(multiscale_variance(res_s), kappa)
    with torch.no_grad():
        v_g = squash_variance(multiscale_variance(res_gt), kappa)
    loss = F.smooth_l1_loss(v_s, v_g, beta=beta)
    with torch.no_grad():
        a = v_s.flatten().float(); b = v_g.flatten().float()
        corr = float(((a - a.mean()) * (b - b.mean())).mean()
                     / (a.std() * b.std() + 1e-8))
    return loss, {"v_s_mean": v_s.mean().item(), "v_gt_mean": v_g.mean().item(),
                  "v_s_std": v_s.std().item(), "v_corr": corr}
