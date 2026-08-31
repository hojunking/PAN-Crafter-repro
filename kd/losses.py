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
