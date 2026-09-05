"""손실·가중 (계획 §6–§8)."""
import torch
import torch.nn.functional as F


def gt_residual_variance(gt, lms, window=5):
    """V_raw = mean_c[AvgPool(R²) − AvgPool(R)²], R = GT − LMS. [B,1,H,W]"""
    r = gt - lms
    p = window // 2
    m = F.avg_pool2d(r, window, stride=1, padding=p, count_include_pad=False)
    m2 = F.avg_pool2d(r * r, window, stride=1, padding=p, count_include_pad=False)
    return (m2 - m * m).clamp_min(0).mean(dim=1, keepdim=True)


def percentile_normalize(x, q_lo, q_hi):
    """clip((x − Q_lo)/(Q_hi − Q_lo + ε), 0, 1). Q 는 training set 에서 고정한 상수."""
    return ((x - q_lo) / (q_hi - q_lo + 1e-6)).clamp(0, 1)


def variance_weight(v_gt, alpha=1.0):
    """w_V = (1 + α V) / mean(1 + α V)  — 표본별 평균 1."""
    w = 1.0 + alpha * v_gt
    return w / w.mean(dim=(1, 2, 3), keepdim=True)


def routed_losses(r_s, r_gt, r_t, u_t, w_v=None):
    """L_hard = <w_V (1+U_T) |R_S − R_GT|>,  L_soft = <w_V (1−U_T) |R_S − R_T|>. u_t/r_t/w_v 는 detach 된 상태를 기대."""
    w = torch.ones_like(u_t) if w_v is None else w_v
    hard = (w * (1.0 + u_t) * (r_s - r_gt).abs()).mean()
    soft = (w * (1.0 - u_t) * (r_s - r_t).abs()).mean()
    return hard, soft


def shift_kd_loss(delta_s, conf_s, delta_t, conf_t, conf_ratio=0.1):
    """L_vec = <c_T SmoothL1(δ_S − δ_T)>,  L_conf = <(c_S − c_T)²>,  L_shift = L_vec + 0.1 L_conf."""
    vec = (conf_t.unsqueeze(1) * F.smooth_l1_loss(delta_s, delta_t, reduction="none")).mean()
    cf = ((conf_s - conf_t) ** 2).mean()
    return vec + conf_ratio * cf, vec, cf


def teacher_forcing_eta(step, s0=5000, s1=20000):
    """η(t): 0–5K 1.0, 5K–20K 선형 1→0, 이후 0."""
    if step <= s0:
        return 1.0
    if step >= s1:
        return 0.0
    return 1.0 - (step - s0) / float(s1 - s0)


def warp_effect_loss(pan_hf_s, pan_hf_t, conf_t, v_gt):
    """L_warp = <c_T V_GT |H(W(P,4δ̂_S)) − H(W(P,4δ̂_T))|> (§7.4, 조건부)."""
    return (conf_t.view(-1, 1, 1, 1) * v_gt * (pan_hf_s - pan_hf_t).abs()).mean()
