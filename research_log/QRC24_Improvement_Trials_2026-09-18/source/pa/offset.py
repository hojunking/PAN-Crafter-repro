"""PO10 — PAN 추가 변위와 offset consistency (research_log/PAN_OffsetConsistency_10GPUh_W96_D124_2026-09-10.md §3–§6, §8).

corrupted step:  P_ε = W(P, ε) (no-grad) · ĉ0 = A^V(P, M) · ĉε = A^V(P_ε, M) · P̃ε = W(P_ε, ĉε) (두 번째 raster warp, gradient 유지) · Ŷε = M + Fθ([P̃ε, M])
native step:     ĉ0 = A^V(P, M) · P̃0 = W(P, ĉ0) · Ŷ0 = M + Fθ([P̃0, M])
offset loss:     |ĉε + ε − ref|, ref = sg(ĉ0) (N2_SG) 또는 ĉ0 (N3_NOSG); N1 은 계수 0 (진단용 값만)
A^V: aligner 는 고정 내부 crop(margin m_A = 4·ceil((R+2)/4)) 만 본다 — padding 단서 배제 (§5). U-Net·GT·MS base 는 자르지 않는다.
ε: 2-D 원판 면적 균등, |ε|₂ ≤ R, 평균 0, 축별 sd R/2 (§3.2). 전용 CPU generator (§7-8).
"""
import math

import torch
import torch.nn.functional as F

from pa.aligner import PANGlobalAligner  # noqa: F401  (인터페이스 참조)
from pa.warp import warp_pan

CASES = ("N1", "N2_SG", "N3_NOSG")


def sample_offsets(batch, radius_hr, generator):
    """원판 면적 균등 (radius = R·sqrt(u)), (dy, dx). CPU generator → 호출부가 device 로 옮긴다."""
    if batch <= 0 or not math.isfinite(radius_hr) or radius_hr <= 0:
        raise ValueError("batch and observed radius must be positive")
    u = torch.rand(batch, 2, generator=generator, dtype=torch.float32)
    radius = radius_hr * u[:, 0].sqrt(); angle = 2.0 * math.pi * u[:, 1]
    return torch.stack((radius * angle.sin(), radius * angle.cos()), 1)


def aligner_margin(radius_hr):
    """m_A = 4·ceil((R + 2) / 4). R=1 → 4 (64² → 56² view, 512² → 504²)."""
    return 4 * math.ceil((radius_hr + 2.0) / 4.0)


def valid_view(x, margin):
    if margin < 0 or min(x.shape[-2:]) <= 2 * margin + 4:
        raise ValueError("invalid fixed aligner crop")
    return x if margin == 0 else x[..., margin:-margin, margin:-margin]


def predict_c(aligner, pan, ms_base, margin):
    """A^V: crop 뒤 aligner (aligner 내부에서 crop 된 view 로 z-score, §2.2). FP32."""
    with torch.autocast(device_type=pan.device.type, enabled=False):
        c = aligner(valid_view(pan.float(), margin), valid_view(ms_base.float(), margin))
    if c.shape != (pan.shape[0], 2):
        raise ValueError("aligner must return [B,2]")
    if not bool(torch.isfinite(c).all()):
        raise FloatingPointError("non-finite aligner output")
    return c


def offset_loss(c_eps, c0, eps, stop_reference):
    """mean |ĉε + ε − ref| (HR px). sample 별 |ε| 로 나누지 않는다 (§4.7)."""
    reference = c0.detach() if stop_reference else c0
    return (c_eps + eps - reference).abs().mean()


def lambda_off(update_index, lam_max, ramp):
    """ramp ≤ 0 이면 처음부터 lam_max (NF16 §4.2: 0→0.01/5K ramp 없이 즉시 활성)."""
    return lam_max if ramp <= 0 else lam_max * min(1.0, (update_index + 1) / float(ramp))


def po_step(model, pan, ms, lpan, gt, *, case, update_index, radius_hr, generator, lam_max=0.01, ramp=5000, margin=None):
    """한 optimizer update 의 forward + loss. 반환 (loss, info). U-Net 은 한 번만 돈다 (§4.2)."""
    if case not in CASES:
        raise ValueError(case)
    margin = aligner_margin(radius_hr) if margin is None else margin
    corrupt = (update_index % 2 == 1)                                   # 0 native, 1 corrupt, ... (§4.1)
    ms_base = F.interpolate(ms, scale_factor=4, mode="bicubic")         # B0/A1 과 같은 bicubic base
    A, bb = model.aligner, model.backbone
    if corrupt:
        eps = sample_offsets(pan.shape[0], radius_hr, generator).to(pan.device)
        with torch.no_grad():
            p_eps = warp_pan(pan, eps)                                  # 첫 raster warp (학습 대상 아님)
        if case == "N3_NOSG":
            c0 = predict_c(A, pan, ms_base, margin)
        else:
            with torch.no_grad():
                c0 = predict_c(A, pan, ms_base, margin)
        c = predict_c(A, p_eps, ms_base, margin)
        with torch.autocast(device_type=pan.device.type, enabled=False):
            p_al = warp_pan(p_eps, c)                                   # 두 번째 raster warp (gradient 유지)
        l_off = offset_loss(c, c0, eps, stop_reference=(case != "N3_NOSG"))
    else:
        eps = torch.zeros(pan.shape[0], 2, device=pan.device)
        c = predict_c(A, pan, ms_base, margin); c0 = c
        with torch.autocast(device_type=pan.device.type, enabled=False):
            p_al = warp_pan(pan, c)
        l_off = c.new_zeros(())
    sw = torch.ones(pan.shape[0], device=pan.device, dtype=pan.dtype)
    res = bb(p_al.to(pan.dtype), lpan, ms, sw)                          # in_mode paper: cat(P̃, ↑MS)
    pred = ms_base + res
    l_rec = (pred - gt).abs().mean()                                    # 전 영역 L1 (§6.3, moving mask 없음)
    w = 0.0 if (case == "N1" or not corrupt) else lambda_off(update_index, lam_max, ramp)
    total = l_rec + w * l_off
    return total, dict(rec=l_rec, off=l_off, weight=w, c=c, c0=c0, eps=eps, pred=pred, pan_aligned=p_al, corrupt=corrupt, p_eps=(p_eps if corrupt else None))
