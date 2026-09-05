"""variant 별 학습/추론 forward — Accelerator 없이 테스트할 수 있게 trainer 에서 분리했다 (T04–T09, T14–T24).

배치 배치: dual MARs 는 [PAN mode(switch 0) | MS mode(switch 1)], j4 는 [PAN clean | MS clean | MS jitter].
모든 변형에서 잔차 base 는 clean ms_base, GT·출력은 M-frame, PAN mode 출력은 inverse/warp/mask 없음.
"""
import torch
import torch.nn.functional as F

from sr.jitter import translate_hr, gaussian_blur_depthwise
from sr.pan_align import split_first_conv, edge_weight


def build_inputs(pan, lpan, ms):
    """원 코드와 같은 bicubic (F.interpolate, align_corners=False). (ms_base, lpan_u, pan_hf)"""
    I = lambda t: F.interpolate(t, scale_factor=4, mode="bicubic")
    ms_base, lpan_u = I(ms), I(lpan)
    return ms_base, lpan_u, pan - lpan_u


def x11(pan, lpan_u, pan_hf, ms_cond):
    return torch.cat((pan, lpan_u, pan_hf, ms_cond), dim=1)      # 순서 고정 (§3.2)


def g1_ms_features(backbone, correlator, x_ms, pan_hf, eps_g, scale=1.0):
    """MS mode 의 first-conv 특징 F_M + F̃_P (§10.2, §11). 반환 (f0, info)."""
    f_p, f_m = split_first_conv(backbone.input, x_ms)
    f_p_syn = translate_hr(f_p, eps_g)
    info = correlator(f_p_syn, f_m, edge_weight(pan_hf))
    return f_m + correlator.apply(f_p_syn, info["delta"], info["gate"], scale), info


def sr_forward(model, variant, pan, lpan, ms, gt, eps=None, eps_g=None, w_off=1.0, lam_cons=0.0,
               blur_sigma=None, g1_scale=1.0):
    """학습 forward. model: SRModel(backbone, correlator). 반환 dict(loss, loss_ms, loss_pan, loss_cons,
    loss_shift, y, ms_base, cond_ms, cond_pan, res_*, info)."""
    bb, corr = model.backbone, model.correlator
    B, C = ms.shape[0], ms.shape[1]
    dev, dt = ms.device, ms.dtype
    ms_base, lpan_u, pan_hf = build_inputs(pan, lpan, ms)
    lpan_rep, pan_rep = lpan_u.repeat(1, C, 1, 1), pan.repeat(1, C, 1, 1)
    zeros = torch.zeros((), device=dev, dtype=dt)
    out = dict(ms_base=ms_base, loss_cons=zeros, loss_shift=zeros, info=None)
    if variant in ("j1", "j2", "j3"):
        cond = gaussian_blur_depthwise(ms_base, blur_sigma) if variant == "j3" else translate_hr(ms_base, eps)
        cond_pan = ms_base if variant == "j2" else cond
        x2 = torch.cat([x11(pan, lpan_u, pan_hf, cond_pan), x11(pan, lpan_u, pan_hf, cond)], 0)
        sw = torch.cat([torch.zeros(B, device=dev, dtype=dt), torch.ones(B, device=dev, dtype=dt)])
        res = bb(None, None, None, sw, x_in=x2)
        r_pan, r_ms = res[:B], res[B:]
        out.update(cond_ms=cond, cond_pan=cond_pan, res_pan=r_pan, res_ms=r_ms)
    elif variant == "j4":
        cond = translate_hr(ms_base, eps)
        xc, xj = x11(pan, lpan_u, pan_hf, ms_base), x11(pan, lpan_u, pan_hf, cond)
        sw = torch.cat([torch.zeros(B, device=dev, dtype=dt), torch.ones(2 * B, device=dev, dtype=dt)])
        res = bb(None, None, None, sw, x_in=torch.cat([xc, xc, xj], 0))
        r_pan, r0, r_eps = res[:B], res[B:2 * B], res[2 * B:]
        out.update(cond_ms=cond, cond_pan=ms_base, res_pan=r_pan, res_ms=r0, res_eps=r_eps,
                   loss_cons=(r_eps - r0.detach()).abs().mean())
    elif variant == "g1":
        xc = x11(pan, lpan_u, pan_hf, ms_base)
        if eps_g is None:
            eps_g = torch.zeros(B, 2, device=dev, dtype=dt)
        f_ms, info = g1_ms_features(bb, corr, xc, pan_hf, eps_g, g1_scale)
        f_pan = bb.input(xc)                                        # PAN mode: F_P + F_M 그대로 (§10.3), aligner 호출 없음
        sw = torch.cat([torch.zeros(B, device=dev, dtype=dt), torch.ones(B, device=dev, dtype=dt)])
        res = bb(None, None, None, sw, f_in=torch.cat([f_pan, f_ms], 0))
        r_pan, r_ms = res[:B], res[B:]
        out.update(cond_ms=ms_base, cond_pan=ms_base, res_pan=r_pan, res_ms=r_ms, info=info, eps_g=eps_g,
                   loss_shift=F.smooth_l1_loss(info["delta"], -eps_g))
    else:
        raise ValueError(variant)
    p_hat = lpan_rep + out["res_pan"]                                # PAN mode: warp/inverse/mask 없음 (T09)
    loss_pan = (pan_rep - p_hat).abs().mean() * w_off
    if variant == "j4":
        y0, ye = ms_base + out["res_ms"], ms_base + out["res_eps"]
        loss_ms = 0.5 * (gt - y0).abs().mean() + 0.5 * (gt - ye).abs().mean()
        y = y0
    else:
        y = ms_base + out["res_ms"]
        loss_ms = (gt - y).abs().mean()
    loss = loss_pan + loss_ms + lam_cons * out["loss_cons"]
    out.update(y=y, p_hat=p_hat, loss_pan=loss_pan, loss_ms=loss_ms, loss=loss)
    return out


@torch.no_grad()
def sr_infer(model, variant, pan, lpan, ms, eps=None, blur_sigma=None, g1_scale=1.0, wrong_sign=False):
    """MS mode 추론. eps=None 이면 jitter 0 (조건 = ms_base 그 객체, T06). 반환 dict(y, ms_base, cond, info)."""
    bb, corr = model.backbone, model.correlator
    B = ms.shape[0]
    ms_base, lpan_u, pan_hf = build_inputs(pan, lpan, ms)
    sw = torch.ones(B, device=ms.device, dtype=ms.dtype)
    info = None
    if variant == "g1":
        xc = x11(pan, lpan_u, pan_hf, ms_base)
        f_p, f_m = split_first_conv(bb.input, xc)
        info = corr(f_p, f_m, edge_weight(pan_hf))
        delta = -info["delta"] if wrong_sign else info["delta"]
        res = bb(None, None, None, sw, f_in=f_m + corr.apply(f_p, delta, info["gate"], g1_scale))
        cond = ms_base
    else:
        cond = ms_base
        if eps is not None:
            cond = translate_hr(ms_base, eps)
        elif variant == "j3" and blur_sigma:
            cond = gaussian_blur_depthwise(ms_base, blur_sigma) if False else ms_base   # 추론은 clean (§5.1 inference_enabled=false)
        res = bb(None, None, None, sw, x_in=x11(pan, lpan_u, pan_hf, cond))
    return dict(y=ms_base + res, ms_base=ms_base, cond=cond, info=info, pan=pan, ms=ms)
