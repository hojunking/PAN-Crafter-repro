"""리샘플링 원시 연산. 부호 규약은 한 곳에서만 정의한다 (계획 §3.2):

    delta = (dy, dx),  aligned[y, x] = moving[y + dy, x + dx]
    M -> P forward : +delta      P -> M inverse : -delta      delta_HR = 4 * delta_LR

검증은 tools/test_alignment.py (T01~T07). 추정으로 부호를 섞지 않는다.

왜 interp23tap 인가 — 계획 §4 는 "phase-2 bicubic" 을 기본값으로 뒀지만, 데이터셋의
lms 는 DLPan `interp23tap` (23-tap CDF 보간, LR 샘플이 HR 4j+2 에 놓임) 이다.
실측: 세 split 모두 interp23tap 은 ZNCC 1.000000 / MAD 0.000, phase-2 bicubic 은
ZNCC 0.9906(train)·0.9955(RR)·0.9986(FR) — 계획 §4.5 gate(≥0.9999)를 bicubic 으로는
어떤 phase 로도 못 넘는다. 그래서 기본 upsampler 는 interp23tap 이고, shift 는 HR
격자에서 bicubic warp 한 번으로 준다 (α·Δ=0 이면 warp 를 아예 호출하지 않아 P0 와
비트 동일 — T06). 계획 원안은 `kind="bicubic_phase2"` 로 남겨 둔다.
"""
import math

import torch
import torch.nn.functional as F

# DLPan wald_utilities.interp23tap 의 계수 (×2 는 원본 그대로)
_CDF23 = [0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0,
          -0.005192756653, 0, 0.000807762146, 0, -0.000060081482]


def _cdf23_kernel(dtype, device):
    c = [2.0 * v for v in _CDF23]
    return torch.tensor(list(reversed(c[1:])) + c, dtype=dtype, device=device)   # 23 taps


def interp23tap(x, ratio=4):
    """wald_utilities.interp23tap 의 torch 이식 (float64 로 1e-9 이내 일치, T02).

    stage 0 은 [1::2,1::2] 에, 이후 stage 는 [::2,::2] 에 zero-stuffing 하고 23-tap 을
    양 축에 circular('wrap') 로 건다. LR 샘플 j 는 HR 4j+2 에 놓인다 (phase 2).
    """
    assert ratio >= 2 and (ratio & (ratio - 1)) == 0, "2 의 거듭제곱 배율만"
    B, C, _, _ = x.shape
    k = _cdf23_kernel(x.dtype, x.device)
    kw = k.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    kh = k.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    for z in range(int(round(math.log2(ratio)))):
        H2, W2 = 2 * x.shape[-2], 2 * x.shape[-1]
        up = x.new_zeros(B, C, H2, W2)
        if z == 0:
            up[..., 1::2, 1::2] = x
        else:
            up[..., ::2, ::2] = x
        up = F.conv2d(F.pad(up, (11, 11, 0, 0), mode="circular"), kw, groups=C)
        up = F.conv2d(F.pad(up, (0, 0, 11, 11), mode="circular"), kh, groups=C)
        x = up
    return x


def hr_grid(B, H, W, dtype, device):
    yy = torch.arange(H, dtype=dtype, device=device).view(1, H, 1).expand(B, H, W)
    xx = torch.arange(W, dtype=dtype, device=device).view(1, 1, W).expand(B, H, W)
    return yy, xx


def _sample(src, src_y, src_x, padding_mode="border"):
    """out[b,:,v,u] = src[b,:, src_y[b,v,u], src_x[b,v,u]]  (bicubic; 정수 좌표는 정확히 원값)."""
    _, _, H, W = src.shape
    gy = 2.0 * src_y / max(H - 1, 1) - 1.0
    gx = 2.0 * src_x / max(W - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(src, grid, mode="bicubic", padding_mode=padding_mode, align_corners=True)


def warp_hr(x, delta_hr, padding_mode="border"):
    """out[y, x] = src[y + dy, x + dx].  delta_hr: [B,2] = (dy, dx), HR 픽셀 단위."""
    B, _, H, W = x.shape
    yy, xx = hr_grid(B, H, W, x.dtype, x.device)
    dy = delta_hr[:, 0].to(x.dtype).view(B, 1, 1)
    dx = delta_hr[:, 1].to(x.dtype).view(B, 1, 1)
    return _sample(x, yy + dy, xx + dx, padding_mode)


def _alpha_col(alpha, B, dtype, device):
    if torch.is_tensor(alpha):
        return alpha.to(dtype=dtype, device=device).reshape(-1, 1).expand(B, 1) if alpha.numel() in (1, B) \
            else alpha.reshape(B, 1)
    return torch.full((B, 1), float(alpha), dtype=dtype, device=device)


def phase_shift_upsample(lr, delta_lr, alpha, phase=2.0, scale=4, padding_mode="border"):
    """계획 §4.3 원안: out[v,u] = lr[(v-phase)/scale + α·dy, (u-phase)/scale + α·dx] — bicubic 한 번."""
    B, _, H, W = lr.shape
    yy, xx = hr_grid(B, scale * H, scale * W, lr.dtype, lr.device)
    a = _alpha_col(alpha, B, lr.dtype, lr.device).view(B, 1, 1)
    dy = delta_lr[:, 0].to(lr.dtype).view(B, 1, 1)
    dx = delta_lr[:, 1].to(lr.dtype).view(B, 1, 1)
    return _sample(lr, (yy - phase) / scale + a * dy, (xx - phase) / scale + a * dx, padding_mode)


def upsample_shift(lr, delta_lr, alpha, kind="interp23tap", scale=4, phase=2.0, padding_mode="border"):
    """case 공통 진입점. 결과 격자는 P-frame 쪽으로 α·Δ 만큼 옮긴 MS 의 HR 버전.

    kind="interp23tap": interp23tap(lr) 뒤 HR warp(scale·α·Δ). α·Δ 가 상수 0 이면 warp 를
                        호출하지 않는다 (P0 와 비트 동일). Δ 가 grad 를 요구하면 0 이어도
                        warp 를 거쳐 ShiftNet 으로 gradient 가 흐른다.
    kind="bicubic_phase2": 계획 원안 (단일 grid_sample).
    """
    if kind == "bicubic_phase2":
        return phase_shift_upsample(lr, delta_lr, alpha, phase, scale, padding_mode)
    assert kind == "interp23tap", kind
    up = interp23tap(lr, scale)
    B = lr.shape[0]
    d = delta_lr.to(lr.dtype) * _alpha_col(alpha, B, lr.dtype, lr.device)
    if not d.requires_grad and float(d.detach().abs().max()) == 0.0:
        return up
    return warp_hr(up, scale * d, padding_mode)


def border_mask(delta_lr, H, W, scale=4, extra=2):
    """계획 §7.2: 표본별 margin m = ceil(scale·max(|dy|,|dx|)) + extra. [B,1,H,W] {0,1}."""
    B = delta_lr.shape[0]
    m = torch.ceil(scale * delta_lr.detach().abs().max(dim=1).values) + extra
    yy, xx = hr_grid(B, H, W, delta_lr.dtype, delta_lr.device)
    mm = m.view(B, 1, 1)
    valid = (yy >= mm) & (yy < H - mm) & (xx >= mm) & (xx < W - mm)
    return valid.unsqueeze(1).to(delta_lr.dtype)


def masked_l1(pred, gt, mask, eps=1e-6):
    """계획 §7.2: loss = Σ|pred-gt|·mask / (Σmask · C + eps)."""
    C = pred.shape[1]
    return ((pred - gt).abs() * mask).sum() / (mask.sum() * C + eps)


def transform_delta(delta, hflip, vflip, rot):
    """영상에 hflip -> vflip -> np.rot90(k=rot, CCW) 순으로 augmentation 이 걸렸을 때의 (dy,dx).

    hflip: dx -> -dx / vflip: dy -> -dy / rot90 CCW 1회: (dy,dx) -> (-dx, dy).
    합성 순서는 영상과 같다 (계획 §6). 검증: T05 impulse.
    """
    dy, dx = delta[:, 0].clone(), delta[:, 1].clone()
    hflip, vflip, rot = hflip.to(delta.device), vflip.to(delta.device), rot.to(delta.device)
    dx = torch.where(hflip.bool(), -dx, dx)
    dy = torch.where(vflip.bool(), -dy, dy)
    for k in (1, 2, 3):
        sel = rot >= k
        ndy, ndx = -dx, dy
        dy = torch.where(sel, ndy, dy)
        dx = torch.where(sel, ndx, dx)
    return torch.stack([dy, dx], dim=1)
