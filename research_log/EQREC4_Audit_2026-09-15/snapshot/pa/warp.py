"""§4 PAN warp: P̃[b,y,x] = P[b, y+dy, x+dx] (sampling 위치 부호), bicubic · border · align_corners=False.

- 학습 중 Δ=0 이어도 sampler 를 우회하지 않는다 (§4.2). grid 는 Δ 로 미분된다.
- FP32 로 계산한다 (autocast 는 호출부가 끈다; 여기서는 dtype 을 float32 로 맞춘다).
- support mask 는 영상값이 아니라 **clamp 이전의 원본 sampling 좌표**와 4-tap bicubic support 로 만든다 (§10.5).
"""
import torch
import torch.nn.functional as F


def warp_pan(pan, delta):
    """pan [B,1,H,W], delta [B,2]=(dy,dx) HR px. 반환 float32 [B,1,H,W] (입력이 float64 면 float64)."""
    B, _, H, W = pan.shape
    p = pan if pan.dtype == torch.float64 else pan.float(); d = delta.to(p.dtype)
    ys = torch.arange(H, device=pan.device, dtype=p.dtype)
    xs = torch.arange(W, device=pan.device, dtype=p.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")                      # [H,W]
    qy = yy[None] + d[:, 0].view(B, 1, 1)                                # 원본 sampling 좌표 (clamp 전)
    qx = xx[None] + d[:, 1].view(B, 1, 1)
    gx = 2.0 * (qx + 0.5) / W - 1.0                                      # align_corners=False 규약 (§4.1)
    gy = 2.0 * (qy + 0.5) / H - 1.0
    grid = torch.stack((gx, gy), dim=-1)                                 # 마지막 축 (x, y)
    return F.grid_sample(p, grid, mode="bicubic", padding_mode="border", align_corners=False)


@torch.no_grad()
def warp_support_mask(H, W, delta, taps=4):
    """[B,1,H,W] bool. 출력 (y,x) 의 bicubic 4-tap 원본 이웃 floor(q)-1..floor(q)+2 가 전부 영상 안이면 True.
    정수 위치에서 가중치 0 인 tap 도 제외하지 않는 보수적 규칙 (§10.5)."""
    B = delta.shape[0]
    d = delta.float()
    ys = torch.arange(H, device=delta.device, dtype=torch.float32)
    xs = torch.arange(W, device=delta.device, dtype=torch.float32)
    qy = ys[None] + d[:, 0:1]; qx = xs[None] + d[:, 1:2]                 # [B,H], [B,W]
    fy, fx = torch.floor(qy), torch.floor(qx)
    oky = (fy - 1 >= 0) & (fy + 2 <= H - 1)
    okx = (fx - 1 >= 0) & (fx + 2 <= W - 1)
    return (oky[:, :, None] & okx[:, None, :])[:, None]


def support_margin_ok(H, W, delta, margin):
    """고정 interior [margin:H-margin, margin:W-margin] 의 모든 sampling 이웃이 영상 안인가 (§6.2 guard). [B] bool."""
    m = warp_support_mask(H, W, delta)
    return m[:, :, margin:H - margin, margin:W - margin].flatten(1).all(dim=1)


@torch.no_grad()
def two_stage_support_mask(H, W, eps, c, taps=4):
    """W(W(P, ε), c) 의 출력 (y,x) 가 실제 관측만 읽는가 (검토 지적: 두 sampling 단계 각각의 support 추적).
    1단계: P_ε 의 유효 행/열 = warp_support_mask(ε). 2단계: 출력 (y,x) 가 읽는 P_ε 의 4-tap 이웃 floor(q)-1..floor(q)+2 가 전부 1단계 유효 영역 안이어야 한다.
    보수적으로 1단계 유효 영역을 [r0, r1]×[c0, c1] 직사각형(유효 행/열의 연속 구간)으로 잡는다. 반환 [B,1,H,W] bool."""
    m1 = warp_support_mask(H, W, eps, taps)                                  # [B,1,H,W]
    B = eps.shape[0]; out = torch.zeros(B, 1, H, W, dtype=torch.bool, device=eps.device)
    d = c.float()
    ys = torch.arange(H, device=eps.device, dtype=torch.float32); xs = torch.arange(W, device=eps.device, dtype=torch.float32)
    for b in range(B):
        rows = torch.nonzero(m1[b, 0].any(1)).flatten(); cols = torch.nonzero(m1[b, 0].any(0)).flatten()
        if rows.numel() == 0 or cols.numel() == 0:
            continue
        r0, r1, c0, c1 = int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())
        fy = torch.floor(ys + d[b, 0]); fx = torch.floor(xs + d[b, 1])
        oky = (fy - 1 >= r0) & (fy + 2 <= r1); okx = (fx - 1 >= c0) & (fx + 2 <= c1)
        out[b, 0] = oky[:, None] & okx[None, :]
    return out
