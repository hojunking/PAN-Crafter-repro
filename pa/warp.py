"""§4 PAN warp: P̃[b,y,x] = P[b, y+dy, x+dx] (sampling 위치 부호), bicubic · border · align_corners=False.

- 학습 중 Δ=0 이어도 sampler 를 우회하지 않는다 (§4.2). grid 는 Δ 로 미분된다.
- FP32 로 계산한다 (autocast 는 호출부가 끈다; 여기서는 dtype 을 float32 로 맞춘다).
- support mask 는 영상값이 아니라 **clamp 이전의 원본 sampling 좌표**와 4-tap bicubic support 로 만든다 (§10.5).
"""
import torch
import torch.nn.functional as F


def warp_pan(pan, delta):
    """pan [B,1,H,W], delta [B,2]=(dy,dx) HR px. 반환 float32 [B,1,H,W]."""
    B, _, H, W = pan.shape
    p = pan.float(); d = delta.float()
    ys = torch.arange(H, device=pan.device, dtype=torch.float32)
    xs = torch.arange(W, device=pan.device, dtype=torch.float32)
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
