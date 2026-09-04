"""GlobalShiftNet (계획 §15). 입력은 raw intensity 가 아니라 audit 와 같은 구조맵 2ch:
robust_norm(Scharr|MTF↓PAN|), robust_norm(Scharr|mean_band(MS)|). 출력 (dy, dx), LR 픽셀,
tanh 로 |·| <= max_shift_lr. head 는 0 초기화라 시작 예측은 정확히 0 이다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_SCHARR_X = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]])


def scharr_mag_t(x):
    """x: [B,1,H,W] -> Scharr 크기 [B,1,H,W] (replicate pad)."""
    kx = _SCHARR_X.to(x.dtype).to(x.device).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    gx, gy = F.conv2d(xp, kx), F.conv2d(xp, ky)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def robust_norm_t(v):
    """표본별 median/MAD 정규화 (estimator.robust_norm 과 동일 정의)."""
    B = v.shape[0]
    flat = v.reshape(B, -1)
    med = flat.median(dim=1).values.view(B, 1, 1, 1)
    mad = (flat - med.view(B, 1)).abs().median(dim=1).values.view(B, 1, 1, 1) + 1e-6
    return (v - med) / mad


def structural_input(lpan, ms):
    """lpan: [B,1,h,w] (MTF↓PAN, phase 2), ms: [B,C,h,w] -> [B,2,h,w]. gradient 는 끊는다
    (ShiftNet 입력은 관측치이지 학습 대상이 아니다)."""
    with torch.no_grad():
        g_pan = robust_norm_t(scharr_mag_t(lpan))
        g_ms = robust_norm_t(scharr_mag_t(ms.mean(dim=1, keepdim=True)))
    return torch.cat([g_pan, g_ms], dim=1)


class GlobalShiftNet(nn.Module):
    def __init__(self, max_shift_lr=1.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(32, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.max_shift_lr = float(max_shift_lr)

    def forward(self, x):
        z = self.body(x).flatten(1)
        return self.max_shift_lr * torch.tanh(self.head(z))     # [B,2] = (dy, dx)
