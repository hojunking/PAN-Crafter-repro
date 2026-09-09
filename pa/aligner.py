"""§3 dual-stem 경량 global shift CNN.

PAN stem(1→16) · MS stem(C→16) → concat 32 → conv s2 32 → conv s2 64 → residual 64 → GAP → 64→32→2.
conv bias 없음 · GN affine · Linear bias · 마지막 Linear 만 0 초기화(시작 Δ̂=(0,0)).
입력은 aligner 용 복사본만 sample·채널별 공간 평균/분산으로 표준화한다(통계 detach, §3.2).
WV3(8 band) params = 105,330 (§3.3 v1 기록과 일치해야 한다 — tools/pa_unit_tests.py 가 확인).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def znorm(x, eps=1e-6):
    """sample·채널별 공간 표준화. 통계에는 gradient 를 주지 않는다 (§3.2)."""
    mu = x.mean(dim=(2, 3), keepdim=True).detach()
    var = x.var(dim=(2, 3), keepdim=True, unbiased=False).detach()
    return (x - mu) / torch.sqrt(var + eps)


def _conv(cin, cout, stride, groups):
    return nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(cin, cout, 3, stride=stride, padding=0, bias=False),
                         nn.GroupNorm(groups, cout), nn.SiLU())


class _Res(nn.Module):
    def __init__(self, c, groups=8):
        super().__init__()
        self.a = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3, bias=False), nn.GroupNorm(groups, c), nn.SiLU())
        self.b = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3, bias=False), nn.GroupNorm(groups, c))

    def forward(self, x):
        return F.silu(x + self.b(self.a(x)))


class PANGlobalAligner(nn.Module):
    def __init__(self, ms_bands=8):
        super().__init__()
        self.pan_stem = _conv(1, 16, 1, 4)
        self.ms_stem = _conv(ms_bands, 16, 1, 4)
        self.joint1 = _conv(32, 32, 2, 8)
        self.joint2 = _conv(32, 64, 2, 8)
        self.res = _Res(64, 8)
        self.fc1 = nn.Linear(64, 32, bias=True)
        self.fc2 = nn.Linear(32, 2, bias=True)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)      # §3.5 identity start

    def forward(self, pan, ms_up):
        """pan [B,1,H,W], ms_up [B,C,H,W] (B0 intensity scale). 반환 Δ̂ [B,2] = (dy,dx), 현재 HR px."""
        p = self.pan_stem(znorm(pan)); m = self.ms_stem(znorm(ms_up))
        x = torch.cat((p, m), dim=1)                     # 같은 위치에서 결합한 뒤 joint conv (§3.3)
        x = self.res(self.joint2(self.joint1(x)))
        x = x.mean(dim=(2, 3))                            # GAP
        return self.fc2(F.silu(self.fc1(x)))
