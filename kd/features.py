# --------------------------------------------------------
# feature tap / feature KD / uncertainty head — 명세 §4·§8·§20 K5.
# 기존 모델 코드를 수정하지 않고 forward hook 으로 feature 를 뽑는다.
#
# tap 이름 -> PANCrafterPaper 서브모듈 (c6/R4 topology 동일 전제):
#   bottleneck_h4 : middle[-1] 출력   (초기 feature KD 는 이것 하나만, §4)
#   dec_h         : decoder1[-1] 출력 (uncertainty head 입력, §8.1)
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd.ops import mean_normalize  # noqa: F401  (호출부 편의 재수출)

TAPS = {"bottleneck_h4": lambda m: m.middle[-1],
        "dec_h": lambda m: m.decoder1[-1]}


class FeatureTap:
    """지정 서브모듈의 마지막 출력을 저장하는 hook. remove() 로 해제."""

    def __init__(self, model, names=("bottleneck_h4",)):
        self.out = {}
        self._handles = []
        base = model.module if hasattr(model, "module") else model
        for n in names:
            mod = TAPS[n](base)
            self._handles.append(mod.register_forward_hook(self._make(n)))

    def _make(self, name):
        def hook(_m, _inp, out):
            self.out[name] = out
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


class FeatureProj(nn.Module):
    """feature KD 용 1×1 사영 (§20 K5): teacher/student 를 공통 차원으로."""

    def __init__(self, t_ch, s_ch, common=64):
        super().__init__()
        self.proj_t = nn.Conv2d(t_ch, common, 1)
        self.proj_s = nn.Conv2d(s_ch, common, 1)

    def forward(self, f_s, f_t_detached, weight=None):
        zs = F.normalize(self.proj_s(f_s), dim=1)
        zt = F.normalize(self.proj_t(f_t_detached), dim=1)   # 입력은 detach 됨 — 사영만 학습
        d = (zs - zt).abs()
        if weight is not None:
            if weight.shape[-2:] != d.shape[-2:]:
                weight = F.interpolate(weight, size=d.shape[-2:], mode="bilinear")
            d = d * weight
        return d.mean()


class UncertaintyHead(nn.Module):
    """pixel 당 1ch uncertainty. Conv3x3(C->C/4) -> SiLU -> Conv1x1(C/4->1).

    out 모드 (구조·state_dict 키는 동일 — 기존 T1/T2 checkpoint 그대로 로드된다):
      "softplus" : theta = softplus(z) + eps  (분산 그 자체, 기존 KD 캠페인)
      "logvar"   : s = z                      (log sigma^2, s2 계획 §2 — 무제약이라
                   head-only calibration 에서 수치적으로 안정)
    """

    def __init__(self, channels, out="softplus"):
        super().__init__()
        assert out in ("softplus", "logvar")
        self.out = out
        self.net = nn.Sequential(
            nn.Conv2d(channels, max(8, channels // 4), 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(max(8, channels // 4), 1, 1))

    def forward(self, feat):
        z = self.net(feat)
        return z if self.out == "logvar" else F.softplus(z) + 1e-6


class WithUncertainty(nn.Module):
    """base 모델 + uncertainty head 래퍼.

    forward 계약(residual 반환)은 기존과 동일해 validate/test 경로가 그대로 돈다.
    theta 는 forward 직후 theta() 로 읽는다 (마지막 decoder feature 에서 계산).
    accelerate 가 이 래퍼 전체를 저장하므로 head 도 checkpoint 에 포함된다.
    """

    def __init__(self, base, channels, head_out="softplus"):
        super().__init__()
        self.base = base
        self.head = UncertaintyHead(channels, out=head_out)
        self._tap = FeatureTap(base, names=("dec_h",))

    def forward(self, pan, lpan, ms, s):
        return self.base(pan, lpan, ms, s)

    def theta(self):
        """head 출력 그대로. head_out='logvar' 면 log sigma^2 다."""
        return self.head(self._tap.out["dec_h"])
