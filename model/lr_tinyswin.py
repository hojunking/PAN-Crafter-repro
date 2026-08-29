# --------------------------------------------------------
# LR-TinySwin — 저해상도 grid 에서 Swin 으로 융합하는 초소형 pan-sharpening.
# 계획: research_log/2026-08-29_swin-24h-plan-v2.md §3 (원안 §5).
#
# LR-Fuse(conv 계열, ERGAS 4.0+ 기각)와 같은 "전 연산 1/16 면적" 패러다임의
# Swin 판이다 — 전역(window) attention 이면 다른가를 반증하는 목적.
#
# 인터페이스는 기존과 동일: forward(pan, lpan, ms, s) -> 8ch 잔차.
#   - 잔차 기준선은 trainer 가 밖에서 더한다 (res: True)
#   - dual MARs 성립 조건: mode 신호가 모델 안 어딘가에 주입돼야 한다.
#     SwinBlock 은 표준형(무조건화)이므로, 입력 conv 직후와 Swin group 직후에
#     ModeModulation(Eq 6, 각 2xC params) 을 넣어 mode 를 준다.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.pancrafter import zero_module
from model.pancrafter_paper import ModeModulation
from model.swin import swin_blocks


class LRTinySwin(nn.Module):
    """PixelUnshuffle -> conv -> (residual Swin group) -> conv -> PixelShuffle.

    in_mode
        "paper"    : unshuffle(PAN) + MS                          (16+8 = 24ch)
        "released" : + LPAN(저주파) + unshuffle(PAN-↑LPAN)(고주파) (41ch)
    """

    def __init__(self, in_channels=1, out_channels=8, hidden_size=64, swin_depth=2,
                 scale=4, num_heads=4, window_size=8, mlp_ratio=2.0, in_mode="paper"):
        super().__init__()
        self.in_mode = in_mode
        self.scale = scale
        C = hidden_size
        n_pan = in_channels * scale * scale
        n_in = (n_pan + out_channels) if in_mode == "paper" \
            else (2 * n_pan + in_channels + out_channels)
        self.unshuffle = nn.PixelUnshuffle(scale)
        self.input = nn.Conv2d(n_in, C, 3, padding=1)
        self.mod_in = ModeModulation(C)
        self.blocks = swin_blocks(swin_depth, C, num_heads, window_size, mlp_ratio)
        self.res_conv = nn.Conv2d(C, C, 3, padding=1)   # residual Swin group 마감 conv
        self.mod_out = ModeModulation(C)
        self.head = nn.Sequential(
            nn.SiLU(),
            zero_module(nn.Conv2d(C, out_channels * scale * scale, 3, padding=1)))
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, pan, lpan, ms, s):
        p = self.unshuffle(pan)
        if self.in_mode == "paper":
            x = torch.cat((p, ms), dim=1)
        else:
            lpan_u = F.interpolate(lpan, scale_factor=self.scale, mode="bicubic")
            x = torch.cat((p, lpan, self.unshuffle(pan - lpan_u), ms), dim=1)
        x = self.input(x)
        g, b = self.mod_in(s)
        x = x * (1 + g) + b
        h = x
        for blk in self.blocks:
            h = blk(h)
        x = x + self.res_conv(h)                        # residual Swin group
        g, b = self.mod_out(s)
        x = x * (1 + g) + b
        return self.shuffle(self.head(x))
