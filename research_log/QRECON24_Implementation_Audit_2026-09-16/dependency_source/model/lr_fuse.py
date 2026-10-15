# --------------------------------------------------------
# LR-Fuse — 고해상도 backbone 을 없앤 초경량 pan-sharpening.
# 24h 탐색 계획 §6: research_log/2026-08-28_architecture-search-24h-plan.md
#
# PAN 을 PixelUnshuffle(4) 로 MS grid 에 접는다. 이 변환은 전단사(무손실)라
# 공간 정보 손실 없이 모든 conv 를 1/16 면적에서 돌릴 수 있다.
#
# 인터페이스는 PANCrafterPaper 와 동일하다: forward(pan, lpan, ms, s) -> 8ch 잔차.
#   - 잔차 기준선(↑MS 또는 mode 별 base)은 trainer 가 밖에서 더한다 (res: True)
#   - MARs dual mode 도 그대로 성립한다: 출력이 항상 8ch 이고(PAN mode 는 PAN 을
#     8밴드로 broadcast 한 것이 target), mode 는 ResBlock 의 ModeModulation(Eq 6)
#     으로 주입된다. batch 복제·loss 구성 모두 train.py 수정 없이 작동한다.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.pancrafter import zero_module
from model.pancrafter_paper import ResBlock, _norm


class LRFuse(nn.Module):
    """저해상도 융합 backbone.

        PAN (B,1,4H,4W) --PixelUnshuffle x4--> (B,16,H,W)
        MS  (B,8,H,W)   ----------------------^ concat
        Conv 3x3 -> ResBlock x n_blocks  (width C, LN+SiLU, attention 없음)
        head 3x3 -> (B, 8*16, H, W) --PixelShuffle x4--> (B,8,4H,4W) 잔차

    in_mode
        "paper"    : unshuffle(PAN) + MS                          (16+8 = 24ch)
        "released" : + LPAN(저주파, LR 원본) + unshuffle(PAN-↑LPAN)(고주파)
                     (16+1+16+8 = 41ch) — 11ch 입력의 저주파/고주파 분해를
                     LR grid 로 옮긴 것. ↑LPAN 을 다시 unshuffle 하면 bicubic 의
                     매끄러움 때문에 채널 간 중복이 커서 LPAN 원본 1ch 로 대신한다.
    """

    def __init__(self, in_channels=1, out_channels=8, hidden_size=64, n_blocks=6,
                 scale=4, dropout=0.0, norm="ln", in_mode="paper"):
        super().__init__()
        self.in_mode = in_mode
        self.scale = scale
        C = hidden_size
        n_pan = in_channels * scale * scale
        n_in = (n_pan + out_channels) if in_mode == "paper" \
            else (2 * n_pan + in_channels + out_channels)
        self.unshuffle = nn.PixelUnshuffle(scale)
        self.input = nn.Conv2d(n_in, C, 3, padding=1)
        self.blocks = nn.ModuleList(
            [ResBlock(C, dropout, norm=norm) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            _norm(norm, C), nn.SiLU(),
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
        for b in self.blocks:
            x = b(x, s)
        return self.shuffle(self.head(x))
