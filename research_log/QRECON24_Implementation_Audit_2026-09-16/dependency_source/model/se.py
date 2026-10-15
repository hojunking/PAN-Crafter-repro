# --------------------------------------------------------
# 표준 Squeeze-and-Excitation gate — SE ablation 용.
# 계획: research_log/se_ablation_two_experiments.md §2.2
#   GAP -> 1x1 (C->C/r) -> ReLU -> 1x1 (C/r->C) -> Sigmoid -> channel 곱
#   r=8 (C=96 에서 hidden 12 — r=16 은 hidden 6 이라 배제, 계획 §2.2)
# --------------------------------------------------------

import torch.nn as nn


class SEGate(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid())

    def forward(self, x):
        return x * self.net(x)

    def gate(self, x):
        """진단용 — gate 값 자체를 돌려준다 (tools/analyze_se_gates.py)."""
        return self.net(x)
