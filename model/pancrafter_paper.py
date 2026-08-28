# --------------------------------------------------------
# PAN-Crafter (논문 충실 재구성) — 원 논문:
#   PAN-Crafter: Learning Modality-Consistent Alignment for PAN-Sharpening
#   Copyright (c) 2025 Jeonghyeok Do, Sungpyo Kim, Geunhyuk Youk, Jaehyup Lee, Munchurl Kim
#   MIT License, 비상업 연구·교육 목적 한정 (LICENSE 참고)
#
# 이 파일은 저자 배포 코드(model/pancrafter.py)가 아니라 **논문 본문·Figure 3 서술**을
# 그대로 옮긴 것이다. 배포 코드와의 차이와 그 근거는 아래 표에 정리했다.
#
#   항목                 논문                              배포 코드
#   ------------------- --------------------------------- ---------------------------
#   spatial scale        3 (Fig 3)                         4
#   Down/UpConv          2 / 2 (Fig 3)                     3 / 3
#   AttnBlock            3 (Fig 3)                         5
#   AttnBlock 배치       "low- and mid-resolution stages"  H/2 에도 배치
#                        "high-res stages use only ResBlock"
#   mode modulation      "gamma_ms, beta_ms, gamma_pan,    mode token + block별
#                         beta_pan in R^C are learnable     Linear(128, 2C)
#                         parameters" (Eq 6)
#   local attention k    k = 3 (전역)                       bottleneck 만 k = 1
#   입력                 concat(I_pan, I_lrms)             PAN, up(LPAN), PAN-up(LPAN), up(MS)
#
# 이 구성의 파라미터 수는 depth 총 12개 기준 **7.1728 M** 로 논문 주장 7.17 M 과
# +0.04% 이내로 일치한다. 배포 코드는 9.969 M 이다.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.pancrafter import CMAAA, DownConv, UpConv, GroupNorm32, zero_module


class ModeModulation(nn.Module):
    """논문 Eq (6): mode 별 gamma, beta 를 직접 학습한다.

    배포 코드는 mode token 을 block 마다 Linear(emb, 2C) 로 사영해 만든다
    (블록당 33,024 params). 논문 서술대로면 블록당 2 x 2 x C = 512 params 다.
    """

    def __init__(self, channels, n_modes=2):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(n_modes, channels))
        self.beta = nn.Parameter(torch.zeros(n_modes, channels))

    def forward(self, s):
        i = s.long()
        return self.gamma[i][:, :, None, None], self.beta[i][:, :, None, None]


class ChannelLayerNorm(nn.Module):
    """공간 위치마다 채널축으로 정규화하는 LayerNorm (DiT/ConvNeXt 방식).

    논문 Eq (5): x <- Conv(SiLU(LN(x))). 배포 코드는 여기에 GroupNorm32 를 쓴다.
    파라미터 수는 둘 다 2C 로 같아서 params 대조로는 이 차이가 드러나지 않는다.
    """

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def _norm(kind, channels):
    if kind == "ln":
        return ChannelLayerNorm(channels)          # 논문 서술
    if kind == "gn":
        return GroupNorm32(32, channels)           # 배포 코드
    raise ValueError(f"norm 은 'ln' 또는 'gn' 이어야 한다: {kind}")


class ResBlock(nn.Module):
    """논문 Eq (5)/(6).

        x <- Conv(SiLU(LN(x)))
        x <- x + Conv(SiLU(Modulate(LN(x); mode)))
    """

    def __init__(self, channels, dropout=0.0, out_channels=None, norm="ln"):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            _norm(norm, channels), nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1))
        self.mod = ModeModulation(self.out_channels)
        self.out_layers = nn.Sequential(
            _norm(norm, self.out_channels), nn.SiLU(), nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)))
        # 배포본 ResBlock 과 동일하게 use_conv=False 경로(1x1)를 쓴다.
        self.skip_connection = (nn.Identity() if self.out_channels == channels
                                else nn.Conv2d(channels, self.out_channels, 1))

    def forward(self, x, s):
        h = self.in_layers(x)
        gamma, beta = self.mod(s)
        h = self.out_layers[0](h) * (1 + gamma) + beta
        h = self.out_layers[1:](h)
        return self.skip_connection(x) + h


class AttnBlock(nn.Module):
    """CM3A + MLP. alpha(결합 가중)도 논문 Eq (8) 대로 mode 별 직접 학습이다."""

    def __init__(self, hidden_size, num_heads=8, pan_channel=1, ms_channel=8,
                 mlp_ratio=4.0, ks=3, ka=3, pan_branch=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = CMAAA(hidden_size, num_heads, pan_channel=pan_channel,
                          ms_channel=ms_channel, pan_ks=ks, ms_ks=ks, ka=ka,
                          fix_key_alias=True,          # 논문 Eq (10)/(11) 대로
                          pan_branch=pan_branch)       # False 면 MS-only local self-attn
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        h = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, h), nn.GELU(approximate="tanh"),
                                 nn.Linear(h, hidden_size))
        self.alpha = ModeModulation(hidden_size)       # alpha1(ms), alpha2(pan)

    def forward(self, x, ms, lpan, pan, s):
        B, C, H, W = x.shape
        N = H * W
        a_ms, a_pan = self.alpha(s)
        xf = x.permute(0, 2, 3, 1).reshape(B, N, -1)
        x_temp = self.norm1(xf).reshape(B, H, W, C).permute(0, 3, 1, 2)
        x_pan, x_ms = self.attn(x_temp, ms, lpan, pan, s)
        xf = xf + (a_ms.squeeze(-1).squeeze(-1).unsqueeze(1)
                   * x_ms.permute(0, 2, 3, 1).reshape(B, N, -1)) \
                + (a_pan.squeeze(-1).squeeze(-1).unsqueeze(1)
                   * x_pan.permute(0, 2, 3, 1).reshape(B, N, -1))
        xf = xf + self.mlp(self.norm2(xf))
        return xf.reshape(B, H, W, C).permute(0, 3, 1, 2)


class PANCrafterPaper(nn.Module):
    """논문 Figure 3 의 3-scale U-Net.

        full-res : ResBlock only                      (high-resolution stage)
          v Down
        H/2      : ResBlock + AttnBlock               (mid)
          v Down
        H/4      : ResBlock + AttnBlock  (bottleneck) (low)
          ^ Up
        H/2      : ResBlock + AttnBlock               (mid)
          ^ Up
        full-res : ResBlock only
    """

    def __init__(self, in_channels=1, out_channels=8, hidden_size=128,
                 depth=(2, 2, 4), dropout=0.0, num_heads=8, mlp_ratio=4.0,
                 ka=3, ks=3, n_attn=3, norm="ln", in_mode="paper",
                 attn_locations=None, cm3a_pan_branch=True, dec_depth=None):
        super().__init__()
        C = hidden_size
        d0, d1, d2 = depth
        self.depth = tuple(depth)
        # dec_depth=(full-res, H/2): decoder 블록 수를 encoder 와 분리한다.
        # None 이면 encoder 를 미러링한다(기존 동작과 동일한 모듈 이름·순서라
        # 기존 체크포인트가 그대로 로드된다). 0 이면 그 해상도의 skip concat 과
        # ResBlock 을 통째로 생략한다 — encoder 쪽 depth 0 과 달리 decoder 는
        # 융합 블록 1개가 강제로 남던 구조였는데, 0 을 주면 그것까지 없앤다.
        dd0, dd1 = (d0, d1) if dec_depth is None else dec_depth
        self.dec_depth = (dd0, dd1)
        R = lambda ch, out=None: ResBlock(ch, dropout, out_channels=out, norm=norm)
        A = lambda: AttnBlock(C, num_heads, pan_channel=in_channels, ms_channel=out_channels,
                              mlp_ratio=mlp_ratio, ks=ks, ka=ka, pan_branch=cm3a_pan_branch)

        # 논문: "Pθ takes as input the channel-wise concatenation of I_pan and I_lrms" -> 9ch.
        # in_mode="released" 는 배포 코드와 같은 11ch (PAN, up(LPAN), PAN-up(LPAN), up(MS)).
        self.in_mode = in_mode
        n_in = (in_channels + out_channels) if in_mode == "paper" else (in_channels * 3 + out_channels)
        self.input = nn.Conv2d(n_in, C, 3, padding=1)
        self.encoder1 = nn.ModuleList([R(C) for _ in range(d0)])
        self.down1 = DownConv(C, out_channels=C)
        self.encoder2 = nn.ModuleList([R(C) for _ in range(d1)])
        self.down2 = DownConv(C, out_channels=C)
        self.middle = nn.ModuleList([R(C, C) for _ in range(d2)])
        self.up2 = UpConv(C, out_channels=C)
        self.decoder2 = nn.ModuleList(
            ([R(2 * C, C)] + [R(C, C) for _ in range(dd1 - 1)]) if dd1 > 0 else [])
        self.up1 = UpConv(C, out_channels=C)
        self.decoder1 = nn.ModuleList(
            ([R(2 * C, C)] + [R(C, C) for _ in range(dd0 - 1)]) if dd0 > 0 else [])
        self.output = nn.Sequential(_norm(norm, C), nn.SiLU(),
                                    zero_module(nn.Conv2d(C, out_channels, 3, padding=1)))
        # attn_locations 가 있으면 위치를 직접 고른다 ("enc","btl","dec" 의 부분집합).
        # 없으면 기존 n_attn 규칙 (1=enc, 2=enc+btl, 3=전부) 을 따른다 — 이전 meta 와 호환.
        if attn_locations is None:
            attn_locations = ("enc", "btl", "dec")[:n_attn]
        self.attn_locations = tuple(attn_locations)
        self.cond2_e = A() if "enc" in self.attn_locations else None
        self.cond_bot = A() if "btl" in self.attn_locations else None
        self.cond2_d = A() if "dec" in self.attn_locations else None
        self.initialize_weights()

    def initialize_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic)
        # 논문 Sec 3.3 의 LocalAttn 이 성립하도록 shift kernel 로 고정한다
        # (배포 코드는 reset_parameters() 호출부가 없어 랜덤 초기화로 학습된다).
        for m in self.modules():
            if isinstance(m, CMAAA):
                m.reset_parameters()
                if m.dep_conv.bias is not None:
                    nn.init.constant_(m.dep_conv.bias, 0)
                    m.dep_conv.bias.requires_grad_(False)

    def forward(self, pan, lpan, ms, s):
        I = lambda t, k: F.interpolate(t, scale_factor=k, mode="bicubic")
        ms_u = I(ms, 4)
        if self.in_mode == "paper":
            x = self.input(torch.cat((pan, ms_u), dim=1))
        else:
            lpan_u = I(lpan, 4)
            x = self.input(torch.cat((pan, lpan_u, pan - lpan_u, ms_u), dim=1))
        for b in self.encoder1:
            x = b(x, s)
        skip1 = x
        x = self.down1(x)
        for b in self.encoder2:
            x = b(x, s)
        if self.cond2_e is not None:
            x = self.cond2_e(x, I(ms, 2), I(lpan, 2), I(pan, 1 / 2), s)
        skip2 = x
        x = self.down2(x)
        for b in self.middle:
            x = b(x, s)
        if self.cond_bot is not None:
            x = self.cond_bot(x, ms, lpan, I(pan, 1 / 4), s)
        x = self.up2(x)
        if len(self.decoder2) > 0:
            x = torch.cat((x, skip2), dim=1)
            for b in self.decoder2:
                x = b(x, s)
        if self.cond2_d is not None:
            x = self.cond2_d(x, I(ms, 2), I(lpan, 2), I(pan, 1 / 2), s)
        x = self.up1(x)
        if len(self.decoder1) > 0:
            x = torch.cat((x, skip1), dim=1)
            for b in self.decoder1:
                x = b(x, s)
        return self.output(x)
