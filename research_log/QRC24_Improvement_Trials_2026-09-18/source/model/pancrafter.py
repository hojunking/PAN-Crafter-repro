# --------------------------------------------------------
# PAN-Crafter: Learning Modality-Consistent Alignment for PAN-Sharpening
# Copyright (c) 2025 Jeonghyeok Do, Sungpyo Kim, Geunhyuk Youk, Jaehyup Lee†, and Munchurl Kim†
#
# This code is released under the MIT License (see LICENSE file for details).
#
# This software is licensed for **non-commercial research and educational use only**.
# For commercial use, please contact: mkimee@kaist.ac.kr
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import Mlp


''' Cross-Modality Alignment-Aware Attention (CM3A) '''
class CMAAA(nn.Module):
    def __init__(self, dim, num_heads=8, pan_channel=1, ms_channel=8, pan_ks=3, ms_ks=3, ka=3, qkv_bias=False, qk_norm=False,
                 attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm, fix_key_alias=False,
                 pan_branch=True, use_k_pan=True, v_pan_hf=True):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.fix_key_alias = fix_key_alias
        # 서브모듈 제거 실험용 (전부 True 가 배포본 동작)
        #   pan_branch=False : PAN 브랜치 전체 제거 -> MS local self-attention 만
        #   use_k_pan=False  : k_pan conv 제거. 배포본(fix_key_alias=False)에서는 출력이 버려지므로 무손실
        #   v_pan_hf=False   : v_pan 입력에서 고주파 채널 (pan - lpan) 제거
        self.pan_branch, self.use_k_pan, self.v_pan_hf = pan_branch, use_k_pan, v_pan_hf
        self.num_heads = num_heads
        self.pan_channel = pan_channel
        self.ms_channel = ms_channel
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pan_ks = pan_ks
        self.ms_ks = ms_ks
        pan_pw = pan_ks // 2
        ms_pw = ms_ks // 2
        self.ka = ka

        self.dep_conv = nn.Conv2d(self.head_dim, self.ka * self.ka * self.head_dim, kernel_size=self.ka,
                                  bias=True, groups=self.head_dim, padding=self.ka // 2)

        v_in = pan_channel * (3 if v_pan_hf else 2)
        self.q = nn.Conv2d(dim + ms_channel, dim, kernel_size=pan_ks, padding=pan_pw, bias=qkv_bias)
        self.k_pan = (nn.Conv2d(dim + pan_channel, dim, kernel_size=pan_ks, padding=pan_pw, bias=qkv_bias)
                      if (pan_branch and use_k_pan) else None)
        self.v_pan = (nn.Conv2d(dim + v_in, dim, kernel_size=pan_ks, padding=pan_pw, bias=qkv_bias)
                      if pan_branch else None)
        self.kv_ms = nn.Conv2d(dim + ms_channel, dim * 2, kernel_size=ms_ks, padding=ms_pw, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm_pan = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm_ms = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.proj_pan = nn.Conv2d(dim, dim, kernel_size=1, bias=True) if pan_branch else None
        self.proj_ms = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def reset_parameters(self):
        # shift initialization for group convolution
        kernel = torch.zeros(self.ka * self.ka, self.ka, self.ka)
        for i in range(self.ka * self.ka):
            kernel[i, i // self.ka, i % self.ka] = 1.
        kernel = kernel.unsqueeze(1).repeat(self.head_dim, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x, ms, lpan, pan, s):
        B, C, H, W = x.shape

        pan_ = lpan.repeat(1, self.ms_channel, 1, 1)
        cond = pan_ * (1.0 - s).view(-1, 1, 1, 1) + ms * s.view(-1, 1, 1, 1)
        q = self.q(torch.cat((x, cond), dim=1)).reshape(B, self.num_heads, self.head_dim, H, W)
        kv_ms = self.kv_ms(torch.cat((x, ms), dim=1)).reshape(B, 2, self.num_heads, self.head_dim, H, W)
        k_ms, v_ms = kv_ms[:, 0], kv_ms[:, 1]

        if not self.pan_branch:
            # MS local self-attention 만. PAN 브랜치(k_pan/v_pan/proj_pan) 전부 제거.
            q = (q.reshape(B, self.num_heads, self.head_dim, 1, H, W) * self.scale)
            km = self.dep_conv(k_ms.reshape(-1, self.head_dim, H, W)).reshape(
                B, self.num_heads, self.head_dim, self.ka * self.ka, H, W)
            vm = self.dep_conv(v_ms.reshape(-1, self.head_dim, H, W)).reshape(
                B, self.num_heads, self.head_dim, self.ka * self.ka, H, W)
            attn = self.attn_drop((q * km).sum(dim=2, keepdim=True).softmax(dim=3))
            out = (attn * vm).sum(dim=3).reshape(B, self.num_heads * self.head_dim, H, W)
            out = self.proj_drop(self.proj_ms(out))
            return torch.zeros_like(out), out

        v_in = (x, lpan, pan, pan - lpan) if self.v_pan_hf else (x, lpan, pan)
        v_pan = self.v_pan(torch.cat(v_in, dim=1)).reshape(B, self.num_heads, self.head_dim, H, W)
        k_pan = (self.k_pan(torch.cat((x, lpan), dim=1)).reshape(B, self.num_heads, self.head_dim, H, W)
                 if self.use_k_pan else k_ms)

        if self.fix_key_alias:
            # 논문 Eq (10)/(11): k_pan 은 PAN key, k_ms 는 MS key 여야 한다.
            q, k_pan, k_ms = self.q_norm(q.permute(0, 1, 3, 4, 2)), self.k_norm_pan(k_pan.permute(0, 1, 3, 4, 2)), self.k_norm_ms(k_ms.permute(0, 1, 3, 4, 2))
        else:
            # 저자 배포본 원본. 좌변 k_pan 이 두 번이라 k_pan 이 MS key 로 덮이고,
            # k_ms 는 재할당되지 않아 다음 줄 permute 왕복이 성립하지 않는다. KNOWN_ISSUES.md A-1 참고.
            q, k_pan, k_pan = self.q_norm(q.permute(0, 1, 3, 4, 2)), self.k_norm_pan(k_pan.permute(0, 1, 3, 4, 2)), self.k_norm_ms(k_ms.permute(0, 1, 3, 4, 2))
        q, k_pan, k_ms = q.permute(0, 1, 4, 2, 3), k_pan.permute(0, 1, 4, 2, 3), k_ms.permute(0, 1, 4, 2, 3)
        k_pan, v_pan = k_pan.reshape(-1, self.head_dim, H, W), v_pan.reshape(-1, self.head_dim, H, W)
        k_ms, v_ms = k_ms.reshape(-1, self.head_dim, H, W), v_ms.reshape(-1, self.head_dim, H, W)
        q = q.reshape(B, self.num_heads, self.head_dim, 1, H, W) * self.scale
        k_pan = self.dep_conv(k_pan).reshape(B, self.num_heads, self.head_dim, self.ka * self.ka, H, W)
        v_pan = self.dep_conv(v_pan).reshape(B, self.num_heads, self.head_dim, self.ka * self.ka, H, W)
        k_ms = self.dep_conv(k_ms).reshape(B, self.num_heads, self.head_dim, self.ka * self.ka, H, W)
        v_ms = self.dep_conv(v_ms).reshape(B, self.num_heads, self.head_dim, self.ka * self.ka, H, W)

        k = torch.stack([k_pan, k_ms], dim=1)
        v = torch.stack([v_pan, v_ms], dim=1)
        attn = (q.unsqueeze(1) * k).sum(dim=3, keepdim=True).softmax(dim=4)
        attn = self.attn_drop(attn)

        x = (attn * v).sum(dim=4).reshape(B, 2, self.num_heads * self.head_dim, H, W)
        x_pan = self.proj_drop(self.proj_pan(x[:, 0]))
        x_ms = self.proj_drop(self.proj_ms(x[:, 1]))
        return x_pan, x_ms


''' Cross-Modality Attention Block (AttnBlock) '''
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AttnBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, pan_channel=1, ms_channel=8, emb_channels=256, mlp_ratio=4.0, pan_ks=3,  ms_ks=3, ka=3,
                 fix_key_alias=False, pan_branch=True, use_k_pan=True, v_pan_hf=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = CMAAA(hidden_size, num_heads=num_heads, pan_channel=pan_channel, ms_channel=ms_channel, pan_ks=pan_ks, ms_ks=ms_ks, ka=ka, qkv_bias=True,
                          fix_key_alias=fix_key_alias, pan_branch=pan_branch,
                          use_k_pan=use_k_pan, v_pan_hf=v_pan_hf)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * hidden_size, bias=True))

    def forward(self, x, ms, lpan, pan, c, s):
        B, C, H, W = x.shape
        N = H * W
        x = x.permute(0, 2, 3, 1).reshape(B, N, -1)
        gate_ms, gate_pan = self.adaLN_modulation(c).chunk(2, dim=1)
        x_temp = self.norm1(x).reshape(B, H, W, C).permute(0, 3, 1, 2)
        x_pan, x_ms = self.attn(x_temp, ms, lpan, pan, s)
        x = x + gate_ms.unsqueeze(1) * x_ms.permute(0, 2, 3, 1).reshape(B, N, -1) + gate_pan.unsqueeze(1) * x_pan.permute(0, 2, 3, 1).reshape(B, N, -1)
        x = x + self.mlp(self.norm2(x))
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x


''' ResidualBlock (ResBlock) '''
class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class ResBlock(nn.Module):
    def __init__(self, channels, emb_channels, dropout, out_channels=None,
                 use_conv=False, use_scale_shift_norm=True):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            GroupNorm32(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1))

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels))

        self.out_layers = nn.Sequential(
            GroupNorm32(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)))

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h


''' Up-Conv '''
class UpConv(nn.Module):
    def __init__(self, channels, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv = nn.ConvTranspose2d(self.channels, self.out_channels, 2, stride=2, padding=0)

    def forward(self, x):
        return self.conv(x)


''' Down-Conv '''
class DownConv(nn.Module):
    def __init__(self, channels, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv = nn.Conv2d(self.channels, self.out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


''' PAN-Crafter '''
class PANCrafter(nn.Module):
    def __init__(self, in_channels=1, out_channels=8, hidden_size=[128, 128, 128, 128], s_embed_size=128,
                 dropout=0.0, use_scale_shift_norm=True, depth=[2, 2, 2, 2],
                 num_heads=8, mlp_ratio=4.0, pan_ks=3, ms_ks=3, ka=3,
                 fix_key_alias=False, fix_local_attn=False, cm3a_locations=None,
                 cm3a_pan_branch=True, cm3a_use_k_pan=True, cm3a_v_pan_hf=True,
                 attn_mode_gate=True, res_mode_mod=True):
        super().__init__()
        # 서브모듈 제거 실험용. 전부 기본값이면 배포본과 완전히 동일하다.
        #   attn_mode_gate=False : AttnBlock 의 mode별 gate 를 mode 무관으로 (Eq 8 의 alpha)
        #   res_mode_mod=False   : ResBlock 의 mode별 scale/shift 를 mode 무관으로 (Eq 6 의 beta,gamma)
        self.attn_mode_gate, self.res_mode_mod = attn_mode_gate, res_mode_mod
        # 사용할 CM3A 위치. 기본값(None)은 배포본과 동일한 5개 전부다.
        #   "2e"=encoder H/2, "3e"=encoder H/4, "4"=bottleneck H/8, "3d"=decoder H/4, "2d"=decoder H/2
        # 목록에서 빠진 위치는 블록을 만들지도, forward 에서 호출하지도 않는다 (통과).
        CM3A_ALL = ("2e", "3e", "4", "3d", "2d")
        loc = CM3A_ALL if cm3a_locations is None else tuple(cm3a_locations)
        unknown = set(loc) - set(CM3A_ALL)
        assert not unknown, f"알 수 없는 cm3a_locations: {sorted(unknown)}. 가능한 값 {CM3A_ALL}"
        self.cm3a_locations = loc
        # KNOWN_ISSUES.md A-1 / A-2. 둘 다 False 면 저자 배포본과 완전히 동일하게 동작한다.
        self.fix_key_alias = fix_key_alias
        self.fix_local_attn = fix_local_attn
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.s_embed_size = s_embed_size
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm
        self.depth = depth

        self.input = nn.Conv2d(in_channels * 3 + out_channels, hidden_size[0], 3, padding=1)
        self.encoder1 = nn.ModuleList(
            [ResBlock(hidden_size[0], s_embed_size, dropout, use_scale_shift_norm=use_scale_shift_norm) for _ in
             range(self.depth[0])])
        self.down1 = DownConv(hidden_size[0], out_channels=hidden_size[1])
        self.encoder2 = nn.ModuleList(
            [ResBlock(hidden_size[1], s_embed_size, dropout, use_scale_shift_norm=use_scale_shift_norm) for _ in
             range(self.depth[1])])
        self.down2 = DownConv(hidden_size[1], out_channels=hidden_size[2])
        self.encoder3 = nn.ModuleList(
            [ResBlock(hidden_size[2], s_embed_size, dropout, use_scale_shift_norm=use_scale_shift_norm) for _ in
             range(self.depth[2])])
        self.down3 = DownConv(hidden_size[2], out_channels=hidden_size[3])
        self.up3 = UpConv(hidden_size[3], out_channels=hidden_size[2])
        self.decoder3 = nn.ModuleList([ResBlock(2 * hidden_size[2], s_embed_size, dropout,
                                                  out_channels=hidden_size[2],
                                                  use_scale_shift_norm=use_scale_shift_norm)] +
                                      [ResBlock(hidden_size[2], s_embed_size, dropout, out_channels=hidden_size[2],
                                                  use_scale_shift_norm=use_scale_shift_norm) for _ in
                                       range(self.depth[2] - 1)])
        self.up2 = UpConv(hidden_size[2], out_channels=hidden_size[1])
        self.decoder2 = nn.ModuleList([ResBlock(2 * hidden_size[1], s_embed_size, dropout,
                                                  out_channels=hidden_size[1],
                                                  use_scale_shift_norm=use_scale_shift_norm)] +
                                      [ResBlock(hidden_size[1], s_embed_size, dropout, out_channels=hidden_size[1],
                                                  use_scale_shift_norm=use_scale_shift_norm) for _ in
                                       range(self.depth[1] - 1)])
        self.up1 = UpConv(hidden_size[1], out_channels=hidden_size[0])
        self.decoder1 = nn.ModuleList([ResBlock(2 * hidden_size[0], s_embed_size, dropout,
                                                  out_channels=hidden_size[0],
                                                  use_scale_shift_norm=use_scale_shift_norm)] +
                                      [ResBlock(hidden_size[0], s_embed_size, dropout, out_channels=hidden_size[0],
                                                  use_scale_shift_norm=use_scale_shift_norm) for _ in
                                       range(self.depth[0] - 1)])
        self.output = nn.Sequential(GroupNorm32(32, hidden_size[0]), nn.SiLU(),
                                    zero_module(nn.Conv2d(hidden_size[0], out_channels, 3, padding=1)))

        self.middle = nn.ModuleList([ResBlock(hidden_size[3], s_embed_size, dropout, out_channels=hidden_size[3],
                                                use_scale_shift_norm=use_scale_shift_norm) for _ in
                                     range(self.depth[3])])

        def _attn(idx, ks_pan, ks_ms):
            return AttnBlock(hidden_size[idx], num_heads, pan_channel=in_channels, ms_channel=out_channels,
                             emb_channels=s_embed_size, mlp_ratio=mlp_ratio, pan_ks=ks_pan, ms_ks=ks_ms,
                             ka=ka, fix_key_alias=fix_key_alias, pan_branch=cm3a_pan_branch,
                             use_k_pan=cm3a_use_k_pan, v_pan_hf=cm3a_v_pan_hf)

        self.cond4   = _attn(3, 1, 1)          if "4"  in loc else None
        self.cond3_e = _attn(2, pan_ks, ms_ks) if "3e" in loc else None
        self.cond2_e = _attn(1, pan_ks, ms_ks) if "2e" in loc else None
        self.cond3_d = _attn(2, pan_ks, ms_ks) if "3d" in loc else None
        self.cond2_d = _attn(1, pan_ks, ms_ks) if "2d" in loc else None

        self.s_token = nn.Parameter(torch.zeros(2, s_embed_size))

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.s_token, std=0.02)

        # Zero-out adaLN modulation layers in ResNet blocks:
        for block in self.encoder1:
            nn.init.constant_(block.emb_layers[-1].weight, 0)
            nn.init.constant_(block.emb_layers[-1].bias, 0)
        for block in self.encoder2:
            nn.init.constant_(block.emb_layers[-1].weight, 0)
            nn.init.constant_(block.emb_layers[-1].bias, 0)
        for block in self.encoder3:
            nn.init.constant_(block.emb_layers[-1].weight, 0)
            nn.init.constant_(block.emb_layers[-1].bias, 0)
        for block in self.decoder1:
            nn.init.constant_(block.emb_layers[-1].weight, 0)
            nn.init.constant_(block.emb_layers[-1].bias, 0)
        for block in self.decoder2:
            nn.init.constant_(block.emb_layers[-1].weight, 0)
            nn.init.constant_(block.emb_layers[-1].bias, 0)
        for block in self.decoder3:
            nn.init.constant_(block.emb_layers[-1].weight, 0)
            nn.init.constant_(block.emb_layers[-1].bias, 0)
        for block in self.middle:
            nn.init.constant_(block.emb_layers[-1].weight, 0)
            nn.init.constant_(block.emb_layers[-1].bias, 0)

        for blk in (self.cond4, self.cond3_e, self.cond2_e, self.cond3_d, self.cond2_d):
            if blk is not None:
                nn.init.constant_(blk.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(blk.adaLN_modulation[-1].bias, 0)

        # KNOWN_ISSUES.md A-2: 배포본은 reset_parameters() 를 호출하지 않아 dep_conv 가
        # 랜덤 초기화된 학습 가능 depthwise conv 로 남는다. 켜면 논문 Sec 3.3 의 LocalAttn
        # (k x k 이웃 gather) 이 되도록 shift kernel 로 고정한다. bias 도 함께 0 고정해야
        # 이웃값에 상수 offset 이 더해지지 않는다.
        if self.fix_local_attn:
            for m in self.modules():
                if isinstance(m, CMAAA):
                    m.reset_parameters()
                    if m.dep_conv.bias is not None:
                        nn.init.constant_(m.dep_conv.bias, 0)
                        m.dep_conv.bias.requires_grad_(False)

    def forward(self, pan, lpan, ms, s):
        # Switch condition. mode 무관으로 두려면 항상 0번 토큰을 쓴다 (파라미터 수는 그대로).
        c = self.s_token[s.long()]
        c_attn = c if self.attn_mode_gate else self.s_token[0].unsqueeze(0).expand(c.shape[0], -1)
        c_res = c if self.res_mode_mod else self.s_token[0].unsqueeze(0).expand(c.shape[0], -1)

        # Input layer
        x = self.input(torch.cat((pan, F.interpolate(ms, scale_factor=4, mode="bicubic"), F.interpolate(lpan, scale_factor=4, mode="bicubic"), pan - F.interpolate(lpan, scale_factor=4, mode="bicubic")),  dim=1))

        # Encoder blocks
        for block in self.encoder1:
            x = block(x, c_res)

        res1 = x
        x = self.down1(res1)
        for block in self.encoder2:
            x = block(x, c_res)

        if self.cond2_e is not None:
            x = self.cond2_e(x, F.interpolate(ms, scale_factor=2, mode="bicubic"), F.interpolate(lpan, scale_factor=2, mode="bicubic"), F.interpolate(pan, scale_factor=1 / 2, mode="bicubic"), c_attn, s)
        res2 = x
        x = self.down2(res2)
        for block in self.encoder3:
            x = block(x, c_res)

        if self.cond3_e is not None:
            x = self.cond3_e(x, ms, lpan, F.interpolate(pan, scale_factor=1 / 4, mode="bicubic"), c_attn, s)
        res3 = x
        x = self.down3(res3)

        # Middle blocks
        for block in self.middle:
            x = block(x, c_res)

        if self.cond4 is not None:
            x = self.cond4(x, F.interpolate(ms, scale_factor=1 / 2, mode="bicubic"), F.interpolate(lpan, scale_factor=1 / 2, mode="bicubic"), F.interpolate(pan, scale_factor=1 / 8, mode="bicubic"), c_attn, s)

        # Decoder blocks
        x = torch.cat((self.up3(x), res3), dim=1)
        for block in self.decoder3:
            x = block(x, c_res)

        if self.cond3_d is not None:
            x = self.cond3_d(x, ms, lpan, F.interpolate(pan, scale_factor=1 / 4, mode="bicubic"), c_attn, s)
        x = torch.cat((self.up2(x), res2), dim=1)
        for block in self.decoder2:
            x = block(x, c_res)

        if self.cond2_d is not None:
            x = self.cond2_d(x, F.interpolate(ms, scale_factor=2, mode="bicubic"), F.interpolate(lpan, scale_factor=2, mode="bicubic"), F.interpolate(pan, scale_factor=1 / 2, mode="bicubic"), c_attn, s)
        x = torch.cat((self.up1(x), res1), dim=1)
        for block in self.decoder1:
            x = block(x, c_res)

        # Output layer
        x = self.output(x)
        return x