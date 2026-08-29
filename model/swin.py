# --------------------------------------------------------
# 표준 Swin Transformer block (W-MSA / SW-MSA) — 직접 구현, 외부 의존성 없음.
# 계획: research_log/2026-08-29_swin-24h-plan-v2.md §1.2 (원안 §1.2 승계)
#
# 설계 결정
#   - mode(MARs) 조건화 없음. mode 주입은 ResBlock 의 ModeModulation(Eq 6)이
#     담당하므로 Swin 은 "표준 Swin Transformer" 그대로 둔다 — 비교 서사에도 유리.
#   - 입출력은 conv 형식 (B, C, H, W). 내부에서 (B, H, W, C)로 바꿔 처리한다.
#   - H, W 가 window 배수가 아니면 우하단 zero-pad 후 잘라낸다. 이 저장소의
#     격자(학습 H/4=16², H/2=32², RR 64², FR 128·256²)는 전부 8의 배수라
#     실제로는 padding 경로를 타지 않지만, 안전장치로 둔다.
#   - SW-MSA attention mask 는 (H, W, device) 별로 만들어 캐시한다.
#     학습(16²)·검증(64²)·FR(128²)이 번갈아 와도 재계산하지 않는다.
#   - relative position bias: trunc_normal(0.02) 초기화. PANCrafterPaper 의
#     initialize_weights(_basic)는 nn.Linear 만 만지므로 이 테이블은 보존된다.
#
# 파라미터 (dim=128, heads=4, window=8, mlp_ratio=2 기준, 블록당):
#   qkv 49,536 + proj 16,512 + MLP 65,920 + LN 512 + rel-pos 900 = 133,380 (0.1334M)
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, w):
    """(B, H, W, C) -> (B * H/w * W/w, w*w, C). H, W 는 w 의 배수여야 한다."""
    B, H, W, C = x.shape
    x = x.view(B, H // w, w, W // w, w, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, w * w, C)


def window_reverse(windows, w, H, W, B):
    """window_partition 의 역변환."""
    x = windows.view(B, H // w, W // w, w, w, -1)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)


class WindowAttention(nn.Module):
    """window 내부 MSA + relative position bias."""

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim, self.w, self.heads = dim, window_size, num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rel_bias = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords = coords.flatten(1)                                # (2, w²)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0)  # (w², w², 2)
        rel[..., 0] += window_size - 1
        rel[..., 1] += window_size - 1
        rel[..., 0] *= 2 * window_size - 1
        self.register_buffer("rel_index", rel.sum(-1), persistent=False)
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

    def forward(self, x, mask=None):
        # x: (B_, N, C), N = w². mask: (nW, N, N) 또는 None
        B_, N, C = x.shape
        qkv = (self.qkv(x).reshape(B_, N, 3, self.heads, C // self.heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)             # (B_, heads, N, N)
        bias = (self.rel_bias[self.rel_index.view(-1)]
                .view(N, N, -1).permute(2, 0, 1))
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = (attn.view(B_ // nW, nW, self.heads, N, N)
                    + mask.unsqueeze(0).unsqueeze(2))
            attn = attn.view(-1, self.heads, N, N)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinBlock(nn.Module):
    """(S)W-MSA 한 블록. shift=0 이면 W-MSA, window//2 이면 SW-MSA.

    쌍(pair) 단위 사용이 기본이다: swin_pair() 또는 홀수 index 에 shift 를 준다.
    """

    def __init__(self, dim, num_heads=4, window_size=8, shift=0, mlp_ratio=2.0):
        super().__init__()
        assert 0 <= shift < window_size
        self.w, self.shift = window_size, shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))
        self._masks = {}                     # (Hp, Wp, device) -> mask. state_dict 밖

    def _attn_mask(self, Hp, Wp, device):
        if self.shift == 0:
            return None
        key = (Hp, Wp, str(device))
        if key not in self._masks:
            img = torch.zeros(1, Hp, Wp, 1, device=device)
            cnt = 0
            for hs in (slice(0, -self.w), slice(-self.w, -self.shift),
                       slice(-self.shift, None)):
                for ws in (slice(0, -self.w), slice(-self.w, -self.shift),
                           slice(-self.shift, None)):
                    img[:, hs, ws, :] = cnt
                    cnt += 1
            mw = window_partition(img, self.w).squeeze(-1)        # (nW, N)
            mask = mw.unsqueeze(1) - mw.unsqueeze(2)
            mask = (mask.masked_fill(mask != 0, -100.0)
                        .masked_fill(mask == 0, 0.0))
            self._masks[key] = mask
        return self._masks[key]

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        xf = x.permute(0, 2, 3, 1)
        pad_h = (self.w - H % self.w) % self.w
        pad_w = (self.w - W % self.w) % self.w
        if pad_h or pad_w:
            xf = F.pad(xf, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = xf.shape[1], xf.shape[2]
        shortcut = xf
        h = self.norm1(xf)
        if self.shift:
            h = torch.roll(h, (-self.shift, -self.shift), dims=(1, 2))
        wins = window_partition(h, self.w)
        wins = self.attn(wins, self._attn_mask(Hp, Wp, x.device))
        h = window_reverse(wins, self.w, Hp, Wp, B)
        if self.shift:
            h = torch.roll(h, (self.shift, self.shift), dims=(1, 2))
        xf = shortcut + h
        xf = xf + self.mlp(self.norm2(xf))
        if pad_h or pad_w:
            xf = xf[:, :H, :W, :]
        return xf.permute(0, 3, 1, 2).contiguous()


def swin_blocks(n, dim, num_heads=4, window_size=8, mlp_ratio=2.0):
    """W→SW 교대 배열 n 개. 짝수 index = W-MSA(shift 0), 홀수 = SW-MSA(shift w/2)."""
    return nn.ModuleList([
        SwinBlock(dim, num_heads, window_size,
                  (window_size // 2) if i % 2 else 0, mlp_ratio)
        for i in range(n)])
