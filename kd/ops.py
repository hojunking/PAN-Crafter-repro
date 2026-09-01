# --------------------------------------------------------
# KD·mutual 공통 연산자 — 명세 §6 (research_log/s1_mutual_and_kd_implementation_spec.md)
#
# MTF 주의: DLPan-Toolbox(GPL-3.0)의 센서 MTF 커널을 학습 경로에 끌어오지 않는다.
# 대신 저장소에서 검증된 lpan 레시피(가우시안 σ=1.98·41×41·4배 데시메이션,
# tools/repair_lpan.py)와 같은 형태의 clean-room 가우시안 근사를 쓴다.
# 밴드별 σ 는 config 로 열어 두었다 — 공개 Nyquist gain 상수로 세분할 수 있다.
# 커널 크기·패딩은 기존 레시피와 동일하게 41×41·replicate (경계 일치).
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel(sigma, ksize):
    ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    k2 = torch.outer(g, g)
    return k2 / k2.sum()


class MTFDownsampler(nn.Module):
    """HRMS -> LRMS scale 열화. 밴드별 가우시안 blur + stride 데시메이션.

    requires_grad=False 커널 — gradient 는 입력 prediction 으로만 흐른다 (§6.1).
    """

    def __init__(self, bands=8, scale=4, sigma=1.98, ksize=41, offset=None):
        super().__init__()
        self.bands, self.scale = bands, scale
        self.offset = (scale // 2) if offset is None else offset
        sigmas = [sigma] * bands if isinstance(sigma, (int, float)) else list(sigma)
        assert len(sigmas) == bands
        k = torch.stack([gaussian_kernel(s, ksize) for s in sigmas])   # (B, k, k)
        self.register_buffer("kernel", k.unsqueeze(1), persistent=False)  # (B,1,k,k)
        self.pad = ksize // 2

    def forward(self, x_hr):
        # x_hr: (B, C, H, W) -> (B, C, H/s, W/s)
        k = self.kernel.to(dtype=x_hr.dtype)
        x = F.pad(x_hr, [self.pad] * 4, mode="replicate")
        x = F.conv2d(x, k, groups=self.bands)
        return x[..., self.offset::self.scale, self.offset::self.scale]


class AbsoluteGradient(nn.Module):
    """Sobel 절대 gradient 크기 (§6.2). 방향이 아니라 크기를 비교한다."""

    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        self.register_buffer("sx", sx.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sy", sx.t().reshape(1, 1, 3, 3), persistent=False)

    def forward(self, x):
        # x: (B, 1, H, W) — luminance 는 호출부에서 mean(dim=1) 로 만든다
        sx = self.sx.to(dtype=x.dtype)
        sy = self.sy.to(dtype=x.dtype)
        x = F.pad(x, [1] * 4, mode="replicate")
        gx = F.conv2d(x, sx)
        gy = F.conv2d(x, sy)
        return torch.sqrt(gx.square() + gy.square() + 1e-6)


def robust_normalize_01(v, q_lo=0.05, q_hi=0.95):
    """샘플별 분위수 정규화 [0,1] (§6.3). 입력은 detach 된 상태를 기대한다."""
    B = v.shape[0]
    flat = v.reshape(B, -1)
    lo = torch.quantile(flat, q_lo, dim=1).view(B, 1, 1, 1)
    hi = torch.quantile(flat, q_hi, dim=1).view(B, 1, 1, 1)
    return ((v - lo) / (hi - lo + 1e-6)).clamp(0, 1)


class LocalVarianceMap(nn.Module):
    """GT 국소 분산 map (§6.3). gradient 를 받지 않는다 (detach 출력)."""

    def __init__(self, kernel_size=5):
        super().__init__()
        self.k = kernel_size

    @torch.no_grad()
    def forward(self, gt):
        # gt: (B, C, H, W) -> (B, 1, H, W) in [0,1]
        z = (gt - gt.mean(dim=(2, 3), keepdim=True)) / (gt.std(dim=(2, 3), keepdim=True) + 1e-6)
        k, p = self.k, self.k // 2
        mean = F.avg_pool2d(z, k, stride=1, padding=p)
        mean2 = F.avg_pool2d(z * z, k, stride=1, padding=p)
        var = (mean2 - mean * mean).clamp_min(0).mean(dim=1, keepdim=True)
        return robust_normalize_01(var)


def build_shift_candidates(lrms, radius):
    """shifted LRMS 후보 (§7.2). (B,C,h,w) -> (B,C,K,h,w), K=(2r+1)²."""
    r = radius
    B, C, h, w = lrms.shape
    padded = F.pad(lrms, [r] * 4, mode="reflect")
    cands = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            cands.append(padded[..., r + dy:r + dy + h, r + dx:r + dx + w])
    return torch.stack(cands, dim=2)


def mean_normalize(w):
    """weight map 평균 1 정규화 (§8.4) — loss scale 변화를 분리한다."""
    return w / (w.mean(dim=(-2, -1), keepdim=True) + 1e-6)


def ramp_then_decay(step, start, full, decay, total=50_000, max_w=1.0):
    """0→ramp→plateau→decay 스케줄 (§13/§22). max_w 를 곱해 쓴다."""
    if step < start:
        return 0.0
    if step < full:
        return max_w * (step - start) / max(1, full - start)
    if step < decay:
        return max_w
    return max_w * max(0.0, (total - step)) / max(1, total - decay)


def local_variance(x, k):
    """window k 의 band-mean 국소 분산: (1/C)Σ_c [A_k(x_c²) − A_k(x_c)²].

    s2 계획 §6. 입력 (B,C,H,W) -> (B,1,H,W).
    """
    p = k // 2
    m = F.avg_pool2d(x, k, stride=1, padding=p, count_include_pad=False)
    m2 = F.avg_pool2d(x * x, k, stride=1, padding=p, count_include_pad=False)
    return (m2 - m * m).clamp_min(0).mean(dim=1, keepdim=True)


def multiscale_variance(x, weights=((3, 0.5), (5, 0.3), (9, 0.2))):
    """V(R) = 0.5·V_3 + 0.3·V_5 + 0.2·V_9 (s2 계획 §6)."""
    out = None
    for k, w in weights:
        v = local_variance(x, k) * w
        out = v if out is None else out + v
    return out


def squash_variance(v, kappa):
    """Ṽ = V/(V+κ) — 학습셋에서 고정한 κ 로 두 map 을 같은 스케일에 둔다."""
    return v / (v + kappa)
