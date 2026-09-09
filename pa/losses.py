"""§5 loss. 수식은 명세 그대로, 계수·ramp 는 호출부(train_pa) 가 준다."""
import math
import torch
import torch.nn.functional as F

_KX = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]]) / 32.0      # Scharr, §5.2


def scharr(x):
    """[B,C,H,W] → (gx, gy) 각 [B,C,H,W]. reflect pad, 채널별 동일 필터, 부호 유지."""
    B, C, H, W = x.shape
    kx = _KX.to(x.device, x.dtype); ky = kx.t()
    k = torch.stack((kx, ky))[:, None]                                   # [2,1,3,3]
    xp = F.pad(x.reshape(B * C, 1, H, W), (1, 1, 1, 1), mode="reflect")
    g = F.conv2d(xp, k)                                                  # [B*C,2,H,W]
    g = g.reshape(B, C, 2, H, W)
    return g[:, :, 0], g[:, :, 1]


def output_edge_loss(y_hat, y):
    """§5.2 L_edge: band 별 signed Scharr 차의 L1, 미분 경계 1px 제외."""
    gx_h, gy_h = scharr(y_hat); gx, gy = scharr(y)
    s = (slice(None), slice(None), slice(1, -1), slice(1, -1))
    return 0.5 * (gx_h[s] - gx[s]).abs().mean() + 0.5 * (gy_h[s] - gy[s]).abs().mean()


def gaussian_kernel_1d(sigma, device, dtype):
    r = int(math.ceil(3.0 * sigma))
    ax = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    return k / k.sum(), r


def common_blur(x, sigma):
    """§5.3(a) 공통 Gaussian G_r (separable, reflect pad). WV3 r=4 → sigma 2, radius 6, 13×13."""
    k, r = gaussian_kernel_1d(sigma, x.device, x.dtype)
    B, C, H, W = x.shape
    xp = F.pad(x.reshape(B * C, 1, H, W), (r, r, r, r), mode="reflect")
    xp = F.conv2d(xp, k.view(1, 1, 1, -1))
    xp = F.conv2d(xp, k.view(1, 1, -1, 1))
    return xp.reshape(B, C, H, W)


def orientation_tensor(gx, gy, eta2):
    """§5.3(c) Q(g;η²) = (gx², √2 gx gy, gy²) / (gx²+gy²+η²). Q(g)=Q(−g). eta2 [B,1,1,1]."""
    den = gx * gx + gy * gy + eta2
    return torch.stack((gx * gx, math.sqrt(2.0) * gx * gy, gy * gy), dim=1) / den[:, None]


def _zn_with(x, mu, sd):
    return (x - mu) / sd


def direct_geometry_loss(pan_warped, pan, gt, sigma, margin):
    """§5.3 L_geo. pan_warped=P̃ (미분 가능), pan=원본 P (척도용, detach), gt=Y (loss 전용).
    반환 (loss [scalar], info dict: weight_sum, zero_weight_samples)."""
    B, _, H, W = pan.shape
    pf = pan.float().detach(); pw = pan_warped.float(); y = gt.float().detach()
    mu = pf.mean(dim=(2, 3), keepdim=True); sd = torch.sqrt(pf.var(dim=(2, 3), keepdim=True, unbiased=False) + 1e-6)
    zp_w = _zn_with(pw, mu, sd)                                          # P̃ 를 P 의 통계로 표준화 (§5.3a)
    zp = _zn_with(pf, mu, sd)
    iy = y.mean(dim=1, keepdim=True)
    zy = (iy - iy.mean(dim=(2, 3), keepdim=True)) / torch.sqrt(iy.var(dim=(2, 3), keepdim=True, unbiased=False) + 1e-6)
    gx_w, gy_w = scharr(common_blur(zp_w, sigma))
    gx_p, gy_p = scharr(common_blur(zp, sigma))
    gx_y, gy_y = scharr(common_blur(zy, sigma))
    V = (slice(None), slice(None), slice(margin, H - margin), slice(margin, W - margin))
    eta_p2 = ((gx_p[V] ** 2 + gy_p[V] ** 2).mean(dim=(1, 2, 3), keepdim=True) + 1e-12).detach()   # unwarped P 의 energy (§5.3b)
    ey2 = gx_y[V] ** 2 + gy_y[V] ** 2
    eta_y2 = (ey2.mean(dim=(1, 2, 3), keepdim=True) + 1e-12).detach()
    q_p = orientation_tensor(gx_w[V], gy_w[V], eta_p2)                   # [B,3,1,h,w]
    q_y = orientation_tensor(gx_y[V], gy_y[V], eta_y2)
    w_y = (ey2 / (ey2 + eta_y2)).detach()                                # [B,1,h,w]
    d2 = ((q_p - q_y) ** 2).sum(dim=1)                                   # [B,1,h,w]
    wsum = w_y.flatten(1).sum(dim=1)                                     # [B]
    per = (w_y * d2).flatten(1).sum(dim=1) / (wsum + 1e-8)
    zero = wsum <= 1e-8
    per = torch.where(zero, torch.zeros_like(per), per)                  # GT gradient 가 없는 sample 은 0 (§5.3c)
    return per.mean(), dict(weight_sum=float(wsum.mean()), zero_weight_samples=int(zero.sum()))


def geometry_support_margin(sigma, margin):
    """L_geo 가 실제로 참조하는 P̃ 영역의 margin. V_g=[margin:H-margin] 의 값은 Gaussian(radius ceil(3σ)) + Scharr(1) 만큼 바깥 P̃ 에
    의존하므로, warp support 는 margin − (r_gauss + 1) 안쪽까지 유효해야 한다 (검토 지적 1: σ=2, margin 11 → [4:60])."""
    r = int(math.ceil(3.0 * sigma)) + 1
    assert margin > r, f"geometry margin {margin} 이 필터 support {r} 보다 커야 한다"
    return margin - r


def lambda_ramp(base, step, ramp=5000):
    return base * min(step / float(ramp), 1.0) if ramp > 0 else base
