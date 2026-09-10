"""출력 통계(국소 이차 모멘트) 표현과 H/T/FIX/WH/AD 통계 loss (plan §8–§9; reference pan_s2_w112_stat_kd_reference.py).

statistic_map(image, kind, window): population moment(correction=0), 유효 창 중심 동일 — 출력 [N, D, H−2m, W−2m], m = 1 + (k−1)/2.
  image_var    : C           (STAT-IV)
  grad_var     : 2C  [dy 전 밴드, dx 전 밴드]   (STAT-GV, Scharr/32 valid conv)
  grad_cov     : 4C  [cyy, cyx, cxy, cxx] / band  (STAT-GC, 중심화 covariance, 대칭 off-diagonal 두 번 등장)
  spectral_cov : C·C row-major                     (STAT-SC)
공분산 비교는 모든 원소 평균 L1 — Frobenius / Gaussian KL 이 아니다. 대각의 음수 roundoff 만 clamp, off-diagonal 음수는 보존.
stat_term(): H = <E_S^V> · T = <E_ST^V> (hard 없음) · FIX = criterion fixed_kd · WH = hard_only · AD = adaptive (criterion 은 tau_V·eps_V 로 calibration)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from kdv.losses_rec import GTAnchoredReconstructionKD, ReconstructionKDResult

KINDS = ('image_var', 'grad_var', 'grad_cov', 'spectral_cov')
STAT_MODES = ('H', 'T', 'FIX', 'WH', 'AD')
MODE_TO_CRITERION = {'H': 'gt', 'FIX': 'fixed_kd', 'WH': 'hard_only', 'AD': 'adaptive'}     # T 는 criterion 을 쓰지 않는다


def stat_margin(window: int) -> int:
    return 1 + (window - 1) // 2


def _pool(x: Tensor, window: int) -> Tensor:
    return F.avg_pool2d(x, kernel_size=window, stride=1, padding=0)


def _center_for_numerics(x: Tensor) -> Tensor:
    return x - x.mean(dim=(-2, -1), keepdim=True).detach()       # 공간 상수 제거는 centered moment 를 바꾸지 않는다


def _variance(x: Tensor, window: int) -> Tensor:
    z = _center_for_numerics(x); mu = _pool(z, window)
    return (_pool(z.square(), window) - mu.square()).clamp_min(0)


def _covariance(x: Tensor, window: int) -> Tensor:
    """[B,C,C,Hv,Wv]; band row 단위로 곱을 pooling (전체 unfold 를 만들지 않는다)."""
    z = _center_for_numerics(x); mu = _pool(z, window)
    rows = [_pool(z[:, i:i + 1] * z, window) - mu[:, i:i + 1] * mu for i in range(z.shape[1])]
    cov = torch.stack(rows, dim=1)
    c = z.shape[1]
    diag = torch.eye(c, dtype=torch.bool, device=z.device)[None, :, :, None, None]
    return torch.where(diag, cov.clamp_min(0), cov)


def statistic_map(image: Tensor, kind: str = 'grad_var', window: int = 5) -> Tensor:
    if kind not in KINDS:
        raise ValueError(f'Unsupported statistic kind: {kind}')
    if image.ndim != 4 or not image.is_floating_point():
        raise ValueError('Expected floating point NCHW image')
    if any(v == 0 for v in image.shape):
        raise ValueError('Empty image dimension')
    if not isinstance(window, int) or window < 3 or window % 2 == 0:
        raise ValueError('window must be an odd integer >= 3')
    if min(image.shape[-2:]) < window + 2:
        raise ValueError('Image too small for Scharr support and valid windows')
    dtype = torch.float64 if image.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=image.device.type, enabled=False):
        x = image.to(dtype=dtype); b, c, _, _ = x.shape
        if kind in ('image_var', 'spectral_cov'):
            interior = x[..., 1:-1, 1:-1]                        # radius-1 미분 + 유효 창과 같은 중심
            if kind == 'image_var':
                return _variance(interior, window)
            cv = _covariance(interior, window)
            return cv.reshape(b, c * c, cv.shape[-2], cv.shape[-1])
        kx = x.new_tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]]) / 32.
        ky = kx.T.contiguous()
        gx = F.conv2d(x, kx[None, None].repeat(c, 1, 1, 1), groups=c)
        gy = F.conv2d(x, ky[None, None].repeat(c, 1, 1, 1), groups=c)
        if kind == 'grad_var':
            return _variance(torch.cat((gy, gx), dim=1), window)
        entries = []
        for band in range(c):
            xy = torch.cat((gy[:, band:band + 1], gx[:, band:band + 1]), dim=1)
            cv = _covariance(xy, window)
            entries.append(cv.reshape(b, 4, cv.shape[-2], cv.shape[-1]))
        return torch.cat(entries, dim=1)


def statistic_loss(student: Tensor, teacher: Tensor, gt: Tensor, *, kind: str, window: int, criterion: GTAnchoredReconstructionKD):
    """참조 wrapper: Teacher/GT 통계는 상수 target, Student 통계만 live. criterion.tau 는 이 통계·창·Teacher 로 calibration 된 값이어야 한다."""
    if not (student.shape == teacher.shape == gt.shape):
        raise ValueError('Student, Teacher and GT shapes differ')
    with torch.no_grad():
        v_t = statistic_map(teacher.detach(), kind, window); v_g = statistic_map(gt.detach(), kind, window)
    v_s = statistic_map(student, kind, window)
    return criterion(v_s, v_t, v_g, return_maps=True)


def stat_term(student: Tensor, teacher, gt: Tensor, *, kind: str, window: int, mode: str, criterion=None, return_maps: bool = False) -> ReconstructionKDResult:
    """통계 항 L_V (plan §9.1). teacher=None 은 H 만 허용. 반환은 ReconstructionKDResult (T: hard=0, soft=loss)."""
    if mode not in STAT_MODES:
        raise ValueError(f'stat mode {mode}')
    if teacher is None and mode != 'H':
        raise ValueError(f'STAT-{mode} 는 Teacher 통계가 필요하다')
    v_s = statistic_map(student, kind, window)
    with torch.no_grad():
        v_g = statistic_map(gt.detach(), kind, window)
        v_t = statistic_map(teacher.detach(), kind, window) if teacher is not None else None
    if mode == 'T':
        soft = (v_s - v_t).abs().mean(); zero = soft.detach() * 0
        with torch.no_grad():
            st = {'plain_gt_l1': (v_s - v_g).abs().mean(), 'plain_teacher_l1': soft.detach(), 'teacher_gt_l1': (v_t - v_g).abs().mean()}
        return ReconstructionKDResult(soft, zero, soft, st, None)
    if teacher is None:                                              # H without Teacher: <E_S^V>
        hard = (v_s - v_g).abs().mean(); zero = hard.detach() * 0
        return ReconstructionKDResult(hard, hard, zero, {'plain_gt_l1': hard.detach()}, None)
    if criterion is None or criterion.mode != MODE_TO_CRITERION[mode]:
        raise ValueError(f'STAT-{mode} 는 criterion mode {MODE_TO_CRITERION[mode]} 가 필요하다 (현재 {getattr(criterion, "mode", None)})')
    return criterion(v_s, v_t, v_g, return_maps=return_maps)


def positive_median(v: Tensor):
    """calibration v_scale = median(|S_V(GT)|) over strictly positive finite entries (plan §9.2). 없으면 None."""
    a = v.detach().abs().flatten()
    a = a[torch.isfinite(a) & (a > 0)]
    return float(a.median()) if a.numel() else None
