"""Local-statistics loss reference for the s2 W112-D124 experiment plan.

This file does NOT implement the W112 U-Net, PAN aligner, data loader, trainer,
covariance uncertainty estimator, or server launcher. It extracts output image
statistics and reuses the separately supplied GTAnchoredReconstructionKD.

Statistics are population moments (correction=0), evaluated at identical valid
window centers. Input order is NCHW; no spectral averaging precedes gradients.
The output descriptor is N,D,H-2*m,W-2*m, with m=1+(window-1)//2 for every kind.
Covariance descriptors retain all matrix entries; the criterion uses their mean
absolute difference, NOT a Frobenius distance or Gaussian KL.

Run: python pan_s2_w112_stat_kd_reference.py
Requires the original pan_gt_anchored_kd.py in the same directory / Python path.
"""
from __future__ import annotations

from typing import Literal
import unittest
import torch
from torch import Tensor
import torch.nn.functional as F
from pan_gt_anchored_kd import GTAnchoredReconstructionKD

Kind = Literal['image_var', 'grad_var', 'grad_cov', 'spectral_cov']
KINDS = {'image_var', 'grad_var', 'grad_cov', 'spectral_cov'}


def _pool(x: Tensor, window: int) -> Tensor:
    return F.avg_pool2d(x, kernel_size=window, stride=1, padding=0)


def _center_for_numerics(x: Tensor) -> Tensor:
    # A constant spatial offset does not alter a centered local covariance.
    return x - x.mean(dim=(-2, -1), keepdim=True).detach()


def _variance(x: Tensor, window: int) -> Tensor:
    z = _center_for_numerics(x)
    mu = _pool(z, window)
    return (_pool(z.square(), window) - mu.square()).clamp_min(0)


def _covariance(x: Tensor, window: int) -> Tensor:
    """Return [B,C,C,Hv,Wv]; process one row of products at a time."""
    z = _center_for_numerics(x)
    mu = _pool(z, window)
    rows = [_pool(z[:, i:i+1] * z, window) - mu[:, i:i+1] * mu
            for i in range(z.shape[1])]
    cov = torch.stack(rows, dim=1)
    # Clamp tiny negative diagonal roundoff only. Negative off-diagonals are valid.
    c = z.shape[1]
    diag = torch.eye(c, dtype=torch.bool, device=z.device)[None, :, :, None, None]
    return torch.where(diag, cov.clamp_min(0), cov)


def statistic_map(image: Tensor, kind: Kind = 'grad_var', window: int = 5) -> Tensor:
    """Extract differentiable statistics; no Teacher/GT semantics in this function.

    image_var: C components.
    grad_var: 2*C components, [all dy bands, all dx bands].
    grad_cov: 4*C components, [cyy, cyx, cxy, cxx] per spectral band.
    spectral_cov: C*C row-major components.
    """
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
        x = image.to(dtype=dtype)
        b, c, _, _ = x.shape
        if kind in {'image_var', 'spectral_cov'}:
            # Match the centers of a radius-1 derivative plus a valid window.
            interior = x[..., 1:-1, 1:-1]
            if kind == 'image_var':
                return _variance(interior, window)
            cv = _covariance(interior, window)
            return cv.reshape(b, c*c, cv.shape[-2], cv.shape[-1])
        kx = x.new_tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]]) / 32.
        ky = kx.T.contiguous()
        gx = F.conv2d(x, kx[None, None].repeat(c, 1, 1, 1), groups=c)
        gy = F.conv2d(x, ky[None, None].repeat(c, 1, 1, 1), groups=c)
        if kind == 'grad_var':
            return _variance(torch.cat((gy, gx), dim=1), window)
        entries = []
        for band in range(c):
            xy = torch.cat((gy[:, band:band+1], gx[:, band:band+1]), dim=1)
            cv = _covariance(xy, window)
            entries.append(cv.reshape(b, 4, cv.shape[-2], cv.shape[-1]))
        return torch.cat(entries, dim=1)


def statistic_loss(student: Tensor, teacher: Tensor, gt: Tensor, *,
                   kind: Kind, window: int, criterion: GTAnchoredReconstructionKD):
    """Teacher/GT statistics are constant targets; only Student statistics are live.

    criterion.tau must have been calibrated for this statistic, window, Teacher,
    intensity convention and ROI. Do not reuse reconstruction tau by accident.
    """
    if not (student.shape == teacher.shape == gt.shape):
        raise ValueError('Student, Teacher and GT shapes differ')
    with torch.no_grad():
        v_t = statistic_map(teacher.detach(), kind, window)
        v_g = statistic_map(gt.detach(), kind, window)
    v_s = statistic_map(student, kind, window)
    return criterion(v_s, v_t, v_g, return_maps=True)


class _Tests(unittest.TestCase):
    def setUp(self):
        self.rng = torch.Generator().manual_seed(1234)

    def _rand(self, c=3):
        return torch.rand(2, c, 13, 15, generator=self.rng, dtype=torch.float64)

    def test_shapes_share_centers(self):
        x = self._rand()
        dims = {'image_var': 3, 'grad_var': 6, 'grad_cov': 12, 'spectral_cov': 9}
        for kind, dim in dims.items():
            self.assertEqual(statistic_map(x, kind, 5).shape, (2, dim, 7, 9))

    def test_constants_zero(self):
        x = torch.full((1, 3, 13, 15), .37, dtype=torch.float64)
        for kind in KINDS:
            self.assertLess(statistic_map(x, kind).abs().max().item(), 1e-26)

    def test_population_variance_explicit(self):
        x = self._rand()
        patches = F.unfold(x[..., 1:-1, 1:-1], kernel_size=5).reshape(2, 3, 25, 7, 9)
        expected = patches.var(dim=2, correction=0)
        torch.testing.assert_close(statistic_map(x, 'image_var'), expected, rtol=1e-10, atol=1e-12)

    def test_scalar_intensity_scaling_squared(self):
        x = self._rand()
        for kind in KINDS:
            torch.testing.assert_close(statistic_map(2.3*x, kind), 2.3**2*statistic_map(x, kind),
                                       rtol=1e-9, atol=1e-11)

    def test_bandwise_brightness_offsets(self):
        x = self._rand()
        shifted = x + x.new_tensor([.2, -.4, .7])[None, :, None, None]
        for kind in KINDS:
            torch.testing.assert_close(statistic_map(shifted, kind), statistic_map(x, kind),
                                       rtol=1e-9, atol=1e-11)

    def test_grad_cov_diagonal_matches_grad_var(self):
        x = self._rand()
        gv = statistic_map(x, 'grad_var')
        gc = statistic_map(x, 'grad_cov').reshape(2, 3, 2, 2, 7, 9)
        torch.testing.assert_close(gc[:, :, 0, 0], gv[:, :3])
        torch.testing.assert_close(gc[:, :, 1, 1], gv[:, 3:])
        torch.testing.assert_close(gc[:, :, 0, 1], gc[:, :, 1, 0])

    def test_spectral_cov_symmetry_psd(self):
        x = self._rand()
        cv = statistic_map(x, 'spectral_cov').reshape(2, 3, 3, 7, 9)
        torch.testing.assert_close(cv, cv.transpose(1, 2))
        ev = torch.linalg.eigvalsh(cv.permute(0, 3, 4, 1, 2))
        self.assertGreaterEqual(ev.min().item(), -1e-12)

    def test_direct_student_gradient(self):
        x = self._rand().requires_grad_()
        for kind in KINDS:
            loss = statistic_map(x, kind).square().mean()
            grad, = torch.autograd.grad(loss, (x,), retain_graph=True)
            self.assertTrue(bool(torch.isfinite(grad).all()))
            self.assertGreater(grad.abs().max().item(), 0)

    def test_target_detach_and_live_student(self):
        s = self._rand().requires_grad_()
        t, y = self._rand().requires_grad_(), self._rand().requires_grad_()
        loss = statistic_loss(s, t, y, kind='grad_var', window=5,
                              criterion=GTAnchoredReconstructionKD(1e-3, eps=1e-8))
        loss.loss.backward()
        self.assertIsNone(t.grad)
        self.assertIsNone(y.grad)
        self.assertGreater(s.grad.abs().max().item(), 0)
        self.assertFalse(loss.maps['difficulty'].requires_grad)

    def test_exact_gt_has_zero_loss(self):
        y, t = self._rand(), self._rand()
        s = y.clone().requires_grad_()
        for kind in KINDS:
            out = statistic_loss(s, t, y, kind=kind, window=5,
                                 criterion=GTAnchoredReconstructionKD(1e-3, eps=1e-8))
            self.assertEqual(out.loss.item(), 0)

    def test_half_cast_and_finite(self):
        x = self._rand().half().requires_grad_()
        y = statistic_map(x, 'grad_var')
        self.assertEqual(y.dtype, torch.float32)
        y.sum().backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))

    def test_bad_config(self):
        x = self._rand()
        for k in [0, 2, 4, 21]:
            with self.assertRaises(ValueError):
                statistic_map(x, 'grad_var', k)
        with self.assertRaises(ValueError):
            statistic_map(x, 'unknown', 5)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
